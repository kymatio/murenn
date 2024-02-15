import numpy as np
import pytorch_wavelets as pytw
import torch.nn

from murenn.dtcwt.lowlevel import prep_filt
from murenn.dtcwt.transform_funcs import FWD_J1, FWD_J2PLUS, INV_J1, INV_J2PLUS


class DTCWTForward(torch.nn.Module):
    """Performs a DTCWT forward decomposition of a PyTorch tensor containing
    1-D signals, potentially multichannel.

    Args:
        level1 (str): One of 'antonini', 'legall', 'near_sym_a', 'near_sym_b'.
            Specifies the first-level biorthogonal wavelet filters.
        qshift (str): One of 'qshift_06', 'qshift_a', 'qshift_b', 'qshift_c',
            'qshift_d'.  Specifies the second-level quarter-shift filters.
        J (int): Number of levels (octaves) of decomposition. Default is 8.
        skip_hps (bools): List of bools of length J which specify whether or
            not to calculate the bandpass outputs at the given scale.
            skip_hps[0] is for the first scale. Can be a single bool in which
            case that is applied to all scales.
        include_scale (bool): If true, return the bandpass outputs. Can also be
            a list of length J specifying which lowpasses to return. I.e. if
            [False, True, True], the forward call will return the second and
            third lowpass outputs, but discard the lowpass from the first level
            transform.
        alternate_gh (bool): If True (default), alternates between filter pairs
            (h0, h1) and (g0, g1) depending on odd vs. even wavelet scale j.
            Otherwise, uses (h0, h1) only. See Selesnick et al. 2005 for details.
        padding_mode (str): One of 'zeros'(defalt), 'reflect', 'replicate', 
            and 'circular'. Padding scheme for the filters. 
        normalize (bool): If True (default), the output will be normalized by a 
            factor of 1/sqrt(2)
    """

    def __init__(
        self,
        level1="near_sym_a",
        qshift="qshift_a",
        J=8,
        skip_hps=False,
        include_scale=False,
        alternate_gh=True,
        padding_mode='zeros',
        normalize=True
    ):
        # Instantiate PyTorch NN Module
        super().__init__()

        # Store metadata
        self.level1 = level1
        self.qshift = qshift
        self.J = J
        self.alternate_gh = alternate_gh
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
        else:
            self.padding_mode = padding_mode
        self.normalize = normalize

        # Load first-level biorthogonal wavelet filters from disk.
        # h0o is the low-pass filter.
        # h1o is the high-pass filter.
        h0o, _, h1o, _ = pytw.dtcwt.coeffs._load_from_file(
            level1, ("h0o", "g0o", "h1o", "g1o")
        )
        self.register_buffer("h0o", prep_filt(h0o))
        self.register_buffer("h1o", prep_filt(h1o))

        # Load higher-level quarter-shift wavelet filters from disk.
        # h0a is the low-pass filter from tree a (real part).
        # h0b is the low-pass filter from tree b (imaginary part).
        # g0a is the low-pass dual filter from tree a (real part).
        # g0b is the low-pass dual filter from tree b (imaginary part).
        # h1a is the high-pass filter from tree a (real part).
        # h1b is the high-pass filter from tree b (imaginary part).
        # g1a is the high-pass dual filter from tree a (real part).
        # g1b is the high-pass dual filter from tree b (imaginary part).
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = pytw.dtcwt.coeffs._load_from_file(
            qshift, ("h0a", "h0b", "g0a", "g0b", "h1a", "h1b", "g1a", "g1b")
        )
        self.register_buffer("h0a", prep_filt(h0a))
        self.register_buffer("h0b", prep_filt(h0b))
        self.register_buffer("g0a", prep_filt(g0a))
        self.register_buffer("g0b", prep_filt(g0b))
        self.register_buffer("h1a", prep_filt(h1a))
        self.register_buffer("h1b", prep_filt(h1b))
        self.register_buffer("g1a", prep_filt(g1a))
        self.register_buffer("g1b", prep_filt(g1b))

        # Parse the "skip_hps" argument for skipping finest scales.
        if isinstance(skip_hps, (list, tuple, np.ndarray)):
            self.skip_hps = skip_hps
        else:
            self.skip_hps = [skip_hps,] * self.J

        # Parse the "include_scale" argument for including other low-pass
        # outputs in addition to the coarsest scale.
        if isinstance(include_scale, (list, tuple, np.ndarray)):
            self.include_scale = include_scale
        else:
            self.include_scale = [include_scale,] * self.J

    def forward(self, x):
        """Forward Dual-Tree Complex Wavelet Transform (DTCWT) of a 1-D signal.

        Args:
            x (PyTorch tensor): Input data. Should be a tensor of shape
                `(B, C, T)` where B is the batch size, C is the number of
                channels and T is the number of time samples.
                Note that T must be a multiple of 2**J, where J is the number
                of wavelet scales (see documentation of DTCWTForward constructor).

        Returns:
            (yl, yh): tuple of low-pass (yl) and band-pass (yh) coefficients.
                If include_scale is True (see DTCWTForward constructor), yl is a
                list of low-pass coefficients at all wavelet scales 1 to (J-1).
                Otherwise (default), yl is a real-valued PyTorch tensor of shape
                `(B, C, T/2**(J-1))`.
                Conversely, yh is a list of PyTorch tensors with J elements,
                containing the band-pass coefficients at all wavelets scales 1 to
                (J-1). These tensors are complex-valued and have shapes:
                `(B, C, T)`, `(B, C, T/2)`, `(B, C, T/4)`, etc."""

        # Initialize lists of empty arrays with same dtype as input
        x_phis = [x.new_zeros([]),] * self.J
        x_psis = [x.new_zeros([]),] * self.J

        # Assert that the length of x is a multiple of 2**J
        T = x.shape[-1]
        assert T % (2 ** self.J) == 0

        ## LEVEL 1 ##
        x_phi, x_psi = FWD_J1.apply(
            x, self.h0o, self.h1o, self.skip_hps[0], self.padding_mode)
        x_psis[0] = x_psi
        if self.include_scale[0]:
            x_phis[0] = x_phi

        ## LEVEL 2 AND GREATER ##
        # Apply multiresolution pyramid by looping over j from fine to coarse
        for j in range(1, self.J):
            if (j%2 == 1) and self.alternate_gh:
                # Pick the dual filters g0a, g1a, etc. instead of h0a, h1a, etc.
                h0a, h1a, h0b, h1b = self.g0a, self.g1a, self.g0b, self.g1b
            else:
                h0a, h1a, h0b, h1b = self.h0a, self.h1a, self.h0b, self.h1b

            x_phi, x_psi = FWD_J2PLUS.apply(
                x_phi, h0a, h1a, h0b, h1b, self.skip_hps[j], self.padding_mode, 
                self.normalize
            )

            if (j%2 == 1) and self.alternate_gh:
                # The result is anti-analytic in the Hilbert sense.
                # We conjugate the result to bring the spectrum back to (0, pi).
                # This is purely by convention and for consistency through j.
                x_psi = torch.conj(x_psi)

            x_psis[j] = x_psi
            if self.include_scale[j]:
                x_phis[j] = x_phi

        # If at least one of the booleans in the list include_scale is True,
        # return the list x_phis as yl. Otherwise, return the last x_phi.
        if True in self.include_scale:
            return x_phis, x_psis
        else:
            return x_phi, x_psis

class DTCWTInverse(torch.nn.Module):
    def __init__(
        self,
        level1="near_sym_a",
        qshift="qshift_a",
        J=8,
        skip_hps=False,
        include_scale=False,
        alternate_gh=True,
        padding_mode='zeros',
        normalize=True
    ):
        # Instantiate PyTorch NN Module
        super().__init__()

        # Store metadata
        self.level1 = level1
        self.qshift = qshift
        self.J = J
        self.alternate_gh = alternate_gh
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
        else:
            self.padding_mode = padding_mode
        self.normalize = normalize

        # Load first-level biorthogonal wavelet filters from disk.
        # h0o is the low-pass filter.
        # h1o is the high-pass filter.
        _, g0o, _, g1o = pytw.dtcwt.coeffs._load_from_file(
            level1, ("h0o", "g0o", "h1o", "g1o")
        )
        self.register_buffer("g0o", prep_filt(g0o))
        self.register_buffer("g1o", prep_filt(g1o))

        # Load higher-level quarter-shift wavelet filters from disk.
        # h0a is the low-pass filter from tree a (real part).
        # h0b is the low-pass filter from tree b (imaginary part).
        # g0a is the low-pass dual filter from tree a (real part).
        # g0b is the low-pass dual filter from tree b (imaginary part).
        # h1a is the high-pass filter from tree a (real part).
        # h1b is the high-pass filter from tree b (imaginary part).
        # g1a is the high-pass dual filter from tree a (real part).
        # g1b is the high-pass dual filter from tree b (imaginary part).
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = pytw.dtcwt.coeffs._load_from_file(
            qshift, ("h0a", "h0b", "g0a", "g0b", "h1a", "h1b", "g1a", "g1b")
        )
        self.register_buffer("h0a", prep_filt(h0a))
        self.register_buffer("h0b", prep_filt(h0b))
        self.register_buffer("g0a", prep_filt(g0a))
        self.register_buffer("g0b", prep_filt(g0b))
        self.register_buffer("h1a", prep_filt(h1a))
        self.register_buffer("h1b", prep_filt(h1b))
        self.register_buffer("g1a", prep_filt(g1a))
        self.register_buffer("g1b", prep_filt(g1b))

        # Parse the "skip_hps" argument for skipping finest scales.
        if isinstance(skip_hps, (list, tuple, np.ndarray)):
            self.skip_hps = skip_hps
        else:
            self.skip_hps = [skip_hps,] * self.J

        # Parse the "include_scale" argument for including other low-pass
        # outputs in addition to the coarsest scale.
        if isinstance(include_scale, (list, tuple, np.ndarray)):
            self.include_scale = include_scale
        else:
            self.include_scale = [include_scale,] * self.J

    def forward(self, coeffs):
        """
        coeffs (yl, yh): tuple of low-pass (yl) and band-pass (yh) coefficients.
        """
        if True in self.include_scale:
            x_phi, x_psis = coeffs[0][self.J-1], coeffs[1]
        else:
            x_phi, x_psis = coeffs
        
        assert len(x_psis) == self.J

        if self.alternate_gh:
            raise NotImplementedError #TBD
        
        ## LEVEL 2 AND GREATER ##
        for j in range(self.J-1, 0, -1):
            x_psi = x_psis[j]
            assert x_psi.shape[-1] * 2 == x_phi.shape[-1]
            
            x_phi = INV_J2PLUS.apply(x_phi, x_psi, self.g0a, self.g1a, self.g0b, self.g1b, self.padding_mode, 
                self.normalize)

        ## LEVEL 1 ##
        x_phi = INV_J1.apply(x_phi, x_psis[0], self.g0o, self.g1o, self.padding_mode, 
            self.normalize)
        
        return x_phi