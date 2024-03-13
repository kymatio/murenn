import numpy as np
import dtcwt
import torch.nn

from murenn.dtcwt.lowlevel import prep_filt
from murenn.dtcwt.transform_funcs import FWD_J1, FWD_J2PLUS, INV_J1, INV_J2PLUS


class DTCWT(torch.nn.Module):
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
        super().__init__()
        self.level1 = level1
        self.qshift = qshift
        self.J = J
        self.alternate_gh = alternate_gh
        self.normalize = normalize

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
        
        if padding_mode == 'zeros':
            self.padding_mode = 'constant'
        else:
            self.padding_mode = padding_mode
        
        # Load first-level biorthogonal wavelet filters from disk.
        # h0o is the low-pass filter.
        # h1o is the high-pass filter.
        h0o, g0o, h1o, g1o = dtcwt.coeffs.biort(level1)
        self.register_buffer("g0o", prep_filt(g0o))
        self.register_buffer("g1o", prep_filt(g1o))
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
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = dtcwt.coeffs.qshift(qshift)
        self.register_buffer("h0a", prep_filt(h0a))
        self.register_buffer("h0b", prep_filt(h0b))
        self.register_buffer("g0a", prep_filt(g0a))
        self.register_buffer("g0b", prep_filt(g0b))
        self.register_buffer("h1a", prep_filt(h1a))
        self.register_buffer("h1b", prep_filt(h1b))
        self.register_buffer("g1a", prep_filt(g1a))
        self.register_buffer("g1b", prep_filt(g1b))

class DTCWTDirect(DTCWT):
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

    def forward(self, x):
        """Forward Dual-Tree Complex Wavelet Transform (DTCWT) of a 1-D signal.

        Args:
            x (PyTorch tensor): Input data. Should be a tensor of shape
                `(B, C, T)` where B is the batch size, C is the number of
                channels and T is the number of time samples.
                Note that T must be a multiple of 2**J, where J is the number
                of wavelet scales (see documentation of DTCWTDirect constructor).

        Returns:
            (yl, yh): tuple of low-pass (yl) and band-pass (yh) coefficients.
                If include_scale is True (see DTCWTDirect constructor), yl is a
                list of low-pass coefficients at all wavelet scales 1 to (J-1).
                Otherwise (default), yl is a real-valued PyTorch tensor of shape
                `(B, C, T/2**(J-1))`.
                Conversely, yh is a list of PyTorch tensors with J elements,
                containing the band-pass coefficients at all wavelets scales 1 to
                (J-1). These tensors are complex-valued and have shapes:
                `(B, C, T)`, `(B, C, T/2)`, `(B, C, T/4)`, etc."""

        # Initialize lists of empty arrays with same dtype as input
        x_phis = []
        x_psis = []

        # Assert that the length of x is a multiple of 2**J
        T = x.shape[-1]
        assert T % (2 ** self.J) == 0

        ## LEVEL 1 ##
        x_phi, x_psi_r, x_psi_i = FWD_J1.apply(
            x, self.h0o, self.h1o, self.skip_hps[0], self.padding_mode)
        x_psis.append(x_psi_r + 1j * x_psi_i)
        if self.include_scale[0]:
            x_phis.append(x_phi)
        else:
            x_phis.append(x_phi.new_zeros(x_phi.shape))

        ## LEVEL 2 AND GREATER ##
        # Apply multiresolution pyramid by looping over j from fine to coarse
        for j in range(1, self.J):
            if (j%2 == 1) and self.alternate_gh:
                # Pick the dual filters g0a, g1a, etc. instead of h0a, h1a, etc.
                h0a, h1a, h0b, h1b = self.g0a, self.g1a, self.g0b, self.g1b
            else:
                h0a, h1a, h0b, h1b = self.h0a, self.h1a, self.h0b, self.h1b

            x_phi, x_psi_r, x_psi_i = FWD_J2PLUS.apply(
                x_phi, h0a, h1a, h0b, h1b, self.skip_hps[j], self.padding_mode, 
                self.normalize
            )

            if (j%2 == 1) and self.alternate_gh:
                # The result is anti-analytic in the Hilbert sense.
                # We conjugate the result to bring the spectrum back to (0, pi).
                # This is purely by convention and for consistency through j.
                x_psi_i *= -1

            x_psis.append(x_psi_r + 1j * x_psi_i)

            if self.include_scale[j]:
                x_phis.append(x_phi)
            else:
                x_phis.append(x_phi.new_zeros(x_phi.shape))

        # If at least one of the booleans in the list include_scale is True,
        # return the list x_phis as yl. Otherwise, return the last x_phi.
        if True in self.include_scale:
            return x_phis, x_psis
        else:
            return x_phi, x_psis

class DTCWTInverse(DTCWT):
    """Performs a DTCWT reconstruction of a sequence of 1-D signals. DTCWTInverse
    should be initialized in the same manner as DTCWTDirect.

    Args: should be the same as DTCWTForward.
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
        padding_mode (str): One of 'symmetric'(default), 'zeros', 'replicate', 
            and 'circular'. Padding scheme for the filters. 
        normalize (bool): If True (default), the output will be normalized by a 
            factor of 1/sqrt(2)
    """

    def forward(self, coeffs):
        """
        coeffs (x_phi, x_psis): tuple of low-pass (x_phi) and band-pass (x_psis) 
            coefficients. Both x_phi and x_psis should be a list of Pytorch Tensor 
            of shape `(B, C, T)` where B is the batch size, C is the number of channels
            and T is the number of time samples.
        """

        # x_phi the low-pass, x_psis the band-pass
        if True in self.include_scale:
            x_phi, x_psis = coeffs[0][self.J-1], coeffs[1]
        else:
            x_phi, x_psis = coeffs
        
        # Assert that the band-pass sequence has the same length as the
        # level of decomposition
        assert len(x_psis) == self.J
        
        ## LEVEL 2 AND GREATER ##
        for j in range(self.J-1, 0, -1):
            # The band-pass coefficients at level j
            # Check the length of the band-pass, low-pass input coefficients
            x_psi = x_psis[j]
            assert x_psi.shape[-1] * 2 == x_phi.shape[-1], f'J={j}\n{x_psi.shape[-1]*2}\n{x_phi.shape[-1]}'
            if (j%2 == 1) and self.alternate_gh:
                x_psi = torch.conj(x_psi)
                g0a, g1a, g0b, g1b = self.h0a, self.h1a, self.h0b, self.h1b
            else:
                g0a, g1a, g0b, g1b = self.g0a, self.g1a, self.g0b, self.g1b   
            
            x_psi_r, x_psi_i = x_psi.real, x_psi.imag
            x_phi = INV_J2PLUS.apply(x_phi, x_psi_r, x_psi_i, g0a, g1a, g0b, g1b, self.padding_mode, 
                self.normalize)

        ## LEVEL 1 ##
        x_psi_r, x_psi_i = x_psis[0].real, x_psis[0].imag
        x_phi = INV_J1.apply(x_phi, x_psi_r, x_psi_i, self.g0o, self.g1o, self.padding_mode)
        
        return x_phi