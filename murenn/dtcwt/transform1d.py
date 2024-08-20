import numpy as np
import dtcwt
import torch.nn
import bisect

from murenn.dtcwt.lowlevel import prep_filt
from murenn.dtcwt.transform_funcs import FWD_J1, FWD_J2PLUS, INV_J1, INV_J2PLUS
from .utils import fix_length


class DTCWT(torch.nn.Module):
    def __init__(
        self,
        level1="near_sym_a",
        qshift="qshift_a",
        J=8,
        skip_hps=False,
        include_scale=False,
        alternate_gh=True,
        padding_mode="symmetric",
        normalize=True,
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
            self.skip_hps = [
                skip_hps,
            ] * self.J

        # Parse the "include_scale" argument for including other low-pass
        # outputs in addition to the coarsest scale.
        if isinstance(include_scale, (list, tuple, np.ndarray)):
            self.include_scale = include_scale
        else:
            self.include_scale = [
                include_scale,
            ] * self.J

        if padding_mode == "zeros":
            self.padding_mode = "constant"
        else:
            self.padding_mode = padding_mode

        # Load first-level biorthogonal wavelet filters from disk.
        # h0o is the low-pass filter.
        # h1o is the high-pass filter.
        # g0o is the low-pass inverse filter.
        # g1o is the high-pass inverse filter.
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
        padding_mode (str): One of 'symmetric'(default), 'zeros', 'replicate',
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
                Note that T must be even length.

        Returns:
            yl: low-pass coefficients. If include_scale is True (see DTCWTDirect
                constructor), yl is a list of low-pass coefficients at all wavelet
                scales 1 to (J-1). Otherwise (default), yl is a real-valued PyTorch
                tensor.
            yh: band-pass coefficients. A list of PyTorch tensors with J elements,
                containing the band-pass coefficients at all wavelets scales 1 to
                (J-1). These tensors are complex-valued."""

        # Initialize lists of empty arrays with same dtype as input
        x_phis = []
        x_psis = []

        # Extend if the length of x is not even
        T = x.shape[-1]
        if T % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=-1)

        ## LEVEL 1 ##
        x_phi, x_psi_r, x_psi_i = FWD_J1.apply(
            x, self.h0o, self.h1o, self.skip_hps[0], self.padding_mode
        )
        x_psis.append(x_psi_r + 1j * x_psi_i)
        if self.include_scale[0]:
            x_phis.append(x_phi)
        else:
            x_phis.append(x_phi.new_zeros(x_phi.shape))

        ## LEVEL 2 AND GREATER ##
        # Apply multiresolution pyramid by looping over j from fine to coarse
        for j in range(1, self.J):
            if (j % 2 == 1) and self.alternate_gh:
                # Pick the dual filters g0a, g1a, etc. instead of h0a, h1a, etc.
                h0a, h1a, h0b, h1b = self.g0a, self.g1a, self.g0b, self.g1b
            else:
                h0a, h1a, h0b, h1b = self.h0a, self.h1a, self.h0b, self.h1b

            # Ensure the lowpass is divisible by 4
            if x_phi.shape[-1] % 4 != 0:
                x_phi = torch.cat((x_phi[:,:,0:1], x_phi, x_phi[:,:,-1:]), dim=-1)
            if self.normalize:
                x_phi = 1/np.sqrt(2) * x_phi
            x_phi, x_psi_r, x_psi_i = FWD_J2PLUS.apply(
                x_phi,
                h0a,
                h1a,
                h0b,
                h1b,
                self.skip_hps[j],
                self.padding_mode,
            )
            if (j % 2 == 1) and self.alternate_gh:
                # The result is anti-analytic in the Hilbert sense.
                # We conjugate the result to bring the spectrum back to (0, pi).
                # This is purely by convention and for consistency through j.
                x_psi_i = -1 * x_psi_i
            x_psis.append(x_psi_r + 1j * x_psi_i)

            if self.include_scale[j]:
                x_phis.append(x_phi)
            else:
                x_phis.append(x_phi.new_zeros(x_phi.shape))

        # If at least one of the booleans in the list include_scale is True,
        # return the list x_phis as yl. Otherwise, return the last x_phi.
        if True in self.include_scale:
            yl, yh = x_phis, x_psis
        else:
            yl, yh = x_phi, x_psis
        return yl, yh


    @property
    def subbands(self):
        """
        Return the subbands boundaries.
        """

        N = 2 ** (self.J + 4)
        x = torch.zeros(1, 1, N)
        x[0, 0, N//2] = 1

        idtcwt = DTCWTInverse(
            J = self.J, 
            alternate_gh=self.alternate_gh, 
        )
        # Compute the DTCWT of the impulse signal
        x_phi, x_psis = self(x)
        ys = []

        for j in range(self.J):
            y_phi = x_phi * 0
            y_psis = [x_psis[k] * (j==k) for k in range(self.J)]
            y_j_hat = torch.abs(torch.fft.fft(idtcwt(y_phi, y_psis).squeeze()))
            ys.append(y_j_hat)

        lp_psis = [x_psis[k] * 0 for k in range(self.J)]
        y_lp_hat = torch.abs(torch.fft.fft(idtcwt(x_phi, lp_psis).squeeze()))
        ys.append(y_lp_hat)

        # Stack tensors to create a 2D tensor where each row is a tensor from the list
        ys = torch.stack(ys)[:, :N//2]
        # Define the threshold
        threshold = 0.2
        # Apply the threshold
        valid_mask = ys >= threshold
        ys = ys * valid_mask.float()
        # Find the subbands of each frequency
        max_values, max_indices = torch.max(ys, dim=0)
        # Find the boundaries of the subbands
        boundaries = torch.where(max_indices[:-1] != max_indices[1:])[0] + 1
        boundaries = boundaries / N
        boundaries = torch.cat((torch.tensor([0.]), boundaries, torch.tensor([0.5]))).flip(dims=(0,))
        return boundaries.tolist()
    

    def hz_to_octs(self, frequencies, sr=1.0):
        """
        Convert a list of frequencies to their corresponding octave subband indices.

        Parameters:
        frequencies (list of float): List of frequencies to convert.
        sr (float): Sampling rate, default is 1.0.

        Returns:
        list of int: List of octave subband indices corresponding to the input frequencies
            -1 indicates out of range.
        """
        subbands = [boundary * sr for boundary in self.subbands]
        subbands.reverse()
        js = []
        for freq in frequencies:
            i = bisect.bisect_left(subbands, freq)
            j = len(subbands) - i - 1 if i > 0 else -1
            js.append(j)
        return js


class DTCWTInverse(DTCWT):
    """Performs a DTCWT reconstruction of a sequence of 1-D signals. DTCWTInverse
    should be initialized in the same manner as DTCWTDirect.
    The only supported padding mode is 'symmetric'.

    Args: should be the same as DTCWTDirect.
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
        padding_mode="symmetric",
        normalize=True,
        length=None,
    ):
        if padding_mode != "symmetric":
            raise NotImplementedError(
                'Only padding_mode="symmetric" is supported. Got: {padding_mode}'
            )
        super().__init__(
            level1=level1,
            qshift=qshift,
            J=J,
            skip_hps=skip_hps,
            include_scale=include_scale,
            alternate_gh=alternate_gh,
            padding_mode=padding_mode,
            normalize=normalize,
        )
        self.length = length

    def forward(self, yl, yh):
        """
        Args:
            yl: low-pass coefficients for the DTCWT reconstruction. If include_scale
                is True (see DTCWTInverse constructor), yl should be a list of low-pass
                coefficients at all wavelet scales 1 to (J-1). Otherwise (default),
                yl should be a real-valued PyTorch tensor.
            yh: band-pass coefficients for the DTCWT reconstruction. A list of PyTorch
                tensors with J elements, containing the band-pass coefficients at all
                wavelets scales 1 to (J-1). These tensors are complex-valued.
        """

        # x_phi the low-pass, x_psis the band-pass
        if True in self.include_scale:
            x_phi, x_psis = yl[self.J - 1], yh
        else:
            x_phi, x_psis = yl, yh

        # Assert that the band-pass sequence has the same length as the
        # level of decomposition
        assert len(x_psis) == self.J

        ## LEVEL 2 AND GREATER ##
        for j in range(self.J - 1, 0, -1):
            # The band-pass coefficients at level j
            # Check the length of the band-pass, low-pass input coefficients
            x_psi = x_psis[j]

            if x_phi.shape[-1] != x_psi.shape[-1] * 2:
                x_phi = x_phi[:,:,1:-1]
            assert (
                x_psi.shape[-1] * 2 == x_phi.shape[-1]
            ), f"J={j}\n{x_psi.shape[-1]*2}\n{x_phi.shape[-1]}"

            if (j % 2 == 1) and self.alternate_gh:
                x_psi = torch.conj(x_psi)
                g0a, g1a, g0b, g1b = self.h0a, self.h1a, self.h0b, self.h1b
            else:
                g0a, g1a, g0b, g1b = self.g0a, self.g1a, self.g0b, self.g1b

            x_psi_r, x_psi_i = x_psi.real, x_psi.imag
            x_phi = INV_J2PLUS.apply(
                x_phi,
                x_psi_r,
                x_psi_i,
                g0a,
                g1a,
                g0b,
                g1b,
                self.padding_mode,
            )
            if self.normalize:
                x_phi = np.sqrt(2) * x_phi

        # LEVEL 1 ##
        if x_phi.shape[-1] != x_psis[0].shape[-1] * 2:
            x_phi = x_phi[:,:,1:-1]

        x_psi_r, x_psi_i = x_psis[0].real, x_psis[0].imag

        x_phi = INV_J1.apply(
            x_phi, x_psi_r, x_psi_i, self.g0o, self.g1o, self.padding_mode
        )
        if self.length:
            x_phi = fix_length(x_phi, size=self.length)
        return x_phi