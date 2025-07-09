import numpy as np
import dtcwt
import torch.nn
import bisect

from murenn.dtcwt.lowlevel import prep_filt
from murenn.dtcwt.utils import pad_


class FWD_J1(torch.nn.Module):
    """Differentiable function doing forward UDT-CWT at level 1.
    Returns low-pass (-pi/4 to pi/4) and high-pass (pi/4 to 3pi/4) as a pair.
    """
    def __init__(self, h0, h1, skip_hps, padding_mode):
        super().__init__()
        self.h0 = h0
        self.h1 = h1
        self.skip_hps = skip_hps
        self.padding_mode = padding_mode

    def forward(self, x):
        b, ch, T = x.shape
        h0_rep = self.h0.repeat(ch*2, 1, 1)
        h1_rep = self.h1.repeat(ch*2, 1, 1)
        # Pad the input signal
        padding_total_lo = h0_rep.shape[-1] - 1
        x_lo = torch.nn.functional.pad(x, (padding_total_lo // 2, padding_total_lo - padding_total_lo // 2), self.padding_mode)
        padding_total_hi = h1_rep.shape[-1] - 1
        x_hi = torch.nn.functional.pad(x, (padding_total_hi // 2, padding_total_hi - padding_total_hi // 2), self.padding_mode)
        # Shift the input signal by one sample for tree b to ensure the analycity
        x_lo_shifted = torch.roll(x_lo, -1, dims=-1)
        x_lo = torch.cat((x_lo, x_lo_shifted), dim=1)
        x_hi_shifted = torch.roll(x_hi, -1, dims=-1)
        x_hi = torch.cat((x_hi, x_hi_shifted), dim=1)
        # Apply low-pass filtering
        lo = torch.nn.functional.conv1d(
            x_lo, h0_rep, groups=ch*2)

        # Apply high-pass filtering. If skipped, create an empty array.
        if self.skip_hps:
            hi = x.new_zeros(x.shape)
        else:
            hi = torch.nn.functional.conv1d(
                x_hi, h1_rep, groups=ch*2)

        # Return low-pass (x_phi), real and imaginary part of high-pass (x_psi)
        return lo, hi
    

class FWD_J2PLUS(torch.nn.Module):
    """Differentiable function doing forward UDT-CWT at level 2+.
    Returns low-pass (-pi/4 to pi/4) and high-pass (pi/4 to 3pi/4) as a pair.
    """
    def __init__(self, h0a, h1a, h0b, h1b, dilation, skip_hps, padding_mode):
        super().__init__()
        self.h0a = h0a
        self.h1a = h1a
        self.h0b = h0b
        self.h1b = h1b
        self.skip_hps = skip_hps
        self.padding_mode = padding_mode
        self.dilation = dilation

    def forward(self, x):
        b, ch, T = x.shape
        h0 = torch.cat((self.h0a, self.h0b), dim=0)
        h0_rep = h0.repeat(ch//2, 1, 1)

        # Pad the input signal
        padding_total = (h0.shape[-1] - 1) * self.dilation
        x = torch.nn.functional.pad(x, (padding_total // 2, padding_total - padding_total // 2), self.padding_mode)

        # Low-pass filtering
        lo = torch.nn.functional.conv1d(
            x, h0_rep, dilation=self.dilation, groups=ch)

        # Apply high-pass filtering. If skipped, create an empty array.
        if self.skip_hps:
            hi = x.new_zeros(x.shape)
        else:
            h1 = torch.cat((self.h1a, self.h1b), dim=0)
            h1_rep = h1.repeat(ch//2, 1, 1)
            hi = torch.nn.functional.conv1d(
            x, h1_rep, dilation=self.dilation, groups=ch)

        return lo, hi


class UDTCWTDirect(torch.nn.Module):
    """
    Undecimated Dual-Tree Complex Wavelet Transform (UDT-CWT) for 1D signals.
    """
    def __init__(
        self,
        level1="near_sym_a",
        qshift="qshift_a",
        J=8,
        skip_hps=False,
        include_scale=False,
        alternate_gh=False,
        padding_mode="zeros",
    ):
        super().__init__()
        self.level1 = level1
        self.qshift = qshift
        self.J = J
        self.alternate_gh = alternate_gh

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
        h0o, g0o, h1o, g1o = dtcwt.coeffs.biort(level1)
        self.register_buffer("g0o", prep_filt(g0o))
        self.register_buffer("g1o", prep_filt(g1o))
        self.register_buffer("h0o", prep_filt(h0o))
        self.register_buffer("h1o", prep_filt(h1o))
        h0a, h0b, g0a, g0b, h1a, h1b, g1a, g1b = dtcwt.coeffs.qshift(qshift)
        self.register_buffer("h0a", prep_filt(h0a))
        self.register_buffer("h0b", prep_filt(h0b))
        self.register_buffer("g0a", prep_filt(g0a))
        self.register_buffer("g0b", prep_filt(g0b))
        self.register_buffer("h1a", prep_filt(h1a))
        self.register_buffer("h1b", prep_filt(h1b))
        self.register_buffer("g1a", prep_filt(g1a))
        self.register_buffer("g1b", prep_filt(g1b))
        self.fwd_j1 = FWD_J1(self.h0o, self.h1o, self.skip_hps[0], self.padding_mode)
        self.fwd_j2plus = torch.nn.ModuleList([
            FWD_J2PLUS(
                self.h0a, self.h1a, self.h0b, self.h1b, 2 ** j, self.skip_hps[j], self.padding_mode
            ) for j in range(1, J)
        ])

    def forward(self, x):

        x_phis = []
        x_psis = []

        # Extend if the length of x is not even
        B, C, T = x.shape
        if T % 2 != 0:
            x = torch.cat((x, x[:,:,-1:]), dim=-1)

        ## LEVEL 1 ##
        x_phi, x_psi = self.fwd_j1(x)
        # x_psis.append(x_psi)
        x_psis.append(x_psi[:, C:2*C, :] + 1j * x_psi[:, :C, :])
        if self.include_scale[0]:
            x_phis.append(x_phi)
        else:
            x_phis.append(x_phi.new_zeros(x_phi.shape))

        ## LEVEL 2 AND GREATER ##
        # Apply multiresolution pyramid by looping over j from fine to coarse
        for j in range(self.J - 1):

            # Ensure the lowpass is divisible by 4
            if x_phi.shape[-1] % 4 != 0:
                x_phi = torch.cat((x_phi[:,:,0:1], x_phi, x_phi[:,:,-1:]), dim=-1)
            x_phi, x_psi = self.fwd_j2plus[j](x_phi)
            x_psis.append(x_psi[:, :C, :] + 1j * x_psi[:, C:2*C, :])

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
