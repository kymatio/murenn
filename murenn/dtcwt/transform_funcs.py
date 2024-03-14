import numpy as np
import torch
from murenn.dtcwt.utils import pad_, mode_to_int, int_to_mode
from murenn.dtcwt.lowlevel import coldfilt, colifilt


class FWD_J1(torch.autograd.Function):
    """Differentiable function doing forward DTCWT at level 1.
    Returns low-pass (-pi/4 to pi/4) and high-pass (pi/4 to 3pi/4) as a pair.
    """

    @staticmethod
    def forward(ctx, x, h0, h1, skip_hps, padding_mode):
        """
        Forward dual-tree complex wavelet transform at level 1.

        Args:
            ctx is the DTCWTDirect object
            x is the input Tensor
            h0 is the low-pass analysis filter
            h1 is the high-pass analysis filter
            skip_hps: if True, skip high-pass filtering
            padding_mode: 'constant', 'symmetric', 'replicate' or 'circular'

        Returns:
            lo: low-pass output (-pi/4 to pi/4)
            hi_r: real part of the high-pass output (pi/4 to 3pi/4)
            hi_i: imaginary part of the high-pass output (pi/4 to 3pi/4)
        """

        # Replicate filters along the channel dimension
        b, ch, T = x.shape
        h0_rep = h0.repeat(ch, 1, 1)
        h1_rep = h1.repeat(ch, 1, 1)
        ctx.save_for_backward(h0_rep, h1_rep)
        ctx.skip_hps = skip_hps
        ctx.mode = mode_to_int(padding_mode)

        # Apply low-pass filtering
        lo = torch.nn.functional.conv1d(
            pad_(x, h0, padding_mode), h0_rep, groups=ch)

        # Apply high-pass filtering. If skipped, create an empty array.
        if skip_hps:
            hi = x.new_zeros(x.shape)
        else:
            hi = torch.nn.functional.conv1d(
                pad_(x, h1, padding_mode), h1_rep, groups=ch)

        # Return low-pass (x_phi), real and imaginary part of high-pass (x_psi) 
        return lo, hi[:,:,::2], hi[:,:,1::2]
    
    @staticmethod
    def backward(ctx, dx_phi, dx_psi_r, dx_psi_i):
        h0, h1 = ctx.saved_tensors
        skip_hps = ctx.skip_hps
        mode = int_to_mode(ctx.mode)
        b, ch, T = dx_phi.shape
        if not ctx.needs_input_grad[0]:
            dx = None
        else:
            dx = torch.nn.functional.conv1d(pad_(dx_phi, h0, mode), h0, groups = ch)
            if not skip_hps:
                dx_psi = torch.stack((dx_psi_r, dx_psi_i), dim=-1).view(b, ch, T)
                dx += torch.nn.functional.conv1d(pad_(dx_psi, h1, mode), h1, groups = ch)
        return dx, None, None, None, None


class FWD_J2PLUS(torch.autograd.Function):
    """
    Differentiable function doing forward DTCWT at scales j>1.

    The real part of bp corresponds to the high-pass output of
    tree a whereas the imaginary part of bp corresponds to the
    high-pass output of tree b."""

    @staticmethod
    def forward(ctx, x_phi, h0a, h1a, h0b, h1b, skip_hps, padding_mode, normalize):
        """
        Forward dual-tree complex wavelet transform at levels 2 and coarser.

        Args:
            ctx: DTCWTDirect object
            x_phi: input Tensor
            h0a: low-pass filter of tree a (real part)
            h1a: high-pass filter of tree a (real part)
            h0b: low-pass filter of tree b (imaginary part)
            h1b: high-pass filter of tree b (imaginary part)
            skip_hps: if True, skip high-pass filtering
            padding_mode: 'constant'(zero padding), 'symmetric', 'replicate' or 'circular'
            normalise: bool, normalise or not

        Returns:
            lo: low-pass output from both trees
            bp_r: band-pass output from tree b.
            bp_i: band-pass output from tree a."""

        # Replicate filters along the channel dimension
        b, ch, T = x_phi.shape
        h0a_rep = h0a.repeat(ch, 1, 1)
        h1a_rep = h1a.repeat(ch, 1, 1)
        h0b_rep = h0b.repeat(ch, 1, 1)
        h1b_rep = h1b.repeat(ch, 1, 1)
        ctx.save_for_backward(h0a_rep, h1a_rep, h0b_rep, h1b_rep)
        ctx.skip_hps = skip_hps
        ctx.mode = mode_to_int(padding_mode)
        ctx.normalize = normalize

        # Apply low-pass filtering on trees a (real) and b (imaginary).
        lo = coldfilt(x_phi, h0a_rep, h0b_rep, padding_mode)
        # 'lo' the low-pass output such that lo[2t]=lo_a[t] and lo[2t+1]=lo_b[t]
        lo = torch.stack([lo[:,:ch], lo[:,ch:2*ch]], dim=-1).view(b, ch, T//2)

        # Apply high-pass filtering. If skipped, create an empty array.
        if skip_hps:
            bp_r = lo.new_zeros((b, ch, T//4))
            bp_i = lo.new_zeros((b, ch, T//4))
        else:
            bp =  coldfilt(x_phi, h1a_rep, h1b_rep, padding_mode)
            bp_r = bp[:,ch:2*ch]
            bp_i = bp[:,:ch]

        # Return low-pass output, and band-pass output in conjunction:
        # real part for tree a and imaginary part for tree b.
        if normalize:
            return 1/np.sqrt(2) * lo, 1/np.sqrt(2) * bp_r, 1/np.sqrt(2) * bp_i
        else:
            return lo, bp_r, bp_i

    @staticmethod
    def backward(ctx, dx_phi, dx_psi_r, dx_psi_i):
        g0b, g1b, g0a, g1a = ctx.saved_tensors
        skip_hps = ctx.skip_hps
        padding_mode = int_to_mode(ctx.mode)
        normalize = ctx.normalize
        b, ch, T = dx_phi.shape
        if not ctx.needs_input_grad[0]:
            dx = None
        else:
            dx = colifilt(dx_phi, g0a, g0b, padding_mode)
            if not skip_hps:
                dx_psi = torch.stack((dx_psi_i, dx_psi_r), dim=-1).view(b, ch, T)
                dx += colifilt(dx_psi, g1a, g1b, padding_mode)
            if normalize:
                dx *= 1/np.sqrt(2)
        return dx, None, None, None, None, None, None, None


class INV_J1(torch.autograd.Function):
    """Differentiable function doing inverse DTCWT at level 1.
    Returns a full-band 1-D signal from a low-pass and a high-pass component.
    """

    @staticmethod
    def forward(ctx, lo, hi_r, hi_i, g0, g1, padding_mode):
        """
        Inverse dual-tree complex wavelet transform at level 1.

        Args:
            ctx is the DTCWTInverse object
            lo is the low-pass input (-pi/4 to pi/4)
            hi_r is the real part high-pass input (pi/4 to 3pi/4)
            hi_i is the imaginary part high-pass input (pi/4 to 3pi/4)
            g0 is the low-pass synthesis filter
            g1 is the high-pass synthesis filter

        Returns:
            x: reconstructed 1-D signal (possibly multichannel)
        """
        # Replicate filters along the channel dimension
        b, ch, T = lo.shape
        g0_rep = g0.repeat(ch, 1, 1)
        g1_rep = g1.repeat(ch, 1, 1)
        ctx.save_for_backward(g0_rep, g1_rep)
        ctx.mode = mode_to_int(padding_mode)

        # Apply dual low-pass filtering
        x0 = torch.nn.functional.conv1d(pad_(lo, g0, padding_mode), g0_rep, groups=ch)

        # Apply dual high-pass filtering
        hi = torch.stack((hi_r, hi_i), dim=-1).view(b, ch, T)
        x1 = torch.nn.functional.conv1d(pad_(hi, g1, padding_mode), g1_rep, groups=ch)

        # Mix low-pass and high-pass contributions
        x = x0 + x1
        return x
    
    @staticmethod
    def backward(ctx, dx):
        g0, g1 = ctx.saved_tensors
        padding_mode = int_to_mode(ctx.mode)

        ch = dx.shape[1]
        dlo, dhi_r, dhi_i = None, None, None
        if ctx.needs_input_grad[0]:
            dlo = torch.nn.functional.conv1d(
                pad_(dx, g0, padding_mode), g0, groups=ch)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            dhi = torch.nn.functional.conv1d(
                pad_(dx, g1, padding_mode), g1, groups=ch)
            if ctx.needs_input_grad[1]:
                dhi_r = dhi[:,:,::2]
            if ctx.needs_input_grad[2]:
                dhi_i = dhi[:,:,1::2]
        return dlo, dhi_r, dhi_i, None, None, None
    


class INV_J2PLUS(torch.autograd.Function):
    """Differentiable function doing inverse DTCWT at levels >1.
    Returns a broadband 1-D signal from a low-pass and a high-pass component.
    """

    @staticmethod
    def forward(ctx, lo, bp_r, bp_i, g0a, g1a, g0b, g1b, padding_mode, normalize):
        """
        Inverse dual-tree complex wavelet transform at levels 2 and coarser.

        Args:
            ctx is the DTCWTInverse object
            lo is the low-pass output from both trees
            bp_r is the real part of the band-pass output
            bp_i is the imaginary part of the band-pass output
            g0a is the dual low-pass filter of tree a (real part)
            g1a is the dual high-pass filter of tree a (real part)
            g0b is the dual low-pass filter of tree b (imaginary part)
            g1b is the dual high-pass filter of tree b (imaginary part)

        Returns:
            x_a: reconstructed 1-D signal from tree a (real part)
            x_b: reconstructed 1-D signal from tree a (imaginary part)
        """
        # Replicate filters along the channel dimension
        b, ch, T = lo.shape
        g0a_rep = g0a.repeat(ch, 1, 1)
        g1a_rep = g1a.repeat(ch, 1, 1)
        g0b_rep = g0b.repeat(ch, 1, 1)
        g1b_rep = g1b.repeat(ch, 1, 1)
        ctx.save_for_backward(g0a_rep, g1a_rep, g0b_rep, g1b_rep)
        ctx.normalize = normalize
        ctx.mode = mode_to_int(padding_mode)

        bp = torch.stack((bp_i, bp_r), dim=-1).view(b, ch, T)
        lo = colifilt(lo, g0a_rep, g0b_rep, padding_mode) + colifilt(bp, g1a_rep, g1b_rep, padding_mode)

        if normalize:
            return np.sqrt(2) * lo
        else:
            return lo

    @staticmethod
    def backward(ctx, dx):
        g0b, g1b, g0a, g1a = ctx.saved_tensors
        padding_mode = int_to_mode(ctx.mode)
        normalize = ctx.normalize
        b, ch, T = dx.shape
        dlo, dbp = None, None
        if ctx.needs_input_grad[0]:
            dlo = coldfilt(dx, g0a, g0b, padding_mode)
            dlo = torch.stack([dlo[:,:ch], dlo[:,ch:2*ch]], dim=-1).view(b, ch, T//2)
            if normalize:
                dlo *= np.sqrt(2)
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                dbp = coldfilt(dx, g1a, g1b, padding_mode)
                if normalize:
                    dbp *= np.sqrt(2)
                if ctx.needs_input_grad[1]:
                    dbp_r = dbp[:,ch:2*ch]
                if ctx.needs_input_grad[2]:
                    dbp_i = dbp[:,:ch]
        return dlo, dbp_r, dbp_i, None, None, None, None, None, None