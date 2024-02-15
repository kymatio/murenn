import numpy as np
import torch
from murenn.dtcwt.utils import pad_
from murenn.dtcwt.lowlevel import colifilt


class FWD_J1(torch.autograd.Function):
    """Differentiable function doing forward DTCWT at level 1.
    Returns low-pass (-pi/4 to pi/4) and high-pass (pi/4 to 3pi/4) as a pair.
    """

    @staticmethod
    def forward(ctx, x, h0, h1, skip_hps, padding_mode):
        """
        Forward dual-tree complex wavelet transform at level 1.

        Args:
            ctx is the DTCWTForward object
            x is the input Tensor
            h0 is the low-pass analysis filter
            h1 is the high-pass analysis filter
            skip_hps: if True, skip high-pass filtering
            padding_mode: 'constant', 'reflect', 'replicate' or 'circular'

        Returns:
            lo: low-pass output (-pi/4 to pi/4)
            hi: high-pass output (pi/4 to 3pi/4)
        """

        # Replicate filters along the channel dimension
        b, ch, T = x.shape
        h0_rep = h0.repeat(ch, 1, 1)
        h1_rep = h1.repeat(ch, 1, 1)
        ctx.save_for_backward(h0_rep, h1_rep)

        # Apply low-pass filtering
        lo = torch.nn.functional.conv1d(
        pad_(x, h0, padding_mode), h0_rep, groups=ch)

        # Apply high-pass filtering. If skipped, create an empty array.
        if skip_hps:
            hi = x.new_zeros([])
        else:
            hi = torch.nn.functional.conv1d(
            pad_(x, h1, padding_mode), h1_rep, groups=ch)

        # Return low-pass (x_phi) and high-pass (x_psi) pair
        return lo, hi[:,:,::2] + 1j * hi[:,:,1::2]
    # TBD: backward function !!!


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
            ctx: DTCWTForward object
            x_phi: input Tensor
            h0a: low-pass filter of tree a (real part)
            h1a: high-pass filter of tree a (real part)
            h0b: low-pass filter of tree b (imaginary part)
            h1b: high-pass filter of tree b (imaginary part)
            skip_hps: if True, skip high-pass filtering
            padding_mode: 'constant'(zero padding), 'reflect', 'replicate' or 'circular'
            normalise: bool, normalise or not

        Returns:
            lo: low-pass output from both trees
            bp: band-pass output from both trees."""

        # Replicate filters along the channel dimension
        b, ch, T = x_phi.shape
        h0a_rep = h0a.repeat(ch, 1, 1)
        h1a_rep = h1a.repeat(ch, 1, 1)
        h0b_rep = h0b.repeat(ch, 1, 1)
        h1b_rep = h1b.repeat(ch, 1, 1)
        ctx.save_for_backward(h0a_rep, h1a_rep, h0b_rep, h1b_rep)

        # Input tensor for tree a and tree b. The first two samples are removed so
        # that the length of 'lo' will be the length of 'x_phi' divided by 2.
        x_phi = pad_(x_phi, h0a, padding_mode, False)
        x_a = x_phi[:,:,2::2]
        x_b = x_phi[:,:,3::2]

        # Apply low-pass filtering on trees a (real) and b (imaginary).
        lo_a = torch.nn.functional.conv1d(
            x_a, h0a_rep, stride=2, groups=ch)    
        lo_b = torch.nn.functional.conv1d(
            x_b, h0b_rep, stride=2, groups=ch)

        # Apply high-pass filtering. If skipped, create an empty array.
        if skip_hps:
            bp = x_a.new_zeros([])
        else:
            bp_a = torch.nn.functional.conv1d(
                x_a, h1a_rep, stride=2, groups=ch
            )
            bp_b = torch.nn.functional.conv1d(
                x_b, h1b_rep, stride=2, groups=ch
                )
            bp = bp_b + 1j * bp_a
        
        # 'lo' the low-pass output such that lo[2t]=lo_a[t] and lo[2t+1]=lo_b[t]
        lo = torch.stack((lo_a, lo_b), dim=-1).view(b, ch, T//2)

        # Return low-pass output, and band-pass output in conjunction:
        # real part for tree a and imaginary part for tree b.
        if normalize:
            return np.sqrt(1/2) * lo, np.sqrt(1/2) * bp
        else:
            return lo, bp
        
    # TBD: backward function !!!


class INV_J1(torch.autograd.Function):
    """Differentiable function doing inverse DTCWT at level 1.
    Returns a full-band 1-D signal from a low-pass and a high-pass component.
    """

    @staticmethod
    def forward(ctx, lo, hi, g0, g1, padding_mode, normalize):
        """
        Inverse dual-tree complex wavelet transform at level 1.

        Args:
            ctx is the DTCWTInverse object
            lo is the low-pass input (-pi/4 to pi/4)
            hi is the high-pass input (pi/4 to 3pi/4)
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

        # Apply dual low-pass filtering
        x0 = torch.nn.functional.conv1d(pad_(lo, g0, padding_mode), g0_rep)

        # Apply dual high-pass filtering
        hi = torch.stack((hi.real, hi.imag), dim=-1).view(b, ch, T)
        x1 = torch.nn.functional.conv1d(pad_(hi, g1, padding_mode), g1_rep)

        # Mix low-pass and high-pass contributions
        x = x0 + x1
        return x


class INV_J2PLUS(torch.autograd.Function):
    """Differentiable function doing inverse DTCWT at levels >1.
    Returns a broadband 1-D signal from a low-pass and a high-pass component.
    """

    @staticmethod
    def forward(ctx, lo, bp, g0a, g1a, g0b, g1b, padding_mode, normalize):
        """
        Inverse dual-tree complex wavelet transform at levels 2 and coarser.

        Args:
            ctx is the DTCWTInverse object
            lo is the low-pass output from both trees
            bp is the band-pass output (complex-valued)
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

        bp = torch.stack((bp.imag, bp.real), dim=-1).view(b, ch, T)
        lo = colifilt(lo, g0a_rep, g0b_rep, padding_mode) + colifilt(bp, g1a_rep, g1b_rep, padding_mode)

        return lo