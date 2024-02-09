import numpy as np
import torch


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    .. codeauthor:: Rich Wareham <rjw57@cantab.net>, Aug 2013
    .. codeauthor:: Nick Kingsbury, Cambridge University, January 1999.

    """
    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def symm_pad(l, m):
    """ Creates indices for symmetric padding. Works for 1-D.

    Inptus:
        l (int): size of input
        m (int): size of filter
    """
    xe = reflect(np.arange(-m, l+m, dtype='int32'), -0.5, l-0.5)
    return xe

def pad_(x, h, padding_mode, same_pad = True):
    """
    Padding the input tensor x so that the later conv1d result have the same 
    length as the input tensor.

    Args:
        x is the input Tensor
        h is the conv1d filter
        padding_mode: 'constant', 'reflect', 'replicate' or 'circular'
        same_pad: if False, full padding will be applied
    """
    if padding_mode == 'reflect':
        if same_pad:
            # Only works for the odd length filters. If the filter is of even length,
            # this will cause the length of output added by 1.
            assert h.shape[-1]%2 == 1
            xe = symm_pad(x.shape[-1],h.shape[-1]//2)
        else:
            xe = symm_pad(x.shape[-1],h.shape[-1])
        out = x[:,:,xe]

    else:
        if same_pad:
            padding_total = h.shape[-1]-1
            padding_left = padding_total//2
            padding_right = padding_total - padding_left
            out = torch.nn.functional.pad(x, (padding_left, padding_right), padding_mode)
        else:
            out = torch.nn.functional.pad(x, (h.shape[-1], h.shape[-1]), padding_mode)
    return out
