import torch


def reflect(x, minx, maxx):
    """Reflect the values in matrix *x* about the scalar values *minx* and
    *maxx*.  Hence a vector *x* containing a long linearly increasing series is
    converted into a waveform which ramps linearly up and down between *minx*
    and *maxx*.  If *x* contains integers and *minx* and *maxx* are (integers +
    0.5), the ramps will have repeated max and min samples.

    Adapted from Rich Wareham's dtcwt NumPy library (2013), which in turn was
    adapted from Nick Kingsbury's MATLAB implementation (1999).
    """
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = torch.remainder(x - minx, rng_by_2)
    normed_mod = torch.where(mod < 0, mod + rng_by_2, mod)
    out = torch.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return out.int()

def pad_(x, h, padding_mode, same_pad = True):
    """
    Padding the input tensor x so that the later conv1d result have the same 
    length as the input tensor.

    Args:
        x is the input Tensor
        h is the conv1d filter
        padding_mode: 'constant', 'symmetric', 'replicate' or 'circular'
        same_pad: if False, full padding will be applied
    """
    padding_total = h.shape[-1]-1
    padding_left = padding_total//2 if same_pad else h.shape[-1]
    padding_right = padding_total//2 if same_pad else h.shape[-1]

    if padding_mode == 'symmetric':
        l = x.shape[-1]
        xe = reflect(torch.arange(-padding_left, l+padding_right, dtype=torch.int32), -0.5, l-0.5)
        out = x[:,:,xe]
    else: 
        out = torch.nn.functional.pad(x, (padding_left, padding_right), padding_mode)
    return out