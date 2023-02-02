import torch


def prep_filt(h):
    """Convert a 1-D filter from pytorch_wavelets (NumPy row vector)
    into a PyTorch object (1x1xT Tensor) for MuReNN."""
    # Note that we drop the trailing singleton dimension because
    # this filter will be used on 1-D signals, not 2-D images.
    # Conversely, we prepend two singleton dimensions: one for
    # the signal index (in the batch) and the other for the
    # channel index (for multimodal time series).
    return torch.tensor(h[None, None, :, 0], dtype=torch.float32)
