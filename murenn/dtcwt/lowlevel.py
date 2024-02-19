import torch
from murenn.dtcwt.utils import pad_


def prep_filt(h):
    """Convert a 1-D filter from pytorch_wavelets (NumPy row vector)
    into a PyTorch object (1x1xT Tensor) for MuReNN."""
    # Note that we drop the trailing singleton dimension because
    # this filter will be used on 1-D signals, not 2-D images.
    # Conversely, we prepend two singleton dimensions: one for
    # the signal index (in the batch) and the other for the
    # channel index (for multimodal time series).
    return torch.tensor(h[None, None, :, 0], dtype=torch.float32)

def colifilt(x, ha, hb, padding_mode):
    if x is None or x.shape == torch.Size([]):
        return torch.zeros(1,1,1,1, device=x.device)
    m = ha.shape[-1]
    m2 = m // 2
    hao = ha[:,:,1::2]
    hae = ha[:,:,::2]
    hbo = hb[:,:,1::2]
    hbe = hb[:,:,::2]
    batch, ch, T = x.shape
    assert T % 2 == 0
    x = pad_(x, hao, padding_mode, False)

    if m2 % 2 == 0:
        h1 = hae
        h2 = hbe
        h3 = hao
        h4 = hbo
        x = torch.cat((x[:,:,:-2:2], x[:,:,1:-2:2], x[:,:,2::2], x[:,:,3::2]), dim=1)
    else:
        h1 = hao
        h2 = hbo
        h3 = hae
        h4 = hbe
        x = torch.cat((x[:,:,1:-1:2], x[:,:,2:-1:2], x[:,:,1:-1:2], x[:,:,2:-1:2]), dim=1)
    h = torch.cat((h1, h2, h3, h4), dim=0)

    x = torch.nn.functional.conv1d(x, h, groups=4*ch)
    x = torch.stack([x[:,:ch], x[:,ch:2*ch], x[:,2*ch:3*ch], x[:,3*ch:]], dim=3).view(batch, ch, T*2)

    return x