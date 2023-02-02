import pytest
import torch

import murenn


@pytest.mark.parametrize("skip_hps", [False, [0, 1]])
@pytest.mark.parametrize("include_scale", [False, [7, 8]])
def test_dtcwt(skip_hps, include_scale):
    W = murenn.DTCWT(skip_hps=skip_hps, include_scale=include_scale)
    assert isinstance(W, torch.nn.Module)
