import numpy as np
import torch

import murenn.dtcwt.lowlevel


def test_prep_filt():
    x = np.array([1, 2, 3])[:, np.newaxis]
    y = murenn.dtcwt.lowlevel.prep_filt(x)
    assert isinstance(y, torch.Tensor)
    assert list(y.shape) == [1, 1, len(x)]
