import pytest

from murenn.dtcwt.utils import *

@pytest.mark.parametrize("y", [torch.ones((16,)), torch.ones((16, 16))])
@pytest.mark.parametrize("m", [-5, 0, 5])
def test_fix_length(y, m):
    n = m + y.shape[-1]
    y_out = fix_length(y, size=n)
    eq_slice = [slice(None)] * y.ndim
    eq_slice[-1] = slice(y.shape[-1])
    if n > y.shape[-1]:
        assert torch.allclose(y, y_out[tuple(eq_slice)])
    else:
        assert torch.allclose(y[tuple(eq_slice)], y)
