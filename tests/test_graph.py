import pytest
import torch
import murenn


@pytest.mark.parametrize("J", list(range(1, 3)))
@pytest.mark.parametrize("Q", [3, 4])
@pytest.mark.parametrize("T", [8, 16])
@pytest.mark.parametrize("padding_mode", ["symmetric", "zeros"])
def test_shape(J, Q, T, padding_mode):
    B, C, N = 2, 3, 2**(J+4)
    x = torch.zeros(B, C, N)
    graph = murenn.MURENN_GRAPH(
        J=J,
        Q=Q,
        T=T,
        padding_mode=padding_mode,
    )
    y = graph(x)
    assert y.shape == (B, Q, J, 2**4)