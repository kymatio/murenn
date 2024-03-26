import pytest
import torch
import murenn


@pytest.mark.parametrize("J", list(range(1, 3)))
@pytest.mark.parametrize("Q", [3, 4])
@pytest.mark.parametrize("T", [8, 16])
@pytest.mark.parametrize("padding_mode", ["symmetric", "zeros"])
def test_direct_shape(J, Q, T, padding_mode):
    B, C, N = 2, 3, 2**(J+4)
    x = torch.zeros(B, C, N)
    graph = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=C,
        padding_mode=padding_mode,
    )
    y = graph(x)
    assert y.shape == (B, C, Q, J, 2**4)
    

def test_direct_diff():
    J, Q, T = 3, 4, 4
    B, C, N = 2, 3, 2**(J+4)
    x = torch.zeros(B, C, N)
    graph = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=C,
    )
    y = graph(x)
    y.mean().backward()
    for conv1d in graph.conv1d:
        assert conv1d.weight.grad != None