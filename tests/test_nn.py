import pytest
import torch
import murenn

from murenn.dtcwt.nn import ModulusStable


if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

@pytest.mark.parametrize("J", list(range(2, 4)))
@pytest.mark.parametrize("Q", [3, 4])
@pytest.mark.parametrize("T", [8, 16])
@pytest.mark.parametrize("padding_mode", ["symmetric", "zeros"])
@pytest.mark.parametrize("N", list(range(10)))
def test_direct_shape(J, Q, T, N, padding_mode):
    B, C, L = 2, 3, 2**J+N
    x = torch.zeros(B, C, L)
    graph = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=C,
        padding_mode=padding_mode,
    )
    y = graph(x)
    assert y.shape[:4] == (B, C, Q, J)
    

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


@pytest.mark.parametrize("Q", [3, 4])
@pytest.mark.parametrize("T", [8, 16])
@pytest.mark.parametrize("N", list(range(5)))
def test_multi_layers(Q, T, N):
    J = 2
    B, C, L = 2, 3, 2**J+N
    x = torch.zeros(B, C, L)
    for i in range(3):
        x = x.view(B, -1, x.shape[-1])
        layer_i = murenn.MuReNNDirect(
            J=J,
            Q=Q,
            T=T,
            in_channels=x.shape[1],
        )
        x = layer_i(x)


def test_modulus():
    # check the value
    x_r = torch.randn(2, 2, 2**5, device=dev, requires_grad=True)
    x_i = torch.randn(2, 2, 2**5, device=dev, requires_grad=True)
    Ux = ModulusStable.apply(x_r, x_i)
    assert torch.allclose(Ux, torch.sqrt(x_r ** 2 + x_i ** 2))
    # check the gradient
    loss = torch.sum(Ux)
    loss.backward()
    Ux2 = Ux.clone()
    x_r2 = x_r.clone()
    x_i2 = x_i.clone()
    xr_grad = x_r2 / Ux2
    xi_grad = x_i2 / Ux2
    assert torch.allclose(x_r.grad, xr_grad, atol = 1e-4)
    assert torch.allclose(x_i.grad, xi_grad, atol = 1e-4)
    # Test the differentiation with a vector made of zeros
    x0r = torch.zeros(2, 2, 2**5, device=dev, requires_grad=True)
    x0i = torch.zeros(2, 2, 2**5, device=dev, requires_grad=True)
    Ux0 = ModulusStable.apply(x0r, x0i)
    loss0 = torch.sum(Ux0)
    loss0.backward()
    assert torch.max(torch.abs(x0r.grad)) <= 1e-7
    assert torch.max(torch.abs(x0i.grad)) <= 1e-7

@pytest.mark.parametrize("Q", [1, 2])
@pytest.mark.parametrize("T", [1, 2])
def test_toconv1d_shape(Q, T):
    J = 4
    tfm = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=2,
    )
    conv1d = tfm.to_conv1d
    assert isinstance(conv1d, torch.nn.Conv1d)