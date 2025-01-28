import murenn.dtcwt
import pytest
import torch
import murenn
import math

from murenn.dtcwt.nn import ModulusStable, Downsampling


if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")

@pytest.mark.parametrize("J", list(range(2, 4)))
@pytest.mark.parametrize("Q", [3, 4])
@pytest.mark.parametrize("T", [8, 16])
@pytest.mark.parametrize("padding_mode", ["symmetric", "zeros"])
@pytest.mark.parametrize("N", list(range(10)))
@pytest.mark.parametrize("include_lp", [True, False])
def test_direct_shape(J, Q, T, N, padding_mode, include_lp):
    B, C, L = 2, 3, 2**J+N
    x = torch.zeros(B, C, L)
    graph = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=C,
        padding_mode=padding_mode,
        include_lp=include_lp,
    )
    y = graph(x)


    assert y.shape[:2] == (B, Q * J + C * include_lp)
    

def test_direct_diff():
    J, Q, T = 3, 4, 4
    B, C, N = 2, 3, 2**(J+4)
    x = torch.zeros(B, C, N)
    tfm = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=C,
    )
    y = tfm(x)
    y.mean().backward()
    for conv1d in tfm.conv1d:
        assert conv1d.weight.grad != None


@pytest.mark.parametrize("Q", [3, 4])
@pytest.mark.parametrize("T", [8, 16])
@pytest.mark.parametrize("N", list(range(5)))
@pytest.mark.parametrize("include_lp", [True, False])
def test_multi_layers(Q, T, N, include_lp):
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
            include_lp=include_lp,
        )
        x = layer_i(x)
        C = Q * J + C * include_lp
        assert x.shape[1] == C


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
@pytest.mark.parametrize("T", [2, 3])
def test_toconv1d(Q, T):
    J = 4
    tfm = murenn.MuReNNDirect(
        J=J,
        Q=Q,
        T=T,
        in_channels=2,
    )
    N = 2**J*16
    x = torch.zeros(1, 1, N)
    x[:,:,N//2] = 1
    conv1ds = tfm.to_conv1d()
    for conv1d in [conv1ds["real"], conv1ds["imag"]]:
        assert isinstance(conv1d, torch.nn.Conv1d)
        y = conv1d(x)
        assert isinstance(y, torch.Tensor)
        assert y.dtype == x.dtype
        assert y.shape == (1, J*Q, N)
    
    # Test the energy gain
    tfm = murenn.MuReNNDirect(
        J=J,
        Q=1,
        T=1,
        in_channels=1,
    )
    # Initialize the learnable filters with dirac
    for conv1d in tfm.conv1d:
        torch.nn.init.dirac_(conv1d.weight)
    # Get the dtcwt filters
    psis = tfm.to_conv1d()
    # Test the energy crossing subbands
    y = psis["complex"](torch.complex(x, x)/math.sqrt(2))
    energy = torch.linalg.norm(y, dim=-1)
    assert torch.allclose(energy, torch.ones(1, J), atol=0.1)
