import pytest
import torch
from torch.autograd import gradcheck
import murenn
import murenn.dtcwt.transform_funcs as tf
from contextlib import contextmanager


if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = torch.device("cpu")


@contextmanager
def set_double_precision():
    old_prec = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        yield
    finally:
        torch.set_default_dtype(old_prec)


@pytest.mark.parametrize("skip_hps", [True, False])
def test_fwd_j1(skip_hps):
    J = 1
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        x = torch.randn(2, 2, 2, device=dev, requires_grad=True)
        fwd = murenn.DTCWTDirect(J=J, skip_hps=skip_hps).to(dev)
    input = (x, fwd.h0o, fwd.h1o, fwd.skip_hps[0], fwd.padding_mode)
    gradcheck(tf.FWD_J1.apply, input, eps=eps, atol=atol)


@pytest.mark.parametrize("skip_hps", [[0, 1], [1, 0]])
def test_fwd_j2(skip_hps):
    J = 2
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        x = torch.randn(2, 2, 4, device=dev, requires_grad=True)
        fwd = murenn.DTCWTDirect(J=J, skip_hps=skip_hps).to(dev)
    input = (
        x,
        fwd.h0a,
        fwd.h1a,
        fwd.h0b,
        fwd.h1b,
        fwd.skip_hps[1],
        fwd.padding_mode,
    )
    gradcheck(tf.FWD_J2PLUS.apply, input, eps=eps, atol=atol)


def test_inv_j1():
    J = 1
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        lo = torch.randn(2, 2, 2, device=dev, requires_grad=True)
        hi_r = torch.randn(2, 2, 1, device=dev, requires_grad=True)
        hi_i = torch.randn(2, 2, 1, device=dev, requires_grad=True)
        inv = murenn.DTCWTInverse(J=J).to(dev)
    input = (lo, hi_r, hi_i, inv.g0o, inv.g1o, inv.padding_mode)
    gradcheck(tf.INV_J1.apply, input, eps=eps, atol=atol)


def test_inv_j2():
    J = 2
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        lo = torch.randn(2, 2, 8, device=dev, requires_grad=True)
        bp_r = torch.randn(2, 2, 4, device=dev, requires_grad=True)
        bp_i = torch.randn(2, 2, 4, device=dev, requires_grad=True)
        inv = murenn.DTCWTInverse(J=J).to(dev)

    input = (
        lo,
        bp_r,
        bp_i,
        inv.g0a,
        inv.g1a,
        inv.g0b,
        inv.g1b,
        inv.padding_mode,
    )
    gradcheck(tf.INV_J2PLUS.apply, input, eps=eps, atol=atol)


@pytest.mark.parametrize("alternate_gh", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_autograd(alternate_gh, normalize):
    b = 2
    ch = 3
    N = 2**5
    x = torch.zeros(b, ch, N, requires_grad=True)

    J = 2
    kwargs = dict(J=J, alternate_gh=alternate_gh, normalize=normalize)
    dtcwt = murenn.DTCWT(**kwargs)
    idtcwt = murenn.IDTCWT(**kwargs)

    x_phi, x_psis = dtcwt(x)
    y = idtcwt(x_phi, x_psis)
    y[:, :, N // 2].mean().backward()
    g = x.grad.sum(axis=1).sum(axis=0)

    dirac = torch.zeros(N)
    dirac[N // 2] = 1

    assert torch.allclose(g, dirac, atol=1e-3)
    assert x.grad.shape == (b, ch, N)