import pytest
import torch
from torch.autograd import gradcheck
import numpy as np
import dtcwt
import murenn
import murenn.dtcwt.transform_funcs as tf
from contextlib import contextmanager


if torch.cuda.is_available():
    dev = torch.device('cuda')
else:
    dev = torch.device('cpu')

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
        x = torch.randn(1,1,2, device=dev, requires_grad=True)
        fwd = murenn.DTCWTDirect(J=J, skip_hps=skip_hps).to(dev)
    input = (x, fwd.h0o, fwd.h1o, fwd.skip_hps[0], fwd.padding_mode)
    gradcheck(tf.FWD_J1.apply, input, eps=eps, atol=atol)

@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("skip_hps", [[0, 1], [1, 0]])
def test_fwd_j2(skip_hps,normalize):
    J = 2
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        x = torch.randn(1,1,4, device=dev, requires_grad=True)
        fwd = murenn.DTCWTDirect(J=J, skip_hps=skip_hps, normalize=normalize).to(dev)
    input = (x, fwd.h0a, fwd.h1a, fwd.h0b, fwd.h1b, fwd.skip_hps[1], fwd.padding_mode, fwd.normalize)
    gradcheck(tf.FWD_J2PLUS.apply, input, eps=eps, atol=atol)

def test_inv_j1():
    J = 1
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        lo = torch.randn(1,1,2, device=dev, requires_grad=True)
        hi_r = torch.randn(1,1,1, device=dev, requires_grad=True)
        hi_i = torch.randn(1,1,1, device=dev, requires_grad=True)
        inv = murenn.DTCWTInverse(J=J).to(dev)
    input = (lo, hi_r, hi_i, inv.g0o, inv.g1o, inv.padding_mode)
    gradcheck(tf.INV_J1.apply, input, eps=eps, atol=atol)


@pytest.mark.parametrize("normalize", [True, False])
def test_fwd_j2(normalize):
    J = 2
    eps = 1e-3
    atol = 1e-4
    with set_double_precision():
        lo = torch.randn(1,1,8, device=dev, requires_grad=True)
        bp_r = torch.randn(1,1,4, device=dev, requires_grad=True)
        bp_i = torch.randn(1,1,4, device=dev, requires_grad=True)
        inv = murenn.DTCWTInverse(J=J, normalize=normalize).to(dev)

    input = (lo, bp_r, bp_i, inv.g0a, inv.g1a, inv.g0b, inv.g1b, inv.padding_mode, inv.normalize)
    gradcheck(tf.INV_J2PLUS.apply, input, eps=eps, atol=atol)