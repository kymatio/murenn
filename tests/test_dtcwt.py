import pytest
import torch
import numpy as np
import dtcwt
import murenn


@pytest.mark.parametrize("skip_hps", [False, [0, 1]])
@pytest.mark.parametrize("include_scale", [False, [7, 8]])
def test_dtcwt(skip_hps, include_scale):
    W = murenn.DTCWT(skip_hps=skip_hps, include_scale=include_scale)
    assert isinstance(W, torch.nn.Module)

@pytest.mark.parametrize('J', list(range(1, 10)))
def test_fwd_same(J):
    decimal = 4
    X = np.random.rand(2**(J+2))
    Xt = torch.tensor(X, dtype=torch.get_default_dtype()).view(1,1,2**(J+2))
    xfm_murenn = murenn.DTCWTForward(J=J, alternate_gh=False, include_scale=False,padding_mode='reflect', normalise=False)
    phis, psis = xfm_murenn(Xt)
    xfm_np = dtcwt.Transform1d()
    out_np = xfm_np.forward(X, nlevels=J)
    phis_np = out_np.lowpass
    psis_np = out_np.highpasses
    np.testing.assert_array_almost_equal(phis_np[:,0], phis.numpy()[0,0,:], decimal = decimal)
    for j in range(J):
        np.testing.assert_array_almost_equal((psis_np[j])[:,0], psis[j].numpy()[0,0,:], decimal = decimal)