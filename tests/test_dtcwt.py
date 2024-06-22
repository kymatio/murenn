import pytest
import torch
import numpy as np
import dtcwt
import murenn


@pytest.mark.parametrize("J", list(range(1, 10)))
def test_fwd_same(J):
    decimal = 4
    X = np.random.rand(44100)
    Xt = torch.tensor(X, dtype=torch.get_default_dtype()).view(1, 1, 44100)
    xfm_murenn = murenn.DTCWTDirect(
        J=J,
        alternate_gh=False,
        include_scale=False,
        padding_mode="symmetric",
        normalize=False,
    )
    phis, psis = xfm_murenn(Xt)
    xfm_np = dtcwt.Transform1d()
    out_np = xfm_np.forward(X, nlevels=J)
    phis_np = out_np.lowpass
    psis_np = out_np.highpasses
    np.testing.assert_array_almost_equal(
        phis_np[:, 0], phis.numpy()[0, 0, :], decimal=decimal
    )
    for j in range(J):
        np.testing.assert_array_almost_equal(
            (psis_np[j])[:, 0], psis[j].numpy()[0, 0, :], decimal=decimal
        )


@pytest.mark.parametrize(
    "qshift", ["qshift_06", "qshift_a", "qshift_b", "qshift_c", "qshift_d"]
)
@pytest.mark.parametrize("level1", ["antonini", "legall", "near_sym_a", "near_sym_b"])
@pytest.mark.parametrize("alternate_gh", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("J", list(range(1, 5)))
@pytest.mark.parametrize("T", [44099, 44100])
def test_pr(level1, qshift, J, T, alternate_gh, normalize):
    Xt = torch.randn(2, 2, T)
    xfm_murenn = murenn.DTCWTDirect(
        J=J,
        level1=level1,
        qshift=qshift,
        alternate_gh=alternate_gh,
        include_scale=False,
        padding_mode="symmetric",
        normalize=normalize,
    )
    lp, bp = xfm_murenn(Xt)
    inv = murenn.DTCWTInverse(
        J=J,
        level1=level1,
        qshift=qshift,
        alternate_gh=alternate_gh,
        include_scale=False,
        padding_mode="symmetric",
        normalize=normalize,
        length=T,
    )
    X_rec = inv(lp, bp)
    torch.testing.assert_close(Xt, X_rec)


@pytest.mark.parametrize("include_scale", [False, [0, 0, 1]])
@pytest.mark.parametrize("skip_hps", [False, [0, 1, 0]])
def test_skip_hps(skip_hps, include_scale):
    J = 3
    Xt = torch.randn(2, 2, 44100)
    xfm_murenn = murenn.DTCWTDirect(J=J, skip_hps=skip_hps, include_scale=include_scale)
    lp, bp = xfm_murenn(Xt)
    inv = murenn.DTCWTInverse(J=J, skip_hps=skip_hps, include_scale=include_scale)
    X_rec = inv(lp, bp)
    assert X_rec.shape == Xt.shape

def test_inv():
    T = 2**10
    Xt = torch.randn(2, 2, T)
    dtcwt = murenn.DTCWTDirect()
    idtcwt = murenn.DTCWTInverse()
    lp, bp = dtcwt(Xt)
    lp = lp.new_zeros(lp.shape)
    X_rec = idtcwt(lp, bp)
    bp_r = [(bp[j].real)*(1+0j) for j in range(8)]
    bp_i = [(bp[j].imag)*(0+1j) for j in range(8)]
    X_rec_r = idtcwt(lp, bp_r)
    X_rec_i = idtcwt(lp, bp_i)
    assert torch.allclose((X_rec_r+X_rec_i), X_rec, atol=1e-3)