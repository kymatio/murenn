import pytest
import torch
import numpy as np
import dtcwt
import murenn
import math


@pytest.mark.parametrize("J", list(range(1, 10)))
def test_fwd_same(J):
    decimal = 4
    X = np.random.rand(44100)
    Xt = torch.tensor(X, dtype=torch.get_default_dtype()).view(1, 1, 44100)
    xfm_murenn = murenn.DTCWTDirect(
        J=J,
        include_scale=False,
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
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("J", list(range(1, 5)))
@pytest.mark.parametrize("T", [44099, 44100])
def test_pr(level1, qshift, J, T, normalize):
    Xt = torch.randn(2, 2, T)
    xfm_murenn = murenn.DTCWTDirect(
        J=J,
        level1=level1,
        qshift=qshift,
        include_scale=False,
        normalize=normalize,
    )
    lp, bp = xfm_murenn(Xt)
    inv = murenn.DTCWTInverse(
        J=J,
        level1=level1,
        qshift=qshift,
        include_scale=False,
        normalize=normalize,
        length=T,
    )
    X_rec = inv(lp, bp)
    torch.testing.assert_close(Xt, X_rec)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("skip_hps", [False, [0, 1, 0]])
def test_skip_hps(skip_hps, normalize):
    J = 3
    Xt = torch.randn(2, 2, 44100)
    xfm_murenn = murenn.DTCWTDirect(J=J, skip_hps=skip_hps, normalize=normalize)
    lp, bp = xfm_murenn(Xt)
    inv = murenn.DTCWTInverse(J=J, skip_hps=skip_hps, normalize=normalize)
    X_rec = inv(lp, bp)
    assert X_rec.shape == Xt.shape
    xfm_allpass = murenn.DTCWTDirect(J=J, normalize=normalize)
    lp_ap, _ = xfm_allpass(Xt)
    assert torch.allclose(lp, lp_ap)


@pytest.mark.parametrize("J", range(1,4))
def test_phi(J):
    '''
    Test the low-pass output phi doesn't diverge.
    '''
    tfm = murenn.DTCWT(J=J, include_scale=True, skip_hps=True)
    N = 2**15
    x = torch.ones(1, 1, N)
    phis, _ = tfm(x)
    for j, phi in enumerate(phis):
        assert torch.allclose(phi, torch.ones(1, 1, N // 2**j))


def test_energy_preservation():
    '''
    Test Parseval’s energy theorem: the energy of the input signal 
    is equal to the energy in the wavelet domain.
    '''
    tfm = murenn.DTCWT(normalize=False)
    N = 2**15
    x = torch.randn(1 ,1, N)
    E_x = torch.linalg.norm(x) ** 2
    E_Ux = 0
    phi, psis = tfm(x)
    E_phi = torch.linalg.norm(phi) ** 2
    E_Ux = E_Ux + E_phi
    for psi in psis:
        Epsi_j = torch.linalg.norm(torch.abs(psi)) ** 2
        E_Ux = E_Ux + Epsi_j
    ratio = E_Ux / E_x
    assert torch.abs(ratio - 1) <= 0.01


@pytest.mark.parametrize("J", range(1, 4))
def test_avrg_energy(J):
    '''
    Test the power of the signals for normalization case.
    '''
    tfm = murenn.DTCWT(J=J, normalize=True)
    N = 2**15
    x = torch.randn(1 ,1, N)
    P_x = torch.linalg.norm(x) ** 2 / x.shape[-1]
    P_Ux = 0
    phi, psis = tfm(x)
    P_phi = torch.linalg.norm(phi) ** 2 / phi.shape[-1]
    P_Ux = P_Ux + P_phi
    for psi in psis:
        psi = psi / math.sqrt(2)
        Ppsi_j = torch.linalg.norm(torch.abs(psi)) ** 2 / psi.shape[-1]
        P_Ux = P_Ux + Ppsi_j
    ratio = P_Ux / P_x
    assert torch.abs(ratio - 1) <= 0.01


def test_default_args():
    Xt = torch.randn(2, 2, 16000)
    xfm_murenn = murenn.DTCWTDirect()
    lp, bp = xfm_murenn(Xt)
    inv = murenn.DTCWTInverse()
    X_rec = inv(lp, bp)
    torch.testing.assert_close(Xt, X_rec)


def test_udtcwt_shape():
    '''
    Test that the UDT-CWT runs without error and produces outputs of the expected shape.
    '''
    J = 3
    N = 2**15
    x = torch.randn(1, 2, N)
    tfm = murenn.UDTCWT(J=J)
    phi, psis = tfm(x)
    assert len(psis) == J
    assert phi.shape == (1, 2, N)
    for j in range(J):
        assert psis[j].shape == (1, 2, N)