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


@pytest.mark.parametrize("alternate_gh", [True, False])
def test_phi(alternate_gh):
    '''
    Test the low-pass output phi doesn't diverge.
    '''
    tfm = murenn.DTCWT(alternate_gh=alternate_gh, include_scale=True, skip_hps=True)
    N = 2**15
    x = torch.ones(1, 1, N)
    phis, _ = tfm(x)
    for j, phi in enumerate(phis):
        assert torch.allclose(phi, torch.ones(1, 1, N // 2**j))


@pytest.mark.parametrize("alternate_gh", [True, False])
def test_energy_preservation(alternate_gh):
    '''
    Test Parseval’s energy theorem: the energy of the input signal 
    is equal to the energy in the wavelet domain.
    '''
    tfm = murenn.DTCWT(alternate_gh=alternate_gh, normalize=False)
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


@pytest.mark.parametrize("alternate_gh", [True, False])
def test_avrg_energy(alternate_gh):
    '''
    Test the power of the signals for normalization case.
    '''
    tfm = murenn.DTCWT(alternate_gh=alternate_gh, normalize=True)
    N = 2**15
    x = torch.randn(1 ,1, N)
    P_x = torch.linalg.norm(x) ** 2 / x.shape[-1]
    P_Ux = 0
    phi, psis = tfm(x)
    P_phi = torch.linalg.norm(phi) ** 2 / phi.shape[-1]
    P_Ux = P_Ux + P_phi
    for psi in psis:
        psi = psi / math.sqrt(2) #?
        Ppsi_j = torch.linalg.norm(torch.abs(psi)) ** 2 / psi.shape[-1]
        P_Ux = P_Ux + Ppsi_j
    ratio = P_Ux / P_x
    assert torch.abs(ratio - 1) <= 0.01


@pytest.mark.parametrize("J", list(range(1, 10)))
def test_subbands(J):
    tfm = murenn.DTCWT(J=J)
    subbands = tfm.subbands
    # Test the number of subbands
    # There are J band-pass subbands and 1 low-pass subband, so J+2 subbands boundaries in total.
    assert len(subbands) == J + 2
    # Test the min/max value
    assert min(subbands) == 0.
    assert max(subbands) == 0.5
    # Check that it's sorted
    assert all(subbands[i] > subbands[i+1] for i in range(len(subbands)-1)
    )

@pytest.mark.parametrize("J", list(range(1, 10)))
def test_hz_to_octs(J):
    sr = 16000
    nyquist = 8000
    dtcwt = murenn.DTCWT(J = J)
    # Test with a very small frequency, expecting it to map to the highest subband index
    assert dtcwt.hz_to_octs([1e-5], sr) == [J]
    # Test with a frequency just above the Nyquist frequency, expecting it to map to -1 (out of range)
    assert dtcwt.hz_to_octs([nyquist+1], sr) == [-1]
    # Test with a frequency just below the Nyquist frequency, expecting it to map to the lowest subband index
    assert dtcwt.hz_to_octs([nyquist-1], sr) == [0]