"""
Verification tests: does the code recover KNOWN answers?
Run from the project folder:   python -m pytest tests -q
"""
import warnings
import numpy as np
from scipy.signal import cont2discrete

from hdmd.hankel import create_hankel
from hdmd.dmd import fit_hdmd, pick_mode, autocorrelation
from hdmd.hdmdc import fit_hdmdc, predict_hdmdc
from simulations.piezo_beam import simulate_ptfv, true_beam_mode


def test_hankel_layout():
    X = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]], float)
    H = create_hankel(X, 3)
    assert H.shape == (6, 3)
    assert np.allclose(H[:, 0], [1, 10, 2, 20, 3, 30])     # newest sample last
    assert np.allclose(H[:, -1], [3, 30, 4, 40, 5, 50])


def test_hdmd_recovers_damped_sinusoids_exactly():
    dt = 1e-3
    t = np.arange(3000) * dt
    x = (np.exp(-5 * t) * np.cos(2 * np.pi * 50 * t)
         + 0.5 * np.exp(-2 * t) * np.sin(2 * np.pi * 12 * t))
    res = fit_hdmd(x[None, :], delays=10, rank=4, dt=dt)
    f = np.sort(np.unique(np.round(np.abs(res["freq_hz"]), 6)))
    g = np.sort(np.unique(np.round(res["growth"], 6)))
    assert np.allclose(f, [12, 50], atol=1e-6)
    assert np.allclose(g, [-5, -2], atol=1e-6)


def test_hdmdc_recovers_a_known_controlled_system():
    rng = np.random.default_rng(0)
    # 2-DOF lightly damped system with one input, ZOH discretised
    Ac = np.array([[0, 1, 0, 0], [-400, -2, 100, 0],
                   [0, 0, 0, 1], [100, 0, -900, -3]], float)
    Bc = np.array([[0], [1], [0], [0]], float)
    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)      # measure positions only
    dt = 0.01
    Ad, Bd, *_ = cont2discrete((Ac, Bc, C, np.zeros((2, 1))), dt)
    n = 4000
    u = rng.standard_normal((1, n))                         # persistently exciting
    s = np.zeros(4); Y = np.zeros((2, n))
    for k in range(n):
        Y[:, k] = C @ s
        s = Ad @ s + Bd[:, 0] * u[0, k]
    tr = 3000
    # rank_out must cover system order (4) + input feed-through ((delays-1)*n_in = 2)
    m = fit_hdmdc(Y[:, :tr], u[:, :tr], delays=3, rank_in=9, rank_out=6, dt=dt)
    for lam in np.linalg.eigvals(Ad):                       # every true eig is found
        assert np.min(np.abs(m["eigs"] - lam)) < 1e-8
    # open-loop prediction on unseen data
    d = m["delays"]
    Yp = predict_hdmdc(m, Y[:, tr:tr + d], u[:, tr:])
    assert np.max(np.abs(Yp - Y[:, tr:])) < 1e-6


def test_matches_pydmd():
    from pydmd import DMD
    dt = 1e-3
    t = np.arange(2000) * dt
    x = np.exp(-3 * t) * np.cos(2 * np.pi * 40 * t)
    H = create_hankel(x[None, :], 8)
    ref = DMD(svd_rank=2, exact=True).fit(H)
    ours = fit_hdmd(x[None, :], delays=8, rank=2, dt=dt)
    assert np.allclose(np.sort_complex(ref.eigs), np.sort_complex(ours["eigs"]), atol=1e-10)


def test_sine_input_is_not_persistently_exciting():
    """Why the original synthetic example could not identify B."""
    t = np.arange(2000) * 0.01
    s = np.linalg.svd(create_hankel(np.sin(0.8 * t)[None, :], 10), compute_uv=False)
    assert np.sum(s > 1e-8 * s[0]) == 2          # rank 2 no matter how many delays


def test_flick_test_decay_with_noise():
    """Free decay + noise: fb-DMD recovers damping; standard is biased."""
    rng = np.random.default_rng(1)
    dt = 1e-3
    t = np.arange(1500) * dt
    f, z = 50.78, 0.03
    wn = 2 * np.pi * f / np.sqrt(1 - z**2)
    x = np.exp(-z * wn * t) * np.cos(2 * np.pi * f * t) + 0.02 * rng.standard_normal(len(t))
    res = fit_hdmd(x[None, :], delays=30, rank=2, dt=dt, method="fb", min_ratio=2)
    i = pick_mode(res)
    assert abs(abs(res["freq_hz"][i]) - f) < 0.2
    assert abs(res["damping"][i] - z) / z < 0.10


def test_raw_turbulent_voltage_gives_wrong_damping():
    """Documents the pitfall: HDMD on RAW forced response under-estimates zeta."""
    f_true, z_true = true_beam_mode()
    _, _, V, _ = simulate_ptfv(duration=60, snr=47.3, seed=3)
    res = fit_hdmd((V - V.mean())[None, :], delays=40, rank=6, dt=1e-3)
    i = pick_mode(res, (30, 80))
    assert res["damping"][i] < 0.5 * z_true


def test_beam_mode_from_autocorrelation_of_noisy_voltage():
    """The main verification: realistic noisy PTFV voltage -> fn, zeta."""
    f_true, z_true = true_beam_mode()
    for seed in range(3):
        _, _, V, _ = simulate_ptfv(duration=60, snr=47.3, seed=seed)
        r = autocorrelation(V, 300)
        res = fit_hdmd(r[None, :], delays=40, rank=6, dt=1e-3, min_ratio=2)
        i = pick_mode(res, (30, 80))
        assert abs(abs(res["freq_hz"][i]) - f_true) < 0.5          # Hz
        assert abs(res["damping"][i] - z_true) / z_true < 0.25      # within 25 %
