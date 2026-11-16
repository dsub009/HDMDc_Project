"""
Synthetic morphing-grid experiment.

Grid blockage sigma(t) switches randomly between the min/max of the N=1 SSG
(0.335 - 0.402, FYP slides). Turbulence intensity follows sigma with a
transport delay and a lag (ASSUMED values - these are exactly what the
real experiment should measure). The beam is then driven by turbulence whose
RMS is modulated by sigma.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from simulations.piezo_beam import simulate_ptfv

SIGMA_MIN, SIGMA_MAX = 0.335, 0.402


def random_step_input(duration, dt, hold=(0.5, 3.0), stroke_time=0.4, seed=1):
    """
    Random step sequence between SIGMA_MIN and SIGMA_MAX, rate-limited to
    mimic the linear actuator (full stroke in `stroke_time` s).
    This is the kind of input to program on the Arduino: persistently
    exciting, unlike a single slow sine.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    target = np.empty(n)
    i = 0
    while i < n:
        L = int(rng.uniform(*hold) / dt)
        target[i:i + L] = rng.uniform(SIGMA_MIN, SIGMA_MAX)
        i += L
    rate = (SIGMA_MAX - SIGMA_MIN) / stroke_time * dt
    sig = np.empty(n); sig[0] = target[0]
    for i in range(1, n):
        sig[i] = sig[i - 1] + np.clip(target[i] - sig[i - 1], -rate, rate)
    return sig


def intensity_response(sigma, dt, gain=6.0, delay=0.075, tau=0.2):
    """
    Turbulence-intensity multiplier seen by the sensor (ASSUMED model):
    target = 1 + gain*(sigma - SIGMA_MIN), then transport delay (e.g.
    0.3 m / 4 m/s = 75 ms) and a first-order lag tau.
    """
    target = 1 + gain * (sigma - SIGMA_MIN)
    nd = int(round(delay / dt))
    target = np.concatenate([np.full(nd, target[0]), target[:len(target) - nd]])
    a = np.exp(-dt / tau)
    out = np.empty_like(target); out[0] = target[0]
    for i in range(1, len(target)):
        out[i] = a * out[i - 1] + (1 - a) * target[i]
    return out


def simulate_morphing(duration=300.0, dt=1e-3, seed=0, **kw):
    sigma = random_step_input(duration, dt, seed=seed + 1)
    scale = intensity_response(sigma, dt, **kw)
    t, d, V, _ = simulate_ptfv(duration, dt, force_scale=scale, seed=seed)
    return t, sigma, scale, d, V


def rms_envelope(V, dt, f_cut=5.0, decimate=20):
    """
    Short-time RMS of V: low-pass V^2 (zero-phase), sqrt, then downsample.
    f_cut must be well below the beam frequency (~50 Hz) but above the
    speed at which you morph the grid.
    Returns (envelope, new_dt).
    """
    sos = butter(4, f_cut, fs=1 / dt, output="sos")
    env = np.sqrt(np.clip(sosfiltfilt(sos, (V - V.mean()) ** 2), 0, None))
    return env[::decimate], dt * decimate
