"""
Synthetic PTFV signal with a KNOWN answer, for verifying HDMD.

Mechanics (Lee et al. 2021, Eq. 3):   m d'' + c d' + k d - theta V = F_y
Electrics (resistive load R, film capacitance Cp):  Cp V' + V/R = -theta d'

Parameters from Lee et al. 2021 Table 3 unless marked ASSUMED.
The forcing F_y is coloured random noise (stands in for turbulence).
Everything is discretised exactly (zero-order hold) at the DAQ rate, so the
true discrete eigenvalues are exactly exp(eig(Ac) * dt).
"""
import numpy as np
from scipy.linalg import expm

LEE2021 = dict(
    m=0.17e-3,          # kg      (Table 3, measured)
    fn=50.78,           # Hz      (Table 3, flick test)
    zeta=0.03,          # -       (Table 3 range 0.012-0.072; pick one)
    theta=7.69e-6,      # C/m     (Table 3)
    Cp=1.0e-9,          # F       ASSUMED - replace with film datasheet value
    R=10e6,             # ohm     (10 MOhm load, Lee et al. 2023)
)


def beam_matrices(p=LEE2021):
    m, zeta, theta, Cp, R = p["m"], p["zeta"], p["theta"], p["Cp"], p["R"]
    k = m * (2 * np.pi * p["fn"]) ** 2
    c = 2 * zeta * np.sqrt(k * m)
    # state s = [d, d', V]
    Ac = np.array([[0, 1, 0],
                   [-k / m, -c / m, theta / m],
                   [0, -theta / Cp, -1 / (R * Cp)]])
    Bc = np.array([[0], [1 / m], [0]])
    return Ac, Bc


def true_beam_mode(p=LEE2021):
    """Ground-truth damped frequency (Hz) and damping ratio of the coupled beam."""
    Ac, _ = beam_matrices(p)
    s = np.linalg.eigvals(Ac)
    s = s[np.argmax(np.abs(s.imag))]
    return abs(s.imag) / (2 * np.pi), -s.real / abs(s)


def discretise(Ac, Bc, dt):
    n, q = Bc.shape
    M = expm(np.block([[Ac, Bc], [np.zeros((q, n + q))]]) * dt)
    return M[:n, :n], M[:n, n:]


def coloured_noise(n, dt, f_corner, rng):
    """Unit-variance white noise through a 1st-order low-pass (an AR(1) process)."""
    a = np.exp(-2 * np.pi * f_corner * dt)
    w = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + w[i]
    return x / np.std(x)


def simulate_ptfv(duration=60.0, dt=1e-3, F_rms=5e-3, snr=47.3,
                  force_scale=None, f_corner=100.0, p=LEE2021, seed=0):
    """
    Simulate tip deflection d (m) and voltage V (V).

    duration, dt : 60 s at 1 kHz matches Lee et al. 2021/2023.
    F_rms : RMS turbulent force (N).
    snr : RMS(V)/RMS(noise); 47.3 is the value reported by Lee et al. 2021.
          Use np.inf for noise-free. Includes 4.88 mV DAQ quantisation.
    force_scale : optional array (n,) multiplying the force, i.e. a
          time-varying turbulence intensity (used for the morphing grid).
    Returns t, d, V_measured, V_clean.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration / dt))
    t = np.arange(n) * dt
    F = F_rms * coloured_noise(n, dt, f_corner, rng)
    if force_scale is not None:
        F = F * force_scale
    Ad, Bd = discretise(*beam_matrices(p), dt)
    s = np.zeros(3)
    S = np.zeros((n, 3))
    for i in range(n):
        S[i] = s
        s = Ad @ s + Bd[:, 0] * F[i]
    d, V = S[:, 0], S[:, 2]
    Vm = V.copy()
    if np.isfinite(snr):
        Vm = Vm + rng.standard_normal(n) * np.std(V) / snr
        Vm = np.round(Vm / 4.88e-3) * 4.88e-3        # LabJack U3-HV resolution
    return t, d, Vm, V
