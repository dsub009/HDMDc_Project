"""
Hankel DMD without control (output-only system identification).

WHICH SIGNAL TO FEED IT
-----------------------
* Free decay (flick test): feed the raw signal. Deterministic -> fine.
  method="fb" helps remove the noise bias in damping here.
* Turbulence-driven signal (static grid in the wind tunnel): DO NOT feed raw
  V. A randomly forced system never "decays", so HDMD on raw V gives the
  wrong damping (too low), and "fb" gives exactly zero damping. Instead
  feed the AUTOCORRELATION of V (see `autocorrelation`): for a linear
  system driven by broadband random forcing, the autocorrelation obeys the
  same free-decay dynamics (NExT: James et al. 1993; this is also the idea
  behind ERA / SSI-COV in operational modal analysis).
"""
import numpy as np
from scipy.linalg import sqrtm

from hdmd.hankel import create_hankel, check_hankel_size


def svht_rank(singular_values, shape):
    """
    Optimal hard threshold for singular values when the noise level is
    unknown (Gavish & Donoho 2014, IEEE Trans. Inf. Theory).
    Use it as a starting point, then confirm with the singular-value plot.
    """
    m, n = sorted(shape)
    beta = m / n
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    tau = omega * np.median(singular_values)
    return max(int(np.sum(singular_values > tau)), 1)


def continuous_eigs(eigs, dt):
    """Discrete eigenvalues -> frequency (Hz), growth rate (1/s), damping ratio."""
    eigs = np.asarray(eigs, dtype=complex)          # avoid log(negative real) = nan
    omega = np.log(eigs) / dt
    freq_hz = np.imag(omega) / (2 * np.pi)
    growth = np.real(omega)
    damping = -growth / np.abs(omega)               # zeta = -Re(s)/|s|
    return omega, freq_hz, growth, damping


def autocorrelation(x, max_lag, n_segments=8):
    """
    Normalised autocorrelation R(0..max_lag-1) of a 1D signal, averaged over
    n_segments blocks (reduces the random error, like Welch averaging).
    Choose max_lag so R has decayed to a few % (about 3 decay times,
    3 / (zeta * 2*pi*fn)); beyond that the estimate is mostly noise.
    For the Lee et al. beam (fn ~ 51 Hz, zeta ~ 0.03) that is ~300 lags at 1 kHz.
    """
    x = np.asarray(x, float) - np.mean(x)
    L = len(x) // n_segments
    acc = np.zeros(max_lag)
    for s in range(n_segments):
        seg = x[s * L:(s + 1) * L]
        F = np.fft.rfft(seg, 2 * L)                 # zero-pad: linear, not circular
        acc += np.fft.irfft(np.abs(F) ** 2)[:max_lag] / L
    return acc / acc[0]


def fit_hdmd(X, delays, rank=None, dt=1.0, method="standard", min_ratio=10):
    """
    Fit Hankel DMD to (n_channels, n_samples) data.

    Parameters
    ----------
    X : array, (n_channels, n_samples). Subtract the mean and, for several
        channels with different units, divide each by its std first.
    delays : number of time-delay copies.
    rank : SVD truncation. None -> automatic (svht_rank).
    dt : sample time (s), so frequencies come out in Hz.
    method : "standard" (least squares, Schmid 2010) or "fb"
        (forward-backward DMD, Dawson et al. 2016). "fb" removes the noise
        bias in damping for DETERMINISTIC (free-decay) data only.
    min_ratio : column/row ratio below which a warning is printed. For a
        smooth deterministic signal (decay, autocorrelation) 2-3 is enough.

    Returns
    -------
    dict with eigs, freq_hz, growth, damping, modes (exact DMD modes),
    energy (mean |mode coefficient|^2 over the data), rank, singular_values.
    """
    H = create_hankel(X, delays)
    check_hankel_size(np.atleast_2d(X).shape[1], delays, H.shape[0], min_ratio)
    Xc, Yc = H[:, :-1], H[:, 1:]

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if rank is None:
        rank = svht_rank(S, Xc.shape)
    r = min(rank, len(S))
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].T

    # Reduced operator in the POD basis of the CURRENT snapshots (Schmid 2010).
    A_f = Ur.T @ Yc @ Vr / Sr
    if method == "standard":
        A_tilde = A_f
    elif method == "fb":
        xt, yt = Ur.T @ Xc, Ur.T @ Yc
        A_b = xt @ np.linalg.pinv(yt)               # backward map y -> x
        A_tilde = sqrtm(A_f @ np.linalg.inv(A_b))
    else:
        raise ValueError("method must be 'standard' or 'fb'")

    eigs, W = np.linalg.eig(A_tilde)
    modes = Yc @ Vr @ np.diag(1 / Sr) @ W           # exact DMD modes (Tu et al. 2014)
    coeffs = np.linalg.lstsq(modes, Xc, rcond=None)[0]
    energy = np.mean(np.abs(coeffs) ** 2, axis=1) * np.linalg.norm(modes, axis=0) ** 2

    omega, f, g, z = continuous_eigs(eigs, dt)
    return dict(eigs=eigs, omega=omega, freq_hz=f, growth=g, damping=z, dt=dt,
                modes=modes, energy=energy, rank=r, singular_values=S,
                A_tilde=A_tilde)


def pick_mode(result, f_band=(None, None)):
    """
    Return the index of the most energetic oscillatory mode inside a
    frequency band (e.g. (30, 80) Hz around the beam's first mode).
    Modes at ~0 Hz (mean) and at Nyquist (noise) are skipped.
    """
    f = np.abs(result["freq_hz"])
    nyquist = 0.5 / result["dt"]
    ok = (f > 1e-3) & (f < 0.98 * nyquist)
    lo, hi = f_band
    if lo is not None:
        ok &= f >= lo
    if hi is not None:
        ok &= f <= hi
    if not np.any(ok):
        raise ValueError("No oscillatory mode in the requested band.")
    idx = np.where(ok)[0]
    return idx[np.argmax(result["energy"][idx])]
