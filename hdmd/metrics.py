import numpy as np
from scipy.signal import welch


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def nrmse(actual, predicted):
    """RMSE / std(actual). Note: nrmse = sqrt(1 - R^2), so report one of them."""
    s = np.std(actual)
    if s < 1e-12:
        raise ValueError("actual has ~zero variance")
    return rmse(actual, predicted) / s


def r_squared(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    if ss_tot < 1e-12:
        raise ValueError("actual has ~zero variance")
    return float(1 - np.sum((actual - predicted) ** 2) / ss_tot)


def psd_peak(signal, dt, nperseg=4096, f_band=(None, None)):
    """
    Dominant frequency from a Welch PSD (averaged, so much less noisy than a
    single FFT of a 60 s turbulent record). Returns (f_peak, f, Pxx).
    """
    f, P = welch(signal - np.mean(signal), fs=1 / dt, nperseg=nperseg)
    ok = np.ones_like(f, bool) & (f > 0)
    if f_band[0] is not None: ok &= f >= f_band[0]
    if f_band[1] is not None: ok &= f <= f_band[1]
    i = np.where(ok)[0][np.argmax(P[ok])]
    return f[i], f, P


def half_power_damping(f, P, f_band=(None, None)):
    """
    Damping ratio from the half-power bandwidth of a PSD peak:
    zeta ~= (f_u - f_l) / (2 f_p). This is the classical estimate; compare it
    with the HDMD damping. (On a POWER spectrum the half-power level is P/2;
    on an amplitude spectrum it is A/sqrt(2), as in Lee et al. 2021 Eq. 6.)
    """
    ok = np.ones_like(f, bool)
    if f_band[0] is not None: ok &= f >= f_band[0]
    if f_band[1] is not None: ok &= f <= f_band[1]
    idx = np.where(ok)[0]
    ip = idx[np.argmax(P[idx])]
    half = P[ip] / 2
    il = ip
    while il > 0 and P[il] > half: il -= 1
    iu = ip
    while iu < len(P) - 1 and P[iu] > half: iu += 1
    # linear interpolation of the crossing points
    fl = np.interp(half, [P[il], P[il + 1]], [f[il], f[il + 1]])
    fu = np.interp(half, [P[iu], P[iu - 1]], [f[iu], f[iu - 1]])
    return (fu - fl) / (2 * f[ip]), f[ip]
