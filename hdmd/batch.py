"""
Batch processing of a whole folder tree of PTFV Excel files.

File names are expected to look like
    0915_3.5mm_11.6_Run_3_RawData.xlsx
    <date>_<condition>_<station>_Run_<n>_RawData.xlsx
and can sit in any sub-folder (Station_.../<date_condition_station>/...).
PNG files and anything else are ignored.

For every run, analyse_file() returns one row of results (a dict). The
script experiments/05_all_conditions.py collects the rows into tables.
"""
import os
import re

import numpy as np
from scipy.signal import welch

from hdmd.io import load_ptfv_excel, deflection_on_voltage_time, best_shift_per_window
from hdmd.dmd import fit_hdmd, autocorrelation, pick_mode
from hdmd.metrics import psd_peak, half_power_damping

NAME_RE = re.compile(
    r"(?P<date>\d{4})_(?P<condition>.+?)_(?P<station>\d+(?:\.\d+)?)_Run_(?P<run>\d+)_RawData\.xlsx$",
    re.IGNORECASE)

DT, BAND = 1e-3, (20, 120)
SETTINGS = [(150, 30, 4), (200, 30, 4), (300, 40, 6)]   # (lags, delays, rank)
NOMINAL_FPS = 1035.6                                     # typical camera rate in these files


def find_runs(root):
    """All run files under `root`, parsed. Duplicate names (same test in two
    folders) are kept once; the list of skipped duplicates is returned too."""
    runs, seen, dupes = [], {}, []
    for folder, _, files in os.walk(root):
        for f in sorted(files):
            m = NAME_RE.search(f)
            if not m or f.startswith("~$"):            # ~$ = Excel lock file
                continue
            key = f.lower()
            path = os.path.join(folder, f)
            if key in seen:
                dupes.append((path, seen[key]))
                continue
            seen[key] = path
            info = m.groupdict()
            info.update(station=float(info["station"]), run=int(info["run"]),
                        file=f, path=path)
            runs.append(info)
    return runs, dupes


def condition_sort_key(cond):
    """Baseline first, then by the number in the name (2mm < 3mm < 3.5mm ...)."""
    if "base" in cond.lower():
        return (0, 0.0, cond)
    num = re.findall(r"\d+(?:\.\d+)?", cond)
    return (1, float(num[0]) if num else np.inf, cond)


def _hdmd_estimate(x, dt):
    e = []
    for L, d, r in SETTINGS:
        res = fit_hdmd(autocorrelation(x, L)[None, :], d, r, dt, min_ratio=0)
        i = pick_mode(res, BAND)
        e.append((abs(res["freq_hz"][i]), res["damping"][i]))
    e = np.array(e)
    return e[:, 0].mean(), e[:, 1].mean(), e[:, 1].std()   # f, zeta, zeta spread over settings


def analyse_file(path):
    """Quality checks + frequency/damping/RMS for one run. Returns a dict."""
    rec = load_ptfv_excel(path)
    out = {}
    fps = (len(rec["t_d"]) - 1) / (rec["t_d"][-1] - rec["t_d"][0])
    out["camera_fps_in_file"] = fps
    # QC 1: camera time column stretched (as in 0824 Baseline Run 2) -> rebuild
    out["camera_fps_fixed"] = abs(fps / NOMINAL_FPS - 1) > 0.02
    if out["camera_fps_fixed"]:
        rec = load_ptfv_excel(path, camera_fps=NOMINAL_FPS)
    V = rec["V"] - rec["V"].mean()
    D = deflection_on_voltage_time(rec)
    D = D - D.mean()
    out["duration_s"] = rec["t_v"][-1] - rec["t_v"][0]
    out["V_rms"] = V.std()
    out["D_rms_mm"] = D.std()
    # QC 2: does intensity change a lot during the run? (grid moving mid-run?)
    h = len(V) // 2
    out["rms_ratio_2nd_vs_1st_half"] = V[h:].std() / V[:h].std()
    # QC 3: camera <-> voltage agreement
    sh = best_shift_per_window(rec)
    out["corr_V_D"] = float(np.mean([abs(c) for *_, c in sh]))
    out["sync_drift_ms"] = 1000 * (max(s for _, s, _ in sh) - min(s for _, s, _ in sh))
    # QC 4: 150 Hz electrical line
    f, P = welch(V, fs=1 / DT, nperseg=4096)
    m = (f > 140) & (f < 160)
    out["line150_dB"] = 10 * np.log10(P[m].max() / np.median(P[(f > 120) & (f < 180)]))
    # Results
    out["f_V"], out["zeta_V"], out["zeta_V_settings_sd"] = _hdmd_estimate(V, DT)
    out["f_D"], out["zeta_D"], out["zeta_D_settings_sd"] = _hdmd_estimate(D, DT)
    fp, ff, PP = psd_peak(V, DT, 2048, BAND)
    out["f_peak_welch"] = fp
    out["zeta_halfpower_V"], _ = half_power_damping(ff, PP, BAND)
    # Flags summarising the QC
    flags = []
    if out["camera_fps_fixed"]:
        flags.append("camera time rebuilt")
    if out["corr_V_D"] < 0.9:
        flags.append("V/D disagree")
    if not (0.67 < out["rms_ratio_2nd_vs_1st_half"] < 1.5):
        flags.append("intensity changes mid-run")
    if out["line150_dB"] > 20:
        flags.append("150 Hz interference")
    out["flags"] = "; ".join(flags)
    return out
