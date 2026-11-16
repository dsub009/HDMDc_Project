"""
Loading the PTFV Excel files from the experiment.

Expected layout (as in Extracted_Data_*.xlsx):
  sheet "Deflection_Data": columns time_d (s), Deflection_tip   (camera, ~1035.5 fps)
  sheet "Voltage_Data"   : columns time_v (s), voltage (V)      (DAQ, 1000 Hz)
"""
import os

import numpy as np
import pandas as pd


def _read_sheets(path):
    """
    Find the deflection and voltage sheets. Normally they are called
    "Deflection_Data" and "Voltage_Data"; otherwise match on 'defl' / 'volt'
    in the sheet name, and as a last resort use sheet 1 = deflection,
    sheet 2 = voltage (and say so).
    """
    sheets = pd.read_excel(path, sheet_name=None)
    names = list(sheets)
    defl = [n for n in names if "defl" in n.lower()]
    volt = [n for n in names if "volt" in n.lower()]
    if defl and volt:
        return sheets[defl[0]], sheets[volt[0]]
    if len(names) >= 2:
        print(f"  NOTE: {os.path.basename(path)} has sheets {names}; "
              f"assuming '{names[0]}' = deflection, '{names[1]}' = voltage")
        return sheets[names[0]], sheets[names[1]]
    raise ValueError(f"expected 2 sheets (deflection, voltage), found {names}")


def load_ptfv_excel(path, camera_fps=None):
    """
    Return dict with raw arrays: t_v, V, t_d, D (each 1D numpy).

    camera_fps : if given, rebuild the camera time axis as frame / camera_fps
        instead of using the time column in the file. Needed when the file's
        camera times were stretched to match the voltage length (e.g.
        0824 Baseline Run 2: file implies 1084.6 fps, true rate ~1035.6 fps).
    """
    d, v = _read_sheets(path)
    t_d = d.iloc[:, 0].to_numpy(float)
    if camera_fps is not None:
        t_d = t_d[0] + np.arange(len(t_d)) / camera_fps
    return dict(t_v=v.iloc[:, 0].to_numpy(float), V=v.iloc[:, 1].to_numpy(float),
                t_d=t_d, D=d.iloc[:, 1].to_numpy(float))


def deflection_on_voltage_time(rec, shift=0.0):
    """
    Interpolate deflection onto the voltage time base (camera and DAQ run at
    different rates). `shift` (s) moves the camera time axis, to correct a
    sync offset between camera and DAQ.
    """
    return np.interp(rec["t_v"], rec["t_d"] + shift, rec["D"])


def best_shift_per_window(rec, window=1.0, max_shift=0.03, step=2e-4):
    """
    For each `window`-second block, the camera time shift (s) that maximises
    |corr(V, D)|. If this wanders, the camera/DAQ sync is drifting.
    Returns list of (window_start, shift, corr).
    """
    tv, V = rec["t_v"], rec["V"]
    out = []
    for t0 in np.arange(tv[0], tv[-1] - window, window):
        m = (tv >= t0) & (tv < t0 + window)
        best = (0.0, 0.0)
        for sh in np.arange(-max_shift, max_shift + step / 2, step):
            c = np.corrcoef(V[m], np.interp(tv[m], rec["t_d"] + sh, rec["D"]))[0, 1]
            if abs(c) > abs(best[1]):
                best = (sh, c)
        out.append((t0, *best))
    return out
