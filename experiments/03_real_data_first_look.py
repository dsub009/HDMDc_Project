"""
Experiment 3 - first look at REAL PTFV data (no-morphing vs morphing).

Applies the methods verified in experiments 01/02:
  * Welch spectrum + half-power damping (classical)
  * HDMD on the autocorrelation (verified: ~1 % damping bias)
  * camera <-> DAQ sync check (best time shift per 1 s window)
  * short-time RMS of V (shows when the turbulence intensity changes)

Run from the project folder:  python experiments/03_real_data_first_look.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from plotstyle import plt, C, show

from hdmd.io import load_ptfv_excel, deflection_on_voltage_time, best_shift_per_window
from hdmd.dmd import fit_hdmd, autocorrelation, pick_mode
from hdmd.metrics import psd_peak, half_power_damping

DT, BAND = 1e-3, (20, 120)
FILES = {"No morphing": "Extracted_Data_No_Morphing.xlsx",
         "Morphing": "Extracted_Data_Morphing.xlsx"}
COL = {"No morphing": C["blue"], "Morphing": C["orange"]}

def find_data(fname):
    """Look in data/raw/ first, then data/."""
    for folder in [os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data")]:
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{fname} not found in data/raw/ or data/")


recs = {k: load_ptfv_excel(find_data(f)) for k, f in FILES.items()}
results = {}
for name, rec in recs.items():
    V = rec["V"]; D = deflection_on_voltage_time(rec)
    print(f"\n=== {name}: {len(V)} voltage samples ({rec['t_v'][-1]:.2f} s), "
          f"camera fs = {1/np.mean(np.diff(rec['t_d'])):.1f} fps")
    print(f"  V rms = {V.std():.3f} V,  deflection rms = {D.std():.3f} (units: ask!)")
    row = {}
    for sig_name, x in [("V", V), ("deflection", D)]:
        fp, f, P = psd_peak(x, DT, 2048, BAND)
        zh, _ = half_power_damping(f, P, BAND)
        ests = []
        for L, d, r in [(150, 30, 4), (200, 30, 4), (300, 40, 6)]:     # sensitivity check
            res = fit_hdmd(autocorrelation(x, L)[None, :], d, r, DT, min_ratio=0)
            i = pick_mode(res, BAND)
            ests.append((abs(res["freq_hz"][i]), res["damping"][i]))
        ests = np.array(ests)
        print(f"  {sig_name:10s}: Welch peak {fp:5.2f} Hz, half-power zeta {zh:.4f} | "
              f"autocorr-HDMD f = {ests[:,0].mean():.2f} ± {ests[:,0].std():.2f} Hz, "
              f"zeta = {ests[:,1].mean():.4f} ± {ests[:,1].std():.4f}")
        row[sig_name] = (f, P, ests)
    sh = best_shift_per_window(rec)
    print("  sync check, best camera shift per 1 s window (ms):",
          " ".join(f"{s*1000:+.1f}" for _, s, _ in sh))
    print("  |corr(V, defl)| after per-window alignment:",
          " ".join(f"{abs(c):.3f}" for *_, c in sh))
    results[name] = (row, sh)

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
a = ax[0, 0]
for name, (row, _) in results.items():
    f, P, _ = row["V"]
    a.semilogy(f, P, color=COL[name], lw=1.5, label=name)
a.axvline(50.78, color=C["muted"], ls="--", lw=1)
a.text(53, 1e-7, "flick test\n50.78 Hz", color=C["muted"], fontsize=8)
a.annotate("150 Hz line: electrical\ninterference (V only)", xy=(150, 3e-3), xytext=(158, 3e-2), fontsize=8, color=C["muted"], arrowprops=dict(arrowstyle="-", color=C["muted"], lw=1))
a.set(xlim=(0, 200), xlabel="Frequency (Hz)", ylabel="PSD of V (V²/Hz)", title="Voltage spectrum")
a.legend(loc="lower left")

a = ax[0, 1]
for j, (name, (row, _)) in enumerate(results.items()):
    for k, sig_name in enumerate(["V", "deflection"]):
        z = row[sig_name][2][:, 1]
        a.errorbar(j * 3 + k, z.mean(), yerr=z.std(), fmt="o", ms=9, color=COL[name],
                   mec="#fcfcfb", mew=2, capsize=4)
a.axhspan(0.012, 0.0716, color=C["grid"], alpha=0.6, lw=0)
a.text(4.6, 0.069, "Lee et al. 2021 range", color=C["muted"], fontsize=8, ha="right", va="top")
a.set_xticks([0, 1, 3, 4], ["V", "defl.", "V", "defl."])
a.set(xlim=(-0.7, 4.8), ylim=(0, 0.08), ylabel="Damping ratio ζ (autocorr-HDMD)",
      title="Damping: no morphing (blue) vs morphing (orange)")

a = ax[1, 0]
for name, rec in recs.items():
    V = rec["V"]; w = 250
    rms = np.array([V[i:i + w].std() for i in range(0, len(V) - w + 1, w)])
    a.step(np.arange(len(rms)) * w * DT, rms, where="post", color=COL[name], label=name)
a.set(xlabel="Time (s)", ylabel="RMS of V per 0.25 s (V)", title="Turbulence intensity over time")
a.legend(loc="upper left")

a = ax[1, 1]
for name, (_, sh) in results.items():
    a.plot([t + 0.5 for t, *_ in sh], [s * 1000 for _, s, _ in sh], "o-", color=COL[name],
           ms=7, mec="#fcfcfb", mew=2, label=name)
a.axhline(0, color=C["muted"], lw=1)
a.set(xlabel="Time (s)", ylabel="Best camera time shift (ms)",
      title="Camera ↔ DAQ sync: should be a flat line")
a.legend(loc="lower right")
fig.tight_layout()
out = os.path.join(ROOT, "figures", "03_real_data_first_look.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150); print("\nsaved", out)
show()
