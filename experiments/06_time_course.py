"""
Experiment 6 - HOW DOES TURBULENCE STRENGTH CHANGE DURING EACH RUN?

The batch results showed that in the morphing conditions the signal
strength changes systematically within the 10 s runs (2mm/3mm drop by
about half, 4.5mm rises, Baseline stays steady). This script measures that
time course:
  * RMS of deflection in 0.5 s windows for every run
  * averaged over the repeats of each (condition, station)
  * "early" (first 2 s) and "late" (last 4 s) RMS, so conditions can be
    compared in a settled state instead of mixing both.

Needs the raw data in data/ (same as experiment 5).
Outputs: results/time_course.csv, results/early_late.csv,
         figures/06_time_course.png
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from plotstyle import plt, C, show

from hdmd.batch import find_runs, condition_sort_key, NOMINAL_FPS
from hdmd.io import load_ptfv_excel, deflection_on_voltage_time

WIN = 0.5            # s, RMS window
EARLY, LATE = 2.0, 4.0
RES = os.path.join(ROOT, "results"); os.makedirs(RES, exist_ok=True)
OUT_TC = os.path.join(RES, "time_course.csv")

runs, _ = find_runs(os.path.join(ROOT, "data"))
print(f"Found {len(runs)} runs.")
if os.path.exists(OUT_TC):
    tc = pd.read_csv(OUT_TC)
    print("Using saved results/time_course.csv (delete it to recompute).")
else:
    rows = []
    for k, r in enumerate(runs, 1):
        try:
            rec = load_ptfv_excel(r["path"])
            fps = (len(rec["t_d"]) - 1) / (rec["t_d"][-1] - rec["t_d"][0])
            if abs(fps / NOMINAL_FPS - 1) > 0.02:
                rec = load_ptfv_excel(r["path"], camera_fps=NOMINAL_FPS)
            t = rec["t_v"] - rec["t_v"][0]
            V = rec["V"] - rec["V"].mean()
            D = deflection_on_voltage_time(rec); D = D - D.mean()
        except Exception as e:
            print(f"  skip {r['file']}: {e}")
            continue
        for t0 in np.arange(0, t[-1] - WIN + 1e-9, WIN):
            m = (t >= t0) & (t < t0 + WIN)
            rows.append(dict(file=r["file"], condition=r["condition"], station=r["station"],
                             run=r["run"], t=t0 + WIN / 2, D_rms=D[m].std(), V_rms=V[m].std()))
        if k % 20 == 0:
            print(f"  {k}/{len(runs)}")
    tc = pd.DataFrame(rows)
    tc.to_csv(OUT_TC, index=False)

# ---- early vs late per run, then per condition/station ----
def early_late(g):
    tend = g["t"].max()
    return pd.Series(dict(D_early=g.loc[g.t < EARLY, "D_rms"].mean(),
                          D_late=g.loc[g.t > tend - LATE, "D_rms"].mean()))
per_run = tc.groupby(["condition", "station", "run", "file"]).apply(early_late).reset_index()
per_run["late_over_early"] = per_run.D_late / per_run.D_early
el = per_run.groupby(["condition", "station"]).agg(
    n=("run", "size"),
    D_early_mean=("D_early", "mean"), D_early_sd=("D_early", "std"),
    D_late_mean=("D_late", "mean"), D_late_sd=("D_late", "std"),
    late_over_early=("late_over_early", "mean")).reset_index()
el["_k"] = el.condition.map(condition_sort_key)
el = el.sort_values(["_k", "station"]).drop(columns="_k")
el.to_csv(os.path.join(RES, "early_late.csv"), index=False)
print("\nDeflection RMS, first 2 s vs last 4 s (mean over runs):")
for _, s in el.iterrows():
    print(f"  {s.condition:>9s} @ {s.station:5.2f}: early {s.D_early_mean:.3f} -> late {s.D_late_mean:.3f} mm"
          f"   (late/early = {s.late_over_early:.2f}, n={s.n})")

# ---- timing of the within-run change (model-free) ----
def timing(g):
    """
    From the repeat-averaged envelope m(t):
      start = mean over the first 1 s, low = minimum (3-point smoothed),
      t_drop50 = first time m falls halfway from start to low,
      t_rise50 = first time AFTER the minimum that m climbs back halfway
                 (NaN if it never does within the run = 'no recovery'),
      end = mean over the last 1 s.
    """
    m = g.groupby("t")["D_rms"].mean().sort_index()
    sm = m.rolling(3, center=True, min_periods=1).mean()
    t = m.index.values
    start = m[t < 1.0].mean()
    i_min = int(np.argmin(sm.values)); low = sm.values[i_min]
    end = m[t > t[-1] - 1.0].mean()
    half = low + 0.5 * (start - low)
    below = np.where(sm.values[:i_min + 1] <= half)[0]
    above = np.where(sm.values[i_min:] >= half)[0]
    return pd.Series(dict(start_mm=start, low_mm=low, end_mm=end, t_low_s=t[i_min],
                          t_drop50_s=t[below[0]] if len(below) else np.nan,
                          t_rise50_s=t[i_min + above[0]] if len(above) else np.nan,
                          start_over_low=start / low, end_over_low=end / low))

tim = tc.groupby(["condition", "station"]).apply(timing).reset_index()
tim["_k"] = tim.condition.map(condition_sort_key)
tim = tim.sort_values(["_k", "station"]).drop(columns="_k")
tim.to_csv(os.path.join(RES, "timing.csv"), index=False)
print("\nTiming of the within-run change (repeat-averaged deflection RMS):")
print("  t_drop50 = time to fall halfway to the low level; t_rise50 = time to climb back halfway")
for _, s in tim.iterrows():
    rise = f"{s.t_rise50_s:4.1f} s" if np.isfinite(s.t_rise50_s) else " none"
    print(f"  {s.condition:>9s} @ {s.station:5.2f}: start {s.start_mm:.2f} -> low {s.low_mm:.2f} mm "
          f"(x{s.start_over_low:.1f} drop) at {s.t_low_s:.1f} s | drop50 {s.t_drop50_s:.1f} s | "
          f"rise50 {rise} | end {s.end_mm:.2f} mm")
print("  (For Baseline, start/low/end differences are just turbulence noise - there is no real drop.)")

# ---- figure: averaged envelope vs time, one panel per station ----
palette = {c: col for c, col in zip(
    sorted(tc.condition.unique(), key=condition_sort_key),
    [C["blue"], C["orange"], C["aqua"], C["yellow"], "#e87ba4", "#008300", "#6250d6", "#e34948"])}
morph_st = sorted(tc[tc.condition.str.lower() != "baseline"].station.unique())
base = tc[tc.condition.str.lower() == "baseline"]
n_pan = len(morph_st) + (1 if len(base) else 0)
ncol = 3; nrow = int(np.ceil(n_pan / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.6 * nrow), sharex=True, squeeze=False)
axes = axes.ravel()
for a, st in zip(axes, morph_st):
    for c, g in tc[(tc.station == st) & (tc.condition.str.lower() != "baseline")].groupby("condition"):
        m = g.groupby("t")["D_rms"].agg(["mean", "std"])
        a.plot(m.index, m["mean"], color=palette[c], label=c)
        a.fill_between(m.index, m["mean"] - m["std"], m["mean"] + m["std"], color=palette[c], alpha=0.12, lw=0)
    a.set(title=f"Station {st:g}", ylabel="Deflection RMS (mm)")
if len(base):
    a = axes[len(morph_st)]
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, base.station.nunique()))
    for col, (st, g) in zip(blues, base.groupby("station")):
        m = g.groupby("t")["D_rms"].mean()
        a.plot(m.index, m.values, color=col, label=f"{st:g}")
    a.set(title="Baseline (each station)", ylabel="Deflection RMS (mm)")
    a.legend(fontsize=7, title="station", loc="upper right")
for a in axes[n_pan:]:
    a.set_visible(False)
for a in axes[max(0, n_pan - ncol):n_pan]:
    a.set_xlabel("Time in run (s)")
axes[0].legend(fontsize=8, loc="upper right")
fig.suptitle("Turbulence strength during each run (mean of repeats ± 1 sd)", fontweight="bold")
fig.tight_layout()
out = os.path.join(ROOT, "figures", "06_time_course.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150); print("\nsaved", out)
show()
