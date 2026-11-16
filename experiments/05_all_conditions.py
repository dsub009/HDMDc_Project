"""
Experiment 5 - ALL recorded conditions in one go.

1. Copy Tat Yin's data folders into data/ exactly as they are, e.g.
       data/Global_Raw_Deflection_Overlap/Station_11.6mm/0915_3.5mm_11.6/...
       data/02_Extracted_Raw_Data/Station_15.10cm/...
   (sub-folder names don't matter; file names do).
2. Run:  python main.py 5      (or python experiments/05_all_conditions.py)

Outputs (in results/):
  per_run.csv            one row per run: quality checks + f, zeta, RMS
  by_condition.csv       mean / sd / n over runs, per (condition, station)
  duplicates.csv         files found twice (skipped)
and figures/05_all_conditions.png.

Runs already analysed are cached in per_run.csv, so re-running only does
new files. Delete results/per_run.csv to redo everything.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
from plotstyle import plt, C, show

from hdmd.batch import find_runs, analyse_file, condition_sort_key

DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)
CACHE = os.path.join(RES, "per_run.csv")

runs, dupes = find_runs(DATA)
print(f"Found {len(runs)} run files under data/ ({len(dupes)} duplicates skipped).")
if not runs:
    sys.exit("No *_Run_<n>_RawData.xlsx files found - check the data/ folder.")
pd.DataFrame(dupes, columns=["skipped", "kept"]).to_csv(os.path.join(RES, "duplicates.csv"), index=False)

cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=["file", "flags"])
cache = cache[~cache["flags"].fillna("").str.startswith("FAILED")]   # retry failed files
done = set(cache["file"])
rows = []
t0 = time.time()
todo = [r for r in runs if r["file"] not in done]
for k, r in enumerate(todo, 1):
    try:
        res = analyse_file(r["path"])
    except Exception as e:                                # keep going, record the problem
        res = {"flags": f"FAILED: {e}"}
    row = {key: r[key] for key in ["file", "date", "condition", "station", "run"]}
    row.update(res)
    rows.append(row)
    print(f"  [{k}/{len(todo)}] {r['file']}  {res.get('flags', '')}")
    if k % 10 == 0 or k == len(todo):                    # save progress regularly
        pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).to_csv(CACHE, index=False)
if todo:
    print(f"Analysed {len(todo)} new files in {time.time() - t0:.0f} s.")
df = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True) if rows else cache
df = df[df["file"].isin({r["file"] for r in runs})]       # only files still present
df.to_csv(CACHE, index=False)

# ---------- summary per (condition, station) ----------
ok = df[~df["flags"].fillna("").str.contains("FAILED|V/D disagree")].copy()
cols = ["f_D", "zeta_D", "f_V", "zeta_V", "zeta_halfpower_V", "D_rms_mm", "V_rms"]
g = ok.groupby(["condition", "station"])[cols]
summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(g.size().rename("n_runs"))
summary = summary.reset_index()
summary["_k"] = summary["condition"].map(condition_sort_key)
summary = summary.sort_values(["_k", "station"]).drop(columns="_k")
summary.to_csv(os.path.join(RES, "by_condition.csv"), index=False)

# Same table without runs whose intensity changes mid-run (robustness check)
st = ok[~ok["flags"].fillna("").str.contains("mid-run")]
g2 = st.groupby(["condition", "station"])[cols]
summary_st = g2.mean().add_suffix("_mean").join(g2.std().add_suffix("_sd")).join(g2.size().rename("n_runs")).reset_index()
summary_st.to_csv(os.path.join(RES, "by_condition_steady_runs_only.csv"), index=False)

pd.set_option("display.width", 200)
print("\nRESULTS per condition and station (deflection-based HDMD; mean ± sd over runs):")
for _, s in summary.iterrows():
    print(f"  {s['condition']:>10s} @ {s['station']:6.2f}: n={int(s['n_runs'])}  "
          f"f = {s['f_D_mean']:.2f} ± {s['f_D_sd']:.2f} Hz   "
          f"zeta = {s['zeta_D_mean']:.4f} ± {s['zeta_D_sd']:.4f}   "
          f"defl rms = {s['D_rms_mm_mean']:.3f} ± {s['D_rms_mm_sd']:.3f} mm")
flagged = df[df["flags"].fillna("") != ""]
print(f"\n{len(flagged)} of {len(df)} runs have quality flags (see 'flags' column in results/per_run.csv):")
for fl, n in flagged["flags"].value_counts().items():
    print(f"  {n:3d} x  {fl}")

# ---------- figure: each measure vs station, one line per condition ----------
palette = [C["blue"], C["orange"], C["aqua"], C["yellow"], "#e87ba4", "#008300", "#6250d6", "#e34948"]
conds = sorted(summary["condition"].unique(), key=condition_sort_key)
if len(conds) > len(palette):
    print("More than 8 conditions - only the first 8 are plotted.")
    conds = conds[:8]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))
panels = [("D_rms_mm", "Deflection RMS (mm)", "Turbulence forcing (film deflection RMS)"),
          ("f_D", "Frequency (Hz)", "Natural frequency"),
          ("zeta_D", "Damping ratio ζ", "Damping (autocorr-HDMD)")]
for a, (col, ylab, title) in zip(ax, panels):
    for j, cnd in enumerate(conds):
        s = summary[summary["condition"] == cnd]
        a.errorbar(s["station"], s[col + "_mean"], yerr=s[col + "_sd"], fmt="o-", ms=7,
                   color=palette[j], mec="#fcfcfb", mew=1.5, capsize=3, label=cnd)
    a.set(xlabel="Station (distance from grid)", ylabel=ylab, title=title)
ax[0].legend(title="condition", fontsize=8, loc="best")
fig.tight_layout()
out = os.path.join(ROOT, "figures", "05_all_conditions.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("\nsaved", out, "\nsaved results/per_run.csv, results/by_condition.csv")
show()
