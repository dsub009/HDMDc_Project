"""
Experiment 7 - STATISTICS on the batch results (results/per_run.csv from
experiment 5; results/early_late.csv from experiment 6 if present).

Answers, with p-values you can quote:
  1. Does the grid setting (2/3/3.5/4.5mm) change turbulence strength,
     damping, frequency, or the within-run change? (per station, ANOVA +
     Kruskal-Wallis, Holm-corrected for testing 5 stations)
  2. Baseline vs morphing settings at the same station.
  3. How fast does turbulence decay with distance (power law)?
  4. Is frequency linked to deflection amplitude?
  5. Is HDMD damping more repeatable than half-power?  V vs deflection.
Writes results/statistics_report.txt.
"""
import os, sys, io
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join(ROOT, "results")
buf = io.StringIO()
def say(s=""):
    print(s); buf.write(s + "\n")

def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1, run)
    return adj

df = pd.read_csv(os.path.join(RES, "per_run.csv"))
df["flags"] = df["flags"].fillna("")
ok = df[~df["flags"].str.contains("FAILED|V/D")].copy()
is_base = ok.condition.str.lower() == "baseline"
morph, base = ok[~is_base], ok[is_base]
say(f"Runs used: {len(ok)} of {len(df)} (excluded: failed or voltage/deflection disagree)\n")

def per_station_test(data, var, label):
    say(f"[{label}] does the grid setting matter?  (per station; p after Holm correction)")
    rows = []
    for st, g in data.groupby("station"):
        groups = [gg[var].dropna().values for _, gg in g.groupby("condition")]
        groups = [x for x in groups if len(x) >= 2]
        if len(groups) < 2:
            continue
        rows.append((st, stats.f_oneway(*groups).pvalue, stats.kruskal(*groups).pvalue,
                     g.groupby("condition")[var].mean()))
    pa, pk = holm([r[1] for r in rows]), holm([r[2] for r in rows])
    for (st, _, _, means), a, k in zip(rows, pa, pk):
        verdict = "DIFFERENT" if max(a, k) < 0.05 else ("borderline" if min(a, k) < 0.05 else "no evidence")
        say(f"  station {st:5g}: ANOVA p={a:.2g}, Kruskal p={k:.2g} -> {verdict:11s} "
            f"| " + ", ".join(f"{c}={v:.3g}" for c, v in means.items()))
    say()

say("=" * 78 + "\n1. EFFECT OF GRID SETTING (morphing conditions only)\n" + "=" * 78)
per_station_test(morph, "D_rms_mm", "deflection RMS, whole run (mm)")
per_station_test(morph, "zeta_D", "damping ratio (HDMD, deflection)")
per_station_test(morph, "f_D", "frequency (HDMD, deflection, Hz)")
per_station_test(morph, "rms_ratio_2nd_vs_1st_half", "within-run change (2nd half / 1st half)")

el_path = os.path.join(RES, "early_late.csv")
if os.path.exists(el_path):
    say("Same question using the LATE (settled, last 4 s) deflection RMS - see experiment 6:")
    tc = pd.read_csv(os.path.join(RES, "time_course.csv"))
    late = tc.groupby(["file", "condition", "station"]).apply(
        lambda g: g.loc[g.t > g.t.max() - 4, "D_rms"].mean()).rename("D_late").reset_index()
    late = late[late.file.isin(ok.file)]
    per_station_test(late[late.condition.str.lower() != "baseline"], "D_late", "deflection RMS, last 4 s (mm)")

say("=" * 78 + "\n2. BASELINE vs MORPHING SETTINGS AT (NEARLY) THE SAME STATION\n" + "=" * 78)
say("NOTE: Baseline was recorded in a different session (August) -> setup differences possible.")
for sb in sorted(base.station.unique()):
    near = morph[np.abs(morph.station - sb) < 0.2]
    if near.empty:
        continue
    b = base[base.station == sb].D_rms_mm
    for c, g in near.groupby("condition"):
        p = stats.ttest_ind(b, g.D_rms_mm, equal_var=False).pvalue
        say(f"  Baseline@{sb:g} ({b.mean():.3f} mm) vs {c}@{g.station.iloc[0]:g} ({g.D_rms_mm.mean():.3f} mm): "
            f"ratio {b.mean() / g.D_rms_mm.mean():.2f}, Welch p={p:.1g}")
say()

say("=" * 78 + "\n3. DECAY WITH DISTANCE: deflection RMS ~ station^b\n" + "=" * 78)
rng = np.random.default_rng(0)
for c, g in ok.groupby("condition"):
    b = np.polyfit(np.log(g.station), np.log(g.D_rms_mm), 1)[0]
    bs = [np.polyfit(np.log(s.station), np.log(s.D_rms_mm), 1)[0]
          for s in (g.sample(len(g), replace=True, random_state=rng.integers(1e9)) for _ in range(2000))]
    say(f"  {c:9s}: b = {b:.2f}   95% CI [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]")
say()

say("=" * 78 + "\n4. FREQUENCY vs DEFLECTION AMPLITUDE\n" + "=" * 78)
r, p = stats.spearmanr(ok.D_rms_mm, ok.f_D)
say(f"  all runs: Spearman r = {r:.2f} (p={p:.1g}) - larger deflection, lower frequency")
for st, g in morph.groupby("station"):
    r, p = stats.spearmanr(g.D_rms_mm, g.f_D)
    say(f"  within station {st:5g} only: r = {r:+.2f} (p={p:.2g})")
say("  -> if the within-station correlations are weak, the overall link is mostly a\n"
    "     distance effect (both change with distance) rather than amplitude itself.\n")

say("=" * 78 + "\n5. METHOD CHECKS\n" + "=" * 78)
cell = ok.groupby(["condition", "station"])
cv = lambda col: np.nanmedian(cell[col].std() / cell[col].mean()) * 100
say(f"  run-to-run scatter of damping (median CV over {cell.ngroups} condition/station cells):")
say(f"    HDMD on deflection {cv('zeta_D'):.0f} %,  HDMD on voltage {cv('zeta_V'):.0f} %,  half-power {cv('zeta_halfpower_V'):.0f} %")
say(f"  voltage vs deflection: frequency differs by {np.mean(np.abs(ok.f_V - ok.f_D)):.2f} Hz on average "
    f"(r = {stats.pearsonr(ok.f_V, ok.f_D)[0]:.2f}); damping r = {stats.pearsonr(ok.zeta_V, ok.zeta_D)[0]:.2f}")

with open(os.path.join(RES, "statistics_report.txt"), "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("\nsaved results/statistics_report.txt")
