# Hankel DMD for PTFV — verification toolkit
Author: Dhakshini Subramanian
System identification for the morphing-grid PTFV project (Kueh Tat Yin, Monash).
Everything is checked against synthetic data **with a known answer** before being
used on real wind-tunnel data.

```
hdmd/            the methods
  hankel.py        Hankel (time-delay) embedding
  dmd.py           HDMD (no input) + autocorrelation + mode picking
  hdmdc.py         HDMDc (with input, Proctor et al. 2016) + open-loop prediction
  metrics.py       RMSE / R² / Welch peak / half-power damping
  io.py            load the Excel files, camera<->DAQ alignment check
  batch.py         scan a whole folder tree of runs, quality checks per run
data/raw/        experiment Excel files (put new ones here)
simulations/
  piezo_beam.py    SDOF piezo beam, Lee et al. 2021 Eq. 3 + Table 3 parameters
  morphing.py      random-step grid blockage σ(t) -> turbulence intensity -> beam
experiments/
  01_static_grid_verification.py   recover fn, ζ from noisy V (3 methods compared)
  02_morphing_hdmdc.py             identify V-response to grid morphing
  03_real_data_first_look.py       real data: f, ζ, sync check, intensity over time
  04_baseline_repeatability.py     5 repeats of one condition -> uncertainty
  05_all_conditions.py             every run in data/ -> results/*.csv + summary plot
tests/test_core.py                  known-answer tests (incl. a cross-check with PyDMD)
```

```bash
pip install numpy scipy matplotlib pydmd pytest
python main.py          # real-data analysis, opens the figure
python main.py 1        # static-grid verification
python main.py 2        # morphing verification (~2 min)
python main.py all      # tests + all experiments
```

See **PLAN.md** for status, next steps and the data to request.

## Rules learned from the verification

1. **Never run HDMD on raw turbulence-driven V to get damping.** It under-estimates ζ by about 70%.
   Run it on the **autocorrelation** of V (`autocorrelation()`), which gives about 1% bias.
2. **Forward-backward DMD (`method="fb"`) is only for free-decay data** (flick test).
   On stationary turbulent data it returns ζ = 0.
3. **The control input must be persistently exciting**: random steps or PRBS, not one sine.
4. **A morphing grid changes the variance of V, not its mean.** HDMDc on raw V with σ as the input finds nothing.
   Use the short-time RMS envelope, and **repeat the same actuation sequence 10–30 times and phase-average**.
5. **`rank_out` must be at least the system order plus the input feed-through**, and is usually smaller than `rank_in`.
6. Choose delays and rank by **validation on a separate record**, never on the training data.
7. R² against noisy data understates model quality. When possible, compare against
   the averaged signal or a known physical quantity (fn, ζ from the flick test).

## Open problem

Getting a reliable **time constant** of the turbulence response to morphing.
HDMDc predicts the envelope well, but the step-response shape it implies changes
a lot with rank. Options: repeated σ_min→σ_max step tests averaged directly
(simplest), longer sequences with more steps, or OKID/ERA.

## References to read
- Schmid 2010, J. Fluid Mech. — DMD (provided)
- Tu et al. 2014, J. Comput. Dyn. — exact DMD
- Proctor, Brunton & Kutz 2016, SIAM J. Appl. Dyn. Syst. — DMD with control
- Arbabi & Mezić 2017, SIAM J. Appl. Dyn. Syst. — Hankel DMD
- Dawson et al. 2016, Exp. Fluids — noise bias in DMD, forward-backward DMD
- Gavish & Donoho 2014, IEEE Trans. Inf. Theory — optimal SVD threshold
- James, Carne & Lauffer 1993 (NExT); Juang & Pappa 1985 (ERA) — output-only modal ID

## Using the full dataset

Copy the data folders into `data/` unchanged, for example
`data/Global_Raw_Deflection_Overlap/Station_11.6mm/0915_3.5mm_11.6/0915_3.5mm_11.6_Run_1_RawData.xlsx`.
Only the file names matter (`<date>_<condition>_<station>_Run_<n>_RawData.xlsx`); PNGs are ignored.
Then run `python main.py 5`. The outputs are:
- `results/per_run.csv`: one row per run, with quality flags and f, ζ, RMS
- `results/by_condition.csv`: mean, sd and number of runs for each (condition, station)
- `figures/05_all_conditions.png`

Interpretation of the names (not confirmed by the student):
- *Station*: distance from the grid. 15.1–25.5 matches the x* range of 150–255 mm on the FYP slide, so it is most likely cm.
- *Condition*: `Baseline` or the grid setting (`2mm`, `3mm`, ...), probably the morphing-flap displacement.
