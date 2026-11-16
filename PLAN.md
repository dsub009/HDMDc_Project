# HDMD work for the morphing-grid PTFV project — plan & status

## Goal of my part
Use Hankel DMD (HDMD / HDMDc) to (1) identify the beam's dynamics (frequency,
damping) from PTFV signals, and (2) model how the PTFV signal responds to grid
morphing, for "on-demand turbulence tuning". Every method is verified on
synthetic data with a known answer before being trusted on real data.

## Done
| # | Item | Where | Result |
|---|------|-------|--------|
| 1 | Review of my first code | — | Found a crash, a non-identifiable sine input, wrong projection basis, log(nan) bug, and no train/test validation |
| 2 | Rebuilt the core package | `hdmd/` | 8 known-answer tests pass, including agreement with PyDMD |
| 3 | Physics-based simulator | `simulations/piezo_beam.py` | Lee 2021 Eq. 3 and Table 3 parameters |
| 4 | Static verification | `experiments/01_...` | HDMD on raw V gives −72% ζ bias; half-power gives −20%; **HDMD on the autocorrelation gives +1%** |
| 5 | Morphing verification | `experiments/02_...` | Raw V + HDMDc fails (R² ≈ 0); RMS envelope + 10–30 averaged repeats works for prediction; step response not yet reliable |
| 6 | First real data (2 × ~10 s) | `experiments/03_...` | Results below |
| 8 | Full dataset, 152 runs (experiments 5–7) | `experiments/05–07` | See "Full-dataset findings" below |
| 7 | 5 baseline repeats (0824_Baseline_25.5, Runs 1–5) | `experiments/04_...` | f = 49.1 ± 0.6 Hz, ζ = 0.057 ± 0.007 (deflection); HDMD is ~2–4× more repeatable than half-power |

### Full-dataset findings (152 runs; 144 usable)
- **Morphing modulates turbulence over time**: in every morphing run the deflection RMS starts high,
  drops 3–5× within ~2–3 s, then rises again later. It is repeatable across 6 repeats and the same timing
  appears at all stations. Baseline runs stay flat, which makes them the control.
- **Settings differ mainly in when the turbulence recovers**: 4.5 mm earliest (~6–7 s), then 3.5 mm (~8–9 s),
  then 3 mm (end of run), and 2 mm shows no clear recovery within 10 s. Start and low levels are similar for all settings.
  Whole-run RMS differences between settings (15–40 %, p < 0.01) mostly reflect this timing.
- Damping does not depend on the grid setting (all p > 0.4).
- Decay with distance is RMS ∝ x^−1.25 for all morphing settings (Baseline −1.6, overlapping CI).
- Baseline is ~3× stronger than the morphing settings, but it was recorded in a different session.
- The frequency trend with amplitude is mostly a distance effect (weak within a station).
- Method: damping scatter is 16 % (HDMD, deflection) vs 44 % (half-power) over 26 groups.
- HDMDc on real data is not possible: the actuator position was not logged. The response is described directly
  (experiment 6 timing); HDMDc is verified on simulation, and its requirements are documented.
- One file (0825_Baseline_23.4_Run_1) has no voltage sheet and was excluded.

### Baseline repeatability (5 runs, same condition)
- "Extracted_Data_No_Morphing" is the same data as **0824 Baseline Run 3**.
- Deflection is in mm; the sensor is on the centreline (Tat Yin).
- Frequency: 48.9 ± 1.2 Hz (from V), 49.1 ± 0.6 Hz (from deflection).
- Damping, autocorr-HDMD: 0.058 ± 0.013 (V), 0.057 ± 0.007 (deflection). Half-power: 0.032 ± 0.015.
  Run-to-run scatter is 22 % / 13 % for HDMD against 47 % for half-power. This matches the simulation.
- The scatter is larger than in the 60 s simulations (~8 %) because these records are only 10 s long.
- **Run 2's camera time column is wrong**: it was stretched to 10.00 s (1084.6 fps). Using ~1035.6 fps like
  the other runs restores |corr(V, deflection)| = 0.98. Corrected in the code; still to confirm with Tat Yin.

### Real-data findings (Extracted_Data_*.xlsx)
- f ≈ 48.8–49.7 Hz (flick test in Lee 2021: 50.78 Hz). V and deflection give the same answer.
- ζ ≈ 0.061 (no morphing), ≈ 0.041 (morphing, whole record). Both inside Lee's range of 0.012–0.072.
- V vs deflection: |corr| 0.97–0.99 once aligned, but **the best camera↔DAQ time shift wanders from 0 to −2.6 ms**
  within one record. That is up to ~46° of phase at 49 Hz. Sync must be fixed before any V→δ model.
- **150 Hz line, 41 dB above the floor, in the morphing voltage only**: electrical interference, not vibration (it is absent in deflection).
  It is likely from the motor/driver supply; 150 Hz is the 3rd harmonic of 50 Hz mains.
- Morphing record: V RMS steps from ~0.9 V to ~2.3 V at ~7.2 s (one grid move?).
  ζ appears to drop from ~0.058 to ~0.030 after the step, but the "after" segment is only 2.1 s, so this is not yet a result.

## Still to do
1. **Data sync**: correct the camera time axis using the LED sync / real frame timestamps, then re-check that the shift is flat.
2. **Noise baselines**: quantify the 150 Hz (and possible 50 Hz) interference using wind-off recordings, then notch-filter or fix the grounding.
3. **Static grid study**: for each fixed blockage σ (several positions between 0.335 and 0.402) and streamwise position x, get f, ζ and RMS from 60 s records with repeats. Plot them against σ and x* (this links to Trend 1 and Trend 2 on the theory slide).
4. **V → δ model**: once synced, compare a static linear fit and HDMDc (V input, δ output, with delays) against the TNN in Lee 2025 (RMSE 0.15 mm).
5. **Morphing study**: random-step σ(t) with the logged actuator position, 10–30 repeats. Fit HDMDc on the RMS envelope, validated on a different sequence.
6. **Response time to morphing**: averaged repeated σ_min→σ_max step tests (model-free), compared with HDMDc. Open problem in the simulation.
7. Write-up: the methods chapter (verification first), then results with uncertainty (repeat-to-repeat spread).

## Still unanswered by the student
- Distance from grid to sensor, and flow speed. What does "25.5" in the file names mean?
- What the grid did in the morphing run, and the data files behind the "0924_2mm_25.5" plots (morphing runs?).
- How the camera time for Run 2 was generated.

## What to request from the student
**Metadata for the two files already sent**
- Units of `Deflection_tip` (mm?). It looks mean-subtracted; was it filtered or scaled?
- Grid position or blockage during each run. For "Morphing": what moved, when, and from where to where?
- Sensor position (x/Dh or mm from the grid, lateral position), inlet velocity, date.
- How camera and DAQ were synchronised (LED?), and the camera's true frame rate (1035.5 fps here, not 1000).

**Raw data**
- The full raw recordings, not extracts: the 60 s DAQ files plus the camera frame timestamps.
- An actuator/Arduino log of commanded position with timestamps (or a limit-switch / LED event marker on a spare DAQ channel).

**New recordings (cheap, high value)**
- Wind OFF, motor OFF: the electrical noise floor.
- Wind OFF, motor ON (holding and moving): shows whether the 150 Hz / 50 Hz lines come from the driver.
- A flick test of the film, for the reference f and ζ.
- Static grid at about 3–5 fixed σ positions, 60 s each, 3 repeats.
- Morphing: random-step sequence (hold 0.5–3 s between σ_min and σ_max), the same sequence repeated 10–30 times,
  plus one different sequence for testing. Simple σ_min↔σ_max step tests repeated ~20 times.
