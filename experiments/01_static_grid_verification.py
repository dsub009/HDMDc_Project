"""
Experiment 1 - STATIC grid: can we recover the beam's fn and zeta from
noisy turbulence-driven voltage?  (Ground truth known from the simulator.)

Compares three methods over many independent 60 s records:
  A. HDMD on raw V                         (what the original code would do)
  B. Welch PSD peak + half-power damping   (classical, as in Lee et al. 2021)
  C. HDMD on the autocorrelation of V      (recommended)

Run from the project folder:  python experiments/01_static_grid_verification.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from plotstyle import plt, C, show

from hdmd.dmd import fit_hdmd, autocorrelation, pick_mode
from hdmd.metrics import psd_peak, half_power_damping
from simulations.piezo_beam import simulate_ptfv, true_beam_mode

DT, BAND, N_RUNS = 1e-3, (30, 80), 20
f_true, z_true = true_beam_mode()
print(f"Ground truth: f = {f_true:.3f} Hz, zeta = {z_true:.4f}")

est = {k: ([], []) for k in ["A  HDMD raw V", "B  half-power", "C  HDMD autocorr"]}
for seed in range(N_RUNS):
    _, _, V, _ = simulate_ptfv(duration=60, dt=DT, snr=47.3, seed=seed)
    r = fit_hdmd((V - V.mean())[None, :], delays=40, rank=6, dt=DT); i = pick_mode(r, BAND)
    est["A  HDMD raw V"][0].append(abs(r["freq_hz"][i])); est["A  HDMD raw V"][1].append(r["damping"][i])
    fp, f, P = psd_peak(V, DT, 4096, BAND); zh, _ = half_power_damping(f, P, BAND)
    est["B  half-power"][0].append(fp); est["B  half-power"][1].append(zh)
    r = fit_hdmd(autocorrelation(V, 300)[None, :], delays=40, rank=6, dt=DT, min_ratio=2)
    i = pick_mode(r, BAND)
    est["C  HDMD autocorr"][0].append(abs(r["freq_hz"][i])); est["C  HDMD autocorr"][1].append(r["damping"][i])

print(f"\n{'method':18s} {'f mean':>8s} {'f sd':>6s} {'zeta mean':>10s} {'zeta sd':>8s} {'zeta bias':>10s}")
for k, (F, Z) in est.items():
    F, Z = np.array(F), np.array(Z)
    print(f"{k:18s} {F.mean():8.2f} {F.std():6.2f} {Z.mean():10.4f} {Z.std():8.4f} {100*(Z.mean()/z_true-1):+9.0f}%")

# sensitivity of method C to delays and rank (single record)
_, _, V, _ = simulate_ptfv(duration=60, dt=DT, snr=47.3, seed=123)
R = autocorrelation(V, 300)
delays_list, ranks = [10, 20, 30, 40, 60, 80], [2, 4, 6, 8, 10]
Zerr = np.full((len(ranks), len(delays_list)), np.nan)
for a, rk in enumerate(ranks):
    for b, d in enumerate(delays_list):
        try:
            res = fit_hdmd(R[None, :], d, rk, DT, min_ratio=0)
            i = pick_mode(res, BAND); Zerr[a, b] = 100 * (res["damping"][i] / z_true - 1)
        except ValueError:
            pass

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fp, f, P = psd_peak(V, DT, 4096)
a = ax[0, 0]; a.semilogy(f, P, color=C["blue"], lw=1.5)
a.axvline(f_true, color=C["muted"], ls="--", lw=1); a.text(f_true + 3, P.max() / 3, f"true f = {f_true:.1f} Hz", color=C["muted"])
a.set(xlim=(0, 200), xlabel="Frequency (Hz)", ylabel="PSD of V (V²/Hz)", title="Simulated PTFV voltage spectrum (one 60 s record)")

res = fit_hdmd(R[None, :], 40, 6, DT, min_ratio=2)
a = ax[0, 1]; th = np.linspace(0, 2 * np.pi, 400)
a.plot(np.cos(th), np.sin(th), color=C["muted"], lw=1)
a.scatter(res["eigs"].real, res["eigs"].imag, s=60, color=C["blue"], edgecolor="#fcfcfb", lw=2, zorder=3, label="HDMD on autocorrelation")
lt = np.exp((-z_true * 2 * np.pi * f_true / np.sqrt(1 - z_true**2) + 2j * np.pi * f_true) * DT)
a.scatter([lt.real, lt.real], [lt.imag, -lt.imag], s=140, facecolor="none", edgecolor=C["orange"], lw=2, zorder=4, label="true beam eigenvalue")
a.set(aspect="equal", xlim=(-0.1, 1.1), ylim=(-0.6, 0.6), xlabel="Re λ", ylabel="Im λ", title="Identified discrete eigenvalues")
a.legend(loc="lower left")

a = ax[1, 0]
for j, (k, (F, Z)) in enumerate(est.items()):
    a.scatter(np.full(len(Z), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(Z)), Z,
              s=30, color=[C["orange"], C["yellow"], C["blue"]][j], edgecolor="#fcfcfb", lw=1)
a.axhline(z_true, color=C["ink"], ls="--", lw=1); a.text(2.35, z_true, "true ζ", va="center", color=C["ink"])
a.set_xticks(range(3), [k[3:] for k in est]); a.set(ylabel="Estimated damping ratio ζ", xlim=(-0.5, 2.7),
      title=f"Damping estimates, {N_RUNS} independent records")

a = ax[1, 1]
im = a.imshow(Zerr, cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
a.set_xticks(range(len(delays_list)), delays_list); a.set_yticks(range(len(ranks)), ranks)
for (yy, xx), v in np.ndenumerate(Zerr):
    a.text(xx, yy, "–" if np.isnan(v) else f"{v:+.0f}", ha="center", va="center", fontsize=9, color=C["ink"])
a.set(xlabel="delays", ylabel="rank", title="Method C: ζ error (%) vs delays & rank"); a.grid(False)
fig.colorbar(im, ax=a, label="ζ error (%)")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "01_static_verification.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150); print("\nsaved", out)
show()
