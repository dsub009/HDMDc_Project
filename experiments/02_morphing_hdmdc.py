"""
Experiment 2 - MORPHING grid: can HDMDc identify how the PTFV signal
responds to the grid blockage sigma(t)?

  A. HDMDc on raw V, input sigma        -> fails (morphing changes VARIANCE)
  B. HDMDc on RMS envelope of V          -> works, but needs phase-averaging
     over repeated identical actuation sequences.

Validation is always on a DIFFERENT actuation sequence and new turbulence.
Run from the project folder:  python experiments/02_morphing_hdmdc.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from plotstyle import plt, C, show

from hdmd.hdmdc import fit_hdmdc, predict_hdmdc
from hdmd.metrics import r_squared
from simulations.morphing import random_step_input, intensity_response, rms_envelope, SIGMA_MIN, SIGMA_MAX
from simulations.piezo_beam import simulate_ptfv

DT, PERIOD, DEC, DELAYS = 1e-3, 20.0, 20, 10


def run(seq_seed, n_rep, noise_seed):
    one = random_step_input(PERIOD, DT, seed=seq_seed)
    sig = np.tile(one, n_rep)
    scale = intensity_response(sig, DT)
    _, _, V, _ = simulate_ptfv(PERIOD * n_rep, DT, force_scale=scale, seed=noise_seed)
    return sig, scale, V


def averaged(seq_seed, n_rep, noise_seed):
    sig, scale, V = run(seq_seed, n_rep, noise_seed)
    env, dte = rms_envelope(V, DT, f_cut=5, decimate=DEC)
    L = int(round(PERIOD / dte))
    return env.reshape(n_rep, L).mean(0), sig[::DEC][:L], scale[::DEC][:L], dte, (sig, V)


# ---- A: raw voltage ----
sig, scale, V = run(11, 15, 100)
tr = int(0.7 * len(V))
X = (V - V[:tr].mean()) / V[:tr].std(); U = (sig - sig[:tr].mean()) / sig[:tr].std()
m = fit_hdmdc(X[None, :tr], U[None, :tr], delays=40, rank_in=12, rank_out=10, dt=DT)
Xp = predict_hdmdc(m, X[None, tr:tr + 40], U[None, tr:])
print(f"A. raw V + sigma : test R² = {r_squared(X[tr:], Xp[0]):.3f}   (≈0 -> HDMDc sees nothing)")

# ---- B0: choose delays/rank by cross-validation on a VALIDATION sequence ----
# (on real data the true response is unknown, so this is the only honest way)
ea, sa, _, dte, _ = averaged(11, 30, 100)          # training
ev, sv, cv, _, _ = averaged(33, 30, 300)          # validation (not the test set)
mu, sd, um, us = ea.mean(), ea.std(), sa.mean(), sa.std()
configs = [(5, 3, 2), (10, 4, 2), (10, 6, 3), (20, 4, 2), (20, 8, 4), (20, 12, 6)]
print("\nB0. model selection (validation R² vs averaged envelope):")
best, best_r2 = None, -np.inf
for (d, p_, r_) in configs:
    m = fit_hdmdc(((ea - mu) / sd)[None, :], ((sa - um) / us)[None, :], d, p_, r_, dte)
    pv = predict_hdmdc(m, ((ev - mu) / sd)[None, :d], ((sv - um) / us)[None, :])[0] * sd + mu
    r2 = r_squared(ev[20:], pv[20:])
    print(f"    delays={d:2d} p={p_:2d} r={r_}: R² = {r2:.3f}")
    if r2 > best_r2:
        best, best_r2 = (d, p_, r_), r2
DELAYS, P_IN, R_OUT = best
print(f"    -> selected delays={DELAYS}, p={P_IN}, r={R_OUT}\n")

# ---- B: envelope + phase averaging ----
reps = [1, 3, 5, 10, 15, 30]
r2_env, r2_true, keep = [], [], {}
for n_rep in reps:
    ea, sa, ca, dte, _ = averaged(11, n_rep, 100)        # training sequence
    eb, sb, cb, _, raw = averaged(22, n_rep, 200)        # unseen test sequence
    mu, sd, um, us = ea.mean(), ea.std(), sa.mean(), sa.std()
    m = fit_hdmdc(((ea - mu) / sd)[None, :], ((sa - um) / us)[None, :], DELAYS, P_IN, R_OUT, dte)
    pred = predict_hdmdc(m, ((eb - mu) / sd)[None, :DELAYS], ((sb - um) / us)[None, :])[0] * sd + mu
    r2_env.append(r_squared(eb[DELAYS:], pred[DELAYS:]))
    # true intensity in envelope units (best linear map, test window only)
    k = np.polyfit(cb, eb, 1); true_env = np.polyval(k, cb)
    r2_true.append(r_squared(true_env[DELAYS:], pred[DELAYS:]))
    keep[n_rep] = (m, mu, sd, um, us, eb, sb, true_env, pred, dte, raw)
    print(f"B. envelope, {n_rep:2d} repeats: test R² vs averaged envelope = {r2_env[-1]:.3f}, "
          f"vs true intensity = {r2_true[-1]:.3f}")

# ---- identified step response (what a controller designer needs) ----
m, mu, sd, um, us, eb, sb, true_env, pred, dte, raw = keep[30]
n_pre, n_post = 150, 60                     # settle at sigma_min, then step
u_step = np.r_[np.full(n_pre, SIGMA_MIN), np.full(n_post, SIGMA_MAX)]
y = predict_hdmdc(m, np.zeros((1, DELAYS)), ((u_step - um) / us)[None, :])[0]
y = y[n_pre - DELAYS:]
y = (y - y[DELAYS - 1]) / (y[-1] - y[DELAYS - 1])
n = len(y); t_step = np.arange(n) * dte
true_step = intensity_response(np.r_[np.full(DELAYS * DEC, SIGMA_MIN), np.full((n - DELAYS) * DEC, SIGMA_MAX)], DT)[::DEC]
true_step = (true_step - true_step[0]) / (true_step[-1] - true_step[0])
t_rel = t_step - DELAYS * dte

def t63(t, y):
    return t[np.argmax(y >= 0.632)]
print(f"\nStep response (sigma {SIGMA_MIN} -> {SIGMA_MAX}): identified overshoot "
      f"{100*(y.max()-1):.0f} % (true 0 %), t63 identified {t63(t_rel, y)*1000:.0f} ms vs true "
      f"{t63(t_rel, true_step)*1000:.0f} ms.\n"
      "  NOTE: the step-response SHAPE is not reliable with this much data - it changes a lot\n"
      "  with delays/rank (equation-error bias from the noisy envelope). Prediction works;\n"
      "  extracting a time constant needs repeated step tests or OKID/ERA (see README).")

# ---------------- figure ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
sig_raw, V_raw = raw
tt = np.arange(len(V_raw)) * DT
a = ax[0, 0]; s = slice(0, int(PERIOD / DT))
a.plot(tt[s], V_raw[s], color=C["blue"], lw=0.4, label="voltage V")
a.set(xlabel="Time (s)", ylabel="V (V)", title="One 20 s cycle: grid steps barely visible in raw V")
a.legend(loc="upper right")
ins = a.inset_axes([0.0, -0.0, 1, 0.22]); ins.plot(tt[s], sig_raw[s], color=C["orange"], lw=1.5)
ins.set_ylabel("σ", color=C["orange"]); ins.set_xticks([]); ins.patch.set_alpha(0); ins.grid(False)
ins.set_ylim(SIGMA_MIN - 0.25, SIGMA_MAX + 0.01); ins.set_yticks([])

te = np.arange(len(eb)) * dte
a = ax[0, 1]
a.plot(te, eb, color=C["muted"], lw=1, label="measured envelope (30-repeat average)")
a.plot(te, true_env, color=C["ink"], lw=1.5, ls="--", label="true intensity response")
a.plot(te[DELAYS:], pred[DELAYS:], color=C["blue"], lw=2, label="HDMDc open-loop prediction")
a.set(xlabel="Time (s)", ylabel="RMS envelope of V (V)", title="Unseen test sequence (30 repeats)")
a.legend(loc="upper left", fontsize=8)

a = ax[1, 0]
a.plot(reps, r2_true, "o-", color=C["blue"], ms=8, mec="#fcfcfb", mew=2, label="vs true intensity")
a.plot(reps, r2_env, "o-", color=C["orange"], ms=8, mec="#fcfcfb", mew=2, label="vs measured (noisy) envelope")
a.set(xscale="log", xticks=reps, xticklabels=reps, xlabel="Repeats of the actuation sequence",
      ylabel="Test R²", title="How many repeats does the experiment need?")
a.legend(loc="lower right")

a = ax[1, 1]
a.plot(t_rel * 1000, true_step, color=C["ink"], ls="--", lw=1.5, label="true")
a.plot(t_rel * 1000, y, color=C["blue"], label="identified by HDMDc")
a.set(xlim=(-100, 1200), xlabel="Time after grid step (ms)", ylabel="Normalised response",
      title="Step response σ min → max: NOT yet reliable")
a.legend(loc="lower right")
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "02_morphing_hdmdc.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150); print("saved", out)
show()
