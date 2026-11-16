"""
Hankel DMD with control (Proctor, Brunton & Kutz 2016, SIAM J. Appl. Dyn. Syst.)

    z_{k+1} = A z_k + B v_k
where z_k is the Hankel (time-delay) state and v_k the Hankel input history.

IMPORTANT: this model assumes the input changes the state ADDITIVELY.
A morphing grid mostly changes turbulence INTENSITY (the variance of the
beam signal), not the mean of V. Feeding raw V and grid position into
HDMDc therefore identifies almost nothing. Use an observable that responds
linearly to the input, e.g. the short-time RMS envelope of V
(see simulations/morphing.py and experiments/03_morphing_control.py).
"""
import numpy as np

from hdmd.hankel import create_hankel, check_hankel_size
from hdmd.dmd import continuous_eigs, svht_rank


def fit_hdmdc(X, U, delays, rank_in=None, rank_out=None, dt=1.0):
    """
    Parameters
    ----------
    X : (n_states, n) measured outputs (mean-subtracted, scaled).
    U : (n_inputs, n) control input (mean-subtracted). Must be persistently
        exciting: random steps / PRBS / multisine, NOT a single sinusoid.
        A single sine is predictable from its own past, so A and B cannot
        be separated.
    delays : number of time delays (applied to both X and U).
    rank_in : truncation p of the SVD of Omega = [X; U]  (Proctor's p).
    rank_out : truncation r of the SVD of X' (Proctor's r). Usually r < p,
        because Omega also contains the input directions.

    Returns
    -------
    dict with A_tilde (r x r), B_tilde (r x n_in*delays), basis U_hat,
    eigs/freq_hz/growth/damping, delays, n_states.
    """
    X = np.atleast_2d(X); U = np.atleast_2d(U)
    Xh, Uh = create_hankel(X, delays), create_hankel(U, delays)
    check_hankel_size(X.shape[1], delays, Xh.shape[0] + Uh.shape[0])

    Xc, Yc, Uc = Xh[:, :-1], Xh[:, 1:], Uh[:, :-1]
    Omega = np.vstack([Xc, Uc])

    Ut, St, Vt = np.linalg.svd(Omega, full_matrices=False)
    p = min(rank_in or svht_rank(St, Omega.shape), len(St))
    Ut, St, Vt = Ut[:, :p], St[:p], Vt[:p].T

    Uh_, Sh_, _ = np.linalg.svd(Yc, full_matrices=False)
    r = min(rank_out or svht_rank(Sh_, Yc.shape), p, len(Sh_))
    Uhat = Uh_[:, :r]

    n_x = Xc.shape[0]
    U1, U2 = Ut[:n_x], Ut[n_x:]
    G = Yc @ Vt / St                      # X' V Sigma^-1
    A_tilde = Uhat.T @ G @ U1.T @ Uhat
    B_tilde = Uhat.T @ G @ U2.T

    eigs = np.linalg.eigvals(A_tilde)
    omega, f, g, z = continuous_eigs(eigs, dt)
    return dict(A_tilde=A_tilde, B_tilde=B_tilde, U_hat=Uhat, eigs=eigs,
                freq_hz=f, growth=g, damping=z, delays=delays,
                n_states=X.shape[0], rank_in=p, rank_out=r,
                input_singular_values=np.linalg.svd(Uh, compute_uv=False))


def predict_hdmdc(model, X_init, U):
    """
    Open-loop simulation on NEW data (e.g. the test split).

    X_init : (n_states, delays) the first `delays` measured samples.
    U : (n_inputs, n) the input over the whole prediction window,
        aligned so U[:, 0] is at the same time as X_init[:, 0].
    Returns X_pred (n_states, n); the first `delays` columns equal X_init.
    """
    d, ns = model["delays"], model["n_states"]
    U = np.atleast_2d(U)
    n = U.shape[1]
    Uh = create_hankel(U, d)                    # column j covers u_j..u_{j+d-1}
    z = model["U_hat"].T @ X_init.reshape(-1, order="F")
    X_pred = np.zeros((ns, n))
    X_pred[:, :d] = X_init
    for j in range(n - d):                      # predict sample j+d
        z = model["A_tilde"] @ z + model["B_tilde"] @ Uh[:, j]
        X_pred[:, j + d] = (model["U_hat"] @ z)[-ns:]   # newest block
    return X_pred
