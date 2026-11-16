import numpy as np


def create_hankel(X, delays):
    """
    Stack `delays` time-shifted copies of X into a Hankel matrix.

    X has shape (n_channels, n_samples). Column j of the result is
        [x_j ; x_{j+1} ; ... ; x_{j+delays-1}]
    so the NEWEST sample of each column is in the LAST block of rows.
    Result shape: (n_channels * delays, n_samples - delays + 1).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_ch, n = X.shape
    if delays < 1:
        raise ValueError("delays must be >= 1")
    if delays > n:
        raise ValueError("delays cannot exceed the number of samples")
    m = n - delays + 1
    H = np.empty((n_ch * delays, m))
    for i in range(delays):
        H[i * n_ch:(i + 1) * n_ch, :] = X[:, i:i + m]
    return H


def check_hankel_size(n_samples, delays, n_rows, min_ratio=10):
    """
    Warn if the Hankel matrix has too few columns compared with rows.

    n_rows is the total number of regression rows (state rows + input rows).
    Rule of thumb: columns should be at least ~10x the rows, otherwise the
    least-squares fit is under-determined and will fit noise.
    """
    n_cols = n_samples - delays
    if n_cols < min_ratio * n_rows:
        print(f"WARNING: Hankel has {n_cols} columns for {n_rows} rows "
              f"(< {min_ratio}x). Use fewer delays or more data.")
    return n_cols
