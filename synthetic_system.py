import numpy as np


def generate_synthetic_data():
    """
    Generate synthetic controlled measurements using sine and cosine signals.

    Returns
    -------
    t : numpy.ndarray
        Time values.

    X : numpy.ndarray
        Measurement data with shape (2, number_of_samples).

    U : numpy.ndarray
        Control input with shape (1, number_of_samples).
    """

    # ---------------------------------------------------------
    # 1. Time settings
    # ---------------------------------------------------------

    dt = 0.01
    t = np.arange(0, 20, dt)

    # ---------------------------------------------------------
    # 2. Synthetic control input
    # ---------------------------------------------------------

    # Think of this as the morphing/actuator input.
    u = np.sin(0.8 * t)

    # ---------------------------------------------------------
    # 3. Create dynamic states
    # ---------------------------------------------------------

    x1 = np.zeros(len(t))
    x2 = np.zeros(len(t))

    for k in range(len(t) - 1):

        x1[k + 1] = (
            0.95 * x1[k]
            + 0.10 * x2[k]
            + 0.20 * u[k]
        )

        x2[k + 1] = (
            -0.10 * x1[k]
            + 0.95 * x2[k]
            + 0.10 * u[k]
        )

    # ---------------------------------------------------------
    # 4. Create synthetic measurements
    # ---------------------------------------------------------

    # Measurement 1 contains a sine component.
    y1 = x1 + 0.20 * np.sin(2.0 * t)

    # Measurement 2 contains a cosine component.
    y2 = x2 + 0.30 * np.cos(3.0 * t)

    # ---------------------------------------------------------
    # 5. Add measurement noise
    # ---------------------------------------------------------

    np.random.seed(42)

    noise = 0.02 * np.random.randn(len(t))

    y1 = y1 + noise
    y2 = y2 + noise

    # ---------------------------------------------------------
    # 6. Arrange data for DMD
    # ---------------------------------------------------------

    # Each row is one measured variable.
    X = np.vstack((y1, y2))

    # Control input must also be arranged as rows.
    U = u.reshape(1, -1)

    return t, X, U