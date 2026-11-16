import matplotlib.pyplot as plt

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         ink="#0b0b0b", muted="#52514e", grid="#e4e3df")
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": C["muted"], "axes.labelcolor": C["ink"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.8,
    "xtick.color": C["muted"], "ytick.color": C["muted"],
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def show():
    """Open the figure window (skipped automatically if there is no screen)."""
    if plt.get_backend().lower() != "agg":
        plt.show()
