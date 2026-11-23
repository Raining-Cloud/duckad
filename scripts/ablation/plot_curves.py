"""Plot ablation test-loss curves (wp + cmd) aggregated over seeds.

Reads checkpoints/<cell>-s<seed>-<timestamp>/training_curve.pkl and draws one figure per
comparison on a log-scale axis: solid = test waypoint loss, dashed = test command loss,
mean +/- std across seeds. Only held-out metrics are shown - train loss is not comparable
across cells (noaug fits easier data).

    python scripts/ablation/plot_curves.py
    python scripts/ablation/plot_curves.py --root checkpoints --out figures/ablation
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FONT_SCALE = 1.1

sns.set_theme(style="whitegrid", rc={
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 16.5 * FONT_SCALE,
    "axes.labelsize": 16.5 * FONT_SCALE,
    "xtick.labelsize": 15 * FONT_SCALE,
    "ytick.labelsize": 15 * FONT_SCALE,
    "legend.fontsize": 15 * FONT_SCALE,
    "axes.edgecolor": ".8",
    "axes.linewidth": 1,
})

# Color follows the cell in every figure (fixed assignment, never cycled).
COLORS = {
    "base":       "#4a86c8",  # blue
    "mlp":        "#9a6bb5",  # purple
    "nopretrain": "#c27ba0",  # magenta
    "nobg":       "#d19a4a",  # orange
    "noaug":      "#6f9e67",  # green
}
# Explicit pastels for the +/- std bands; tint() covers the rest.
BANDS = {
    "base":  "#83B3E6",
    "mlp":   "#C39ED4",
    "noaug": "#D5E6D1",
}


def tint(hex_color: str, f: float = 0.78) -> str:
    """Blend toward white for the +/- std bands (f = white fraction)."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#{:02x}{:02x}{:02x}".format(
        *(round(c * (1 - f) + 255 * f) for c in (r, g, b)))


COMPARISONS = {
    "head": {"base": "GRU Head", "mlp": "MLP Head"},
    "init": {"base": "Baseline", "nopretrain": "No Pretraining"},
    "bg":   {"base": "Baseline", "nobg": "No Background Swap"},
    "aug":  {"base": "Baseline", "noaug": "No Augmentation"},
}
METRICS = [("wp", "Waypoint Loss"), ("cmd", "Command Loss")]

def load_runs(root: str) -> dict[str, dict[str, np.ndarray]]:
    """-> {cell: {metric: array [seeds, epochs]}}"""
    runs: dict[str, dict[str, list]] = {}
    for pkl in sorted(glob.glob(os.path.join(root, "*", "training_curve.pkl"))):
        name = os.path.basename(os.path.dirname(pkl))
        m = re.fullmatch(r"([a-z]+)-s(\d+)-\d{8}-\d{6}", name)
        if not m:
            continue
        cell = m.group(1)
        hist = pickle.load(open(pkl, "rb"))["history"]
        if any(ep.get("test") is None for ep in hist):
            print(f"WARNING: {name} has epochs without test metrics, skipping")
            continue
        for metric, _ in METRICS:
            runs.setdefault(cell, {}).setdefault(metric, []).append(
                [ep["test"][metric] for ep in hist])
    out = {}
    for cell, metrics in runs.items():
        n_epochs = min(len(c) for c in metrics["wp"])
        out[cell] = {k: np.asarray([c[:n_epochs] for c in v]) for k, v in metrics.items()}
        print(f"{cell}: {out[cell]['wp'].shape[0]} seeds x {n_epochs} epochs")
    return out


STYLES = {"wp": "-", "cmd": (0, (4, 2))}  # solid = waypoint, dashed = command


def _draw_panel(ax, cells: dict, members: list[str]) -> None:
    """Draw one comparison; log y so both losses share the axis, decimal tick labels."""
    ax.set_yscale("log")
    ax.set_yticks([1, 0.5, 0.1, 0.04])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    for metric, _ in METRICS:
        for cell in members:
            arr = cells[cell][metric]
            epochs = np.arange(1, arr.shape[1] + 1)
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(epochs, mean, color=COLORS[cell], linewidth=2,
                    linestyle=STYLES[metric])
            ax.fill_between(epochs, mean - std, mean + std,
                            color=BANDS.get(cell, tint(COLORS[cell])),
                            alpha=0.45, linewidth=0)
    ax.set_xlabel("Epoch")
    last_epoch = max(cells[c]["wp"].shape[1] for c in members)
    ax.set_xlim(1, last_epoch)
    ax.set_xticks([1, 5, 10, 15])


def _cell_handle(cell: str, label: str):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=COLORS[cell], linewidth=2, label=label)


def _metric_handles():
    """Linestyle legend entries: gray, so they name no cell."""
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=".4", linewidth=2, linestyle=STYLES[m],
                   label=title) for m, title in METRICS]


def plot_comparison(cells: dict, labels: dict[str, str], out_path: str) -> None:
    members = [c for c in labels if c in cells]
    fig, ax = plt.subplots(figsize=(5.6, 3.8), facecolor="white")
    _draw_panel(ax, cells, members)
    handles = [_cell_handle(c, f"{labels[c]} (n={cells[c]['wp'].shape[0]})")
               for c in members] + _metric_handles()
    ax.legend(handles=handles, loc="center right", frameon=True,
              edgecolor=".8", framealpha=1)
    ax.set_ylabel("Test Loss")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="checkpoints")
    p.add_argument("--out", default="figures/ablation")
    args = p.parse_args()

    cells = load_runs(args.root)
    os.makedirs(args.out, exist_ok=True)
    for name, labels in COMPARISONS.items():
        if not any(c in cells for c in labels):
            print(f"{name}: no runs yet, skipping")
            continue
        plot_comparison(cells, labels, os.path.join(args.out, f"{name}.pdf"))


if __name__ == "__main__":
    main()
