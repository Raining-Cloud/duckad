"""Plot one exported domain-gap prediction: spatial vs temporal side by side.

Reads the E-key export from src.eval.domain_gap (domain_gap_trajectories.json)
and draws the LAST exported (image, command) group by default: left panel the
arc-length-spaced spatial waypoints, right panel the time-spaced temporal
points (marker spacing encodes speed), one color per checkpoint cell using the
ablation color scheme. Top-down vehicle frame: origin = camera pose, forward
is up.

    python scripts/ablation/plot_domain_gap_traj.py
    python scripts/ablation/plot_domain_gap_traj.py --image 20260702_165115.jpg --command left
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_curves import COLORS, FONT_SCALE  # also applies the shared seaborn theme

LABELS = {
    "base": "GRU",
    "mlp": "MLP",
    "nopretrain": "No Pretraining",
    "nobg": "No Background Swap",
    "noaug": "No Augmentation",
}
PANELS = [("trajectory", "Spatial"), ("temporal", "Temporal")]
SIM_TO_REAL = 20  # sim meters -> real duckietown meters (the lab track is 1:20)
ASPECT = 2  # forward (y) axis drawn at twice the lateral (x) scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--json", default="domain_gap_trajectories.json")
    p.add_argument("--image", default=None, help="image name; default: last exported")
    p.add_argument("--command", default=None, help="command name; default: last exported")
    p.add_argument("--out", default="figures/ablation/mlp_gru")
    args = p.parse_args()

    records = json.load(open(args.json))["records"]
    image = args.image or records[-1]["image"]
    command = args.command or records[-1]["command"]
    group = [r for r in records if r["image"] == image and r["command"] == command]
    if not group:
        raise SystemExit(f"no records for image={image} command={command}")

    # Same 6-inch canvas width as the aug/bev figure: both files are saved
    # y-cropped at full width, so a common LaTeX width gives identical scale.
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), facecolor="white",
                             sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        for r in group:
            pts = r[key]
            if pts is None:
                continue
            cell = r["checkpoint"].split("-")[0]
            ys = [pt[1] / SIM_TO_REAL for pt in pts]  # lateral (right positive) -> horizontal
            xs = [pt[0] / SIM_TO_REAL for pt in pts]  # forward -> vertical
            ax.plot(ys, xs, "-o", color=COLORS.get(cell, ".4"), linewidth=2,
                    markersize=5, label=LABELS.get(cell, cell))
        ax.plot(0, 0, marker="^", color=".3", markersize=12)  # ego vehicle pose
        ax.set_title(title, fontsize=16.5 * FONT_SCALE)
        ax.set_xlabel("Lateral (m)")
        ax.set_aspect(ASPECT)
    axes[0].set_ylabel("Forward (m)")
    # End the y-axis exactly on its last tick, so tick gridline and panel border
    # coincide instead of drawing as a close double line.
    fig.canvas.draw()
    bottom, top = axes[0].get_ylim()
    axes[0].set_ylim(bottom, max(t for t in axes[0].get_yticks() if t <= top))
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    # The fixed aspect centers each panel in its slot; anchor them toward each
    # other so the remaining slack ends up at the figure edges, not between them.
    axes[0].set_anchor("E")
    axes[1].set_anchor("W")
    # Recenter the content (y-label + panels) on the canvas: the y-label eats
    # into the left slot, pushing the visual block right of figure center.
    for _ in range(3):  # iterate: each shift changes the measured extent slightly
        fig.canvas.draw()
        tb = fig.get_tightbbox(fig.canvas.get_renderer())
        w = fig.get_size_inches()[0]
        dx = (w - (tb.x0 + tb.x1)) / 2 / w
        sp = fig.subplotpars
        fig.subplots_adjust(left=sp.left + dx, right=sp.right + dx)
    axes[1].legend(loc="center right", bbox_to_anchor=(1, 0.35),
                   frameon=True, edgecolor=".8", framealpha=1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Crop top/bottom whitespace but keep the FULL canvas width, matching the
    # aug/bev figure so both render at identical scale in LaTeX.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w = fig.get_size_inches()[0]
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor="white",
                    pad_inches=0.02, bbox_inches=Bbox([[0, tb.y0], [w, tb.y1]]))
        print(f"wrote {args.out}.{ext}  ({image}, cmd={command})")
    plt.close(fig)


if __name__ == "__main__":
    main()
