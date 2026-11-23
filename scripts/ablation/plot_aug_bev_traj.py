"""Combined presentation figure: ablation loss curves + domain-gap trajectories.

One slide canvas holding both result figures. Layout "stacked" (default) puts
the BEV and augmentation test-loss panels side by side on top and the spatial /
temporal domain-gap trajectory panels below; "wide" puts the stacked loss
panels left and the full-height trajectory panels right. Style, colors and
data loading are imported from plot_curves / plot_domain_gap_traj so the
combined figure can never drift from the standalone ones.

    python scripts/ablation/plot_aug_bev_traj.py
    python scripts/ablation/plot_aug_bev_traj.py --layout wide
    python scripts/ablation/plot_aug_bev_traj.py --image 20260702_165115.jpg --command left
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

from plot_curves import (COMPARISONS, FONT_SCALE, _cell_handle, _draw_panel,
                         _metric_handles, load_runs)
from plot_domain_gap_traj import COLORS, LABELS, PANELS, SIM_TO_REAL

# Slide variant: the MLP head in grey, keeping the accent colors on the cells.
TRAJ_COLORS = {**COLORS, "mlp": "#7f7f7f"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default="checkpoints")
    p.add_argument("--json", default="domain_gap_trajectories.json")
    p.add_argument("--image", default=None, help="image name; default: last exported")
    p.add_argument("--command", default=None, help="command name; default: last exported")
    p.add_argument("--layout", choices=("stacked", "wide"), default="stacked")
    p.add_argument("--out", default="figures/aug_bev_traj")
    args = p.parse_args()

    cells = load_runs(args.root)
    records = json.load(open(args.json))["records"]
    image = args.image or records[-1]["image"]
    command = args.command or records[-1]["command"]
    group = [r for r in records if r["image"] == image and r["command"] == command]
    if not group:
        raise SystemExit(f"no records for image={image} command={command}")

    # Explicit margins everywhere: tight_layout cannot handle fixed-aspect /
    # row-spanning axes (it distorts the slots against each other).
    if args.layout == "wide":
        fig = plt.figure(figsize=(12.6, 5.6), facecolor="white")
        gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1],
                              left=0.055, right=0.995, top=0.90, bottom=0.11,
                              wspace=0.28, hspace=0.16)
        ax_bev = fig.add_subplot(gs[0, 0])
        ax_aug = fig.add_subplot(gs[1, 0], sharex=ax_bev, sharey=ax_bev)
        ax_sp = fig.add_subplot(gs[:, 1])
        ax_tm = fig.add_subplot(gs[:, 2], sharex=ax_sp, sharey=ax_sp)
    else:
        fig = plt.figure(figsize=(8.5, 8), facecolor="white")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5],
                              left=0.09, right=0.98, top=0.92, bottom=0.075,
                              wspace=0.12, hspace=0.5)
        ax_bev = fig.add_subplot(gs[0, 0])
        ax_aug = fig.add_subplot(gs[0, 1], sharex=ax_bev, sharey=ax_bev)
        ax_sp = fig.add_subplot(gs[1, 0])
        ax_tm = fig.add_subplot(gs[1, 1], sharex=ax_sp, sharey=ax_sp)

    # Loss panels: BEV and Aug comparisons (same panels as plot_curves).
    for ax, name in zip((ax_bev, ax_aug), ("bev", "aug")):
        labels = COMPARISONS[name]
        members = [c for c in labels if c in cells]
        _draw_panel(ax, cells, members)
        ax.legend(handles=[_cell_handle(c, labels[c]) for c in members],
                  loc="center right", frameon=True, edgecolor=".8", framealpha=1)
    ax_bev.set_ylabel("Test Loss")
    if args.layout == "wide":  # stacked column: only the bottom panel keeps x labels
        ax_aug.set_ylabel("Test Loss")
        ax_bev.set_xlabel("")
        ax_bev.tick_params(labelbottom=False)
    else:  # side by side: only the left panel keeps y labels
        ax_aug.tick_params(labelleft=False)

    # Trajectory panels: spatial + temporal (same panels as plot_domain_gap_traj).
    for ax, (key, title) in zip((ax_sp, ax_tm), PANELS):
        for r in group:
            pts = r[key]
            if pts is None:
                continue
            cell = r["checkpoint"].split("-")[0]
            ys = [pt[1] / SIM_TO_REAL for pt in pts]  # lateral -> horizontal
            xs = [pt[0] / SIM_TO_REAL for pt in pts]  # forward -> vertical
            ax.plot(ys, xs, "-o", color=TRAJ_COLORS.get(cell, ".4"), linewidth=2,
                    markersize=5, label=LABELS.get(cell, cell))
        ax.plot(0, 0, marker="^", color=".3", markersize=12)  # ego vehicle pose
        ax.set_title(title, fontsize=16.5 * FONT_SCALE)
        ax.set_xlabel("Lateral (m)")
        ax.set_aspect("equal")  # 1:1, unlike the paper figure's 2:1 forward stretch
    ax_sp.set_ylabel("Forward (m)")
    ax_tm.tick_params(labelleft=False)
    ax_tm.legend(loc="center right", bbox_to_anchor=(1, 0.35),
                 frameon=True, edgecolor=".8", framealpha=1)
    # The fixed-aspect trajectory panels letterbox inside their slots; anchor
    # them toward each other so the slack lands at the figure edges instead.
    ax_sp.set_anchor("E")
    ax_tm.set_anchor("W")

    # Linestyle legend (waypoint/command) above the loss panels: right-aligned
    # over the loss column in the wide layout, centered over both panels stacked.
    top = gs.top + 0.005
    if args.layout == "wide":
        loc, anchor = "lower right", (ax_bev.get_position().x1, top)
    else:
        loc, anchor = "lower center", ((gs.left + gs.right) / 2, top)
    fig.legend(handles=_metric_handles(), loc=loc, ncol=2,
               frameon=True, edgecolor=".8", framealpha=1,
               bbox_to_anchor=anchor, borderaxespad=0,
               fontsize=12.5 * FONT_SCALE, handlelength=1.3, columnspacing=1.0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, facecolor="white",
                    bbox_inches="tight", pad_inches=0.05)
        print(f"wrote {args.out}.{ext}  ({image}, cmd={command})")
    plt.close(fig)


if __name__ == "__main__":
    main()
