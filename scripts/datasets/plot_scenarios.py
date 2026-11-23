"""Render the triple-camera rig figure for the presentation (companion to
plot_augmentations.py, same style and dimensions).

One driving moment (mid-way through a junction left turn, at partial speed, with a
traffic bot alongside, from the traffic + roadside-duckies replay) seen simultaneously by the three camera rigs - left / front /
right, the DAVE-2-style recovery setup. Columns are the rigs; rows show the rig's
camera image, its raw BEV label map (untouched - the BEV target is never augmented),
and its rig-frame trajectory panel (PygameViewer-style). All three rigs
record the SAME driven path, but
each expresses it in its own shifted ground frame - visible as the lateral offset
between the three trajectory panels, which is what teaches recovery behavior. Spatial
waypoints are drawn in the command color (gray/blue/green/orange = default/left/
straight/right, the repo-wide convention); the yellow polyline is the temporal
("speed") trajectory. Run from the repo root.

Usage: uv run python scripts/datasets/plot_scenarios.py [--episode DIR --frame ID]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aggregation.command import COMMAND_COLORS_BGR, Command
from src.aggregation.imaging import DUCKIETOWN_LABELS

# One tick, three rigs: collect.py writes the rigs as consecutive rows sharing one
# frame id, in rig-list order front, left, right. This triplet is mid-way through a
# junction left turn (~3.4 m/s, partial speed) with a traffic bot alongside, so the
# spatial waypoints curve left immediately and the temporal polyline shows the
# slower in-turn pace.
EPISODE = "data/final/ducks_roadside_traffic/duckietown_04_lab_01_20260711-141631"
FRAME = 99578
RIG_COLUMNS = [("Left camera", 1), ("Front camera", 0), ("Right camera", 2)]
OUT = Path("figures/scenarios.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"

# BEV geometry (config.BEVConfig / CameraRigConfig): footprint center 8.5m ahead,
# 12.1m square; front camera 1.62m ahead of the vehicle origin. Each rig's BEV and
# trajectory are expressed in that rig's own shifted frame, so these constants apply
# to every column unchanged.
BEV_X, BEV_SIZE_M = 8.5, 12.1
CAM_XY = (1.62, 0.0)
S = 448  # supersampled panel resolution for crisp overlay drawing

# BGR, from visualize._CLASS_COLORS_BGR - kept identical so this figure matches the
# live preview tooling.
CLASS_COLORS_BGR = {
    "center_lane": (0, 215, 255),
    "side_lane": (255, 255, 255),
    "asphalt": (60, 60, 60),
    "stop_lane": (30, 30, 200),
    "sign": (255, 120, 60),
    "bot": (0, 140, 255),
    "duck": (255, 0, 255),
}
LEGEND = ["center_lane", "side_lane", "stop_lane", "sign", "bot", "duck"]
TEMPORAL_RGB = (255, 255, 0)


def command_rgb(command: int) -> tuple[int, int, int]:
    b, g, r = COMMAND_COLORS_BGR[Command(command)]
    return (r, g, b)


def colorize_bev(ids: np.ndarray) -> np.ndarray:
    """(64,64) raw tag ids -> (S,S,3) RGB preview, visualize.py colors."""
    bgr = np.full((*ids.shape, 3), (30, 30, 30), dtype=np.uint8)
    for name, tag_id in DUCKIETOWN_LABELS.items():
        bgr[ids == tag_id] = CLASS_COLORS_BGR[name]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (S, S), interpolation=cv2.INTER_NEAREST)


def draw_trajectories(img: np.ndarray, spatial: np.ndarray, temporal: np.ndarray,
                      command: int, to_px) -> None:
    """Command-colored spatial waypoints + yellow temporal polyline, PygameViewer-style."""
    prev = None
    for x, y in temporal:
        pt = to_px(x, y)
        if prev is not None:
            cv2.line(img, prev, pt, TEMPORAL_RGB, 2, cv2.LINE_AA)
        cv2.circle(img, pt, 4, TEMPORAL_RGB, -1, cv2.LINE_AA)
        prev = pt
    for x, y in spatial:
        cv2.circle(img, to_px(x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, to_px(x, y), 5, command_rgb(command), -1, cv2.LINE_AA)


def trajectory_panel(spatial: np.ndarray, temporal: np.ndarray, command: int) -> np.ndarray:
    """Dark vehicle-frame trajectory plot: BEV footprint rectangle, camera/origin
    markers, spatial dots, temporal polyline - matching the PygameViewer layout."""
    img = np.full((S, S, 3), 20, dtype=np.uint8)
    scale = S / (2 * BEV_SIZE_M * 0.8)
    cx, cy = S // 2, S - round(15 * S / 256)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(round(cx + y * scale)), int(round(cy - x * scale))

    half = BEV_SIZE_M / 2
    rect = np.array([to_px(BEV_X + half, half), to_px(BEV_X + half, -half),
                     to_px(BEV_X - half, -half), to_px(BEV_X - half, half)])
    cv2.polylines(img, [rect], True, (0, 200, 200), 2, cv2.LINE_AA)
    cv2.circle(img, to_px(BEV_X, 0), 6, (30, 120, 255), -1, cv2.LINE_AA)
    cv2.circle(img, to_px(*CAM_XY), 6, (0, 255, 0), -1, cv2.LINE_AA)
    draw_trajectories(img, spatial, temporal, command, to_px)
    cv2.circle(img, (cx, cy), 7, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def load_tick(episode: str, frame: int) -> dict[str, np.ndarray]:
    """The rows recorded at one frame id (one per rig, front/left/right). Located by episode
    and frame id, not shard name, so it works on both raw recordings and the Hub rebuild."""
    for shard in sorted(Path(episode).glob("batch_*.npz")):
        with np.load(shard) as d:
            hit = np.flatnonzero(d["frame"] == frame)
            if hit.size:
                return {k: d[k][hit] for k in d.files if d[k].ndim > 0}
    raise FileNotFoundError(f"frame {frame} not found under {episode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", default=EPISODE, help="episode directory of batch_*.npz")
    parser.add_argument("--frame", type=int, default=FRAME, help="frame id of the tick to draw")
    args = parser.parse_args()
    d = load_tick(args.episode, args.frame)
    n_cols = len(RIG_COLUMNS)
    fig = plt.figure(figsize=(5 * 2.35, 11.8), facecolor=SURFACE)
    gs = fig.add_gridspec(3, n_cols,
                          left=0.01, right=0.99, top=0.99, bottom=0.01,
                          hspace=0.03, wspace=0.03)

    for col, (title, index) in enumerate(RIG_COLUMNS):
        command = int(d["command"][index])
        spatial, temporal = d["trajectory"][index], d["temporal_trajectory"][index]

        cam_ax = fig.add_subplot(gs[0, col])
        cam_ax.imshow(cv2.cvtColor(d["main_img"][index], cv2.COLOR_BGR2RGB))

        bev_ax = fig.add_subplot(gs[1, col])
        bev_ax.imshow(colorize_bev(d["bev_seg"][index][0]))

        traj_ax = fig.add_subplot(gs[2, col])
        traj_ax.imshow(trajectory_panel(spatial, temporal, command))

        cam_ax.set_axis_off()
        bev_ax.set_axis_off()
        traj_ax.set_axis_off()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
