from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..command import COMMAND_NAMES
from ..imaging import BEV_CLASS_NAMES, BEV_NUM_CLASSES, bev_class_index
from ..trajectory import trajectory_turn_and_lateral

TURN_BIN_EDGES_DEG = [-20.0, -5.0, 5.0, 20.0]
TURN_BIN_LABELS = ["sharp_left", "left", "straight", "right", "sharp_right"]

RARE_CLASS_PRESENCE_THRESHOLD = 0.001

STOPPED_SPEED_MS = 0.1
DEAD_STOP_DISP_M = 0.05
CREEP_SPEED_MS = 1.0


def feature_groups(num_commands: int, num_bev: int) -> list[tuple[str, int]]:
    return [("geometry", 2), ("motion", 2), ("command", num_commands), ("bev", num_bev)]


def group_scaled_features(
    features: np.ndarray, groups: list[tuple[str, int]]
) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)
    start = 0
    for _, width in groups:
        scaled[:, start : start + width] /= np.sqrt(width)
        start += width
    return scaled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster collect.py's recorded batches by trajectory curvature and BEV "
        "scene content, and print the resulting distributions, to reveal dataset imbalance "
        "before training."
    )
    parser.add_argument(
        "dir",
        help="run directory (or parent of several run directories) containing batch_*.npz files",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="number of joint motion+BEV+trajectory clusters (KMeans)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="path to save the distribution plot PNG (default: <dir>/balance_report.png)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bev_class_fractions(bev_seg: np.ndarray) -> np.ndarray:
    class_index = bev_class_index(bev_seg)
    return np.stack(
        [(class_index == c).mean(axis=(1, 2)) for c in range(BEV_NUM_CLASSES)], axis=1
    )


def load_dataset(root: Path):
    batch_paths = sorted(root.rglob("batch_*.npz"))
    if not batch_paths:
        raise FileNotFoundError(f"No batch_*.npz files found under {root}")

    trajectories, fractions, commands, speeds, future_disp = (
        [],
        [],
        [],
        [],
        [],
    )
    for path in batch_paths:
        with np.load(path) as data:
            trajectory = data["trajectory"]
            trajectories.append(trajectory)
            fractions.append(bev_class_fractions(data["bev_seg"]))
            commands.append(data["command"])
            speeds.append(data["speed"])
            temporal = data["temporal_trajectory"]
            future_disp.append(np.linalg.norm(temporal[:, -1, :], axis=1))
    return (
        np.concatenate(trajectories),
        np.concatenate(fractions),
        np.concatenate(commands),
        np.concatenate(speeds),
        np.concatenate(future_disp),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.dir)
    trajectory, bev_fractions, command, speed, future_disp = load_dataset(root)
    turn_deg, lateral_m = trajectory_turn_and_lateral(trajectory)
    n = len(turn_deg)
    print(f"Loaded {n} samples from {root}\n")

    stopped = speed < STOPPED_SPEED_MS
    dead_stop = stopped & (future_disp < DEAD_STOP_DISP_M)
    about_to_go = stopped & ~dead_stop
    creeping = (~stopped) & (speed < CREEP_SPEED_MS)
    cruising = speed >= CREEP_SPEED_MS
    print(
        "Motion distribution (speed now vs displacement over the full temporal horizon):"
    )
    for label, mask in (
        ("dead_stop", dead_stop),
        ("about_to_go", about_to_go),
        ("creeping", creeping),
        ("cruising", cruising),
    ):
        print(f"  {label:12s}: {mask.sum():6d} ({100 * mask.mean():5.1f}%)")
    print()

    turn_bin_idx = np.digitize(turn_deg, TURN_BIN_EDGES_DEG)
    print(
        "Trajectory curvature distribution (net heading change, first->last segment):"
    )
    for label, count in zip(
        TURN_BIN_LABELS, np.bincount(turn_bin_idx, minlength=len(TURN_BIN_LABELS))
    ):
        print(f"  {label:12s}: {count:5d} ({100 * count / n:5.1f}%)")
    print()

    command_counts = np.bincount(command, minlength=len(COMMAND_NAMES))
    print("Recorded command distribution:")
    for label, count in zip(COMMAND_NAMES, command_counts):
        print(f"  {label:12s}: {count:5d} ({100 * count / n:5.1f}%)")
    print()

    print(
        f"BEV rare-class presence (share of frames with >{RARE_CLASS_PRESENCE_THRESHOLD:.1%} of the crop):"
    )
    for name, class_frac in zip(BEV_CLASS_NAMES, bev_fractions.T):
        present = (class_frac > RARE_CLASS_PRESENCE_THRESHOLD).mean()
        print(
            f"  {name:12s}: present in {100 * present:5.1f}% of frames  (mean pixel share {100 * class_frac.mean():4.1f}%)"
        )
    print()

    command_onehot = np.eye(len(COMMAND_NAMES))[command]
    features = np.column_stack(
        [turn_deg, lateral_m, speed, future_disp, command_onehot, bev_fractions]
    )
    groups = feature_groups(len(COMMAND_NAMES), BEV_NUM_CLASSES)
    scaled_features = group_scaled_features(features, groups)
    kmeans = KMeans(n_clusters=args.k, random_state=args.seed, n_init=10).fit(
        scaled_features
    )
    labels = kmeans.labels_
    counts = np.bincount(labels, minlength=args.k)

    bev_start = 4 + len(COMMAND_NAMES)
    print(f"Joint motion+command+BEV+trajectory clusters (k={args.k}), largest first:")
    order = np.argsort(-counts)
    for cluster_id in order:
        mean_feature = features[labels == cluster_id].mean(axis=0)
        dominant_bev = BEV_CLASS_NAMES[np.argmax(mean_feature[bev_start:])]
        dominant_cmd = COMMAND_NAMES[np.argmax(mean_feature[4:bev_start])]
        print(
            f"  cluster {cluster_id}: {counts[cluster_id]:6d} ({100 * counts[cluster_id] / n:5.1f}%)  "
            f"turn={mean_feature[0]:+6.1f}deg  lat={mean_feature[1]:+5.2f}m  "
            f"speed={mean_feature[2]:4.2f}m/s  horizon_disp={mean_feature[3]:5.2f}m  "
            f"cmd={dominant_cmd:8s} bev={dominant_bev}"
        )
    print(
        f"\nCluster imbalance ratio (largest/smallest): {counts.max() / counts.min():.1f}x"
    )

    out_path = Path(args.out) if args.out else root / "balance_report.png"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(turn_deg, bins=40)
    axes[0, 0].axvline(0, color="black", linewidth=0.5)
    axes[0, 0].set_title("Trajectory net turn angle (deg)")

    axes[0, 1].bar(BEV_CLASS_NAMES, bev_fractions.mean(axis=0))
    axes[0, 1].set_title("Mean BEV pixel share per class")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[0, 2].bar(COMMAND_NAMES, command_counts)
    axes[0, 2].set_title("Recorded command distribution")

    axes[1, 0].bar([str(c) for c in order], counts[order])
    axes[1, 0].set_title(f"Joint cluster sizes (k={args.k})")
    axes[1, 0].set_xlabel("cluster id")

    axes[1, 1].scatter(turn_deg, lateral_m, c=labels, cmap="tab10", s=8, alpha=0.6)
    axes[1, 1].set_xlabel("turn (deg)")
    axes[1, 1].set_ylabel("lateral offset (m)")
    axes[1, 1].set_title("Trajectory feature space, colored by cluster")

    axes[1, 2].scatter(speed, future_disp, c=labels, cmap="tab10", s=8, alpha=0.6)
    axes[1, 2].set_xlabel("speed now (m/s)")
    axes[1, 2].set_ylabel("displacement over horizon (m)")
    axes[1, 2].set_title("Motion feature space, colored by cluster")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved distribution plot to {out_path}")


if __name__ == "__main__":
    main()
