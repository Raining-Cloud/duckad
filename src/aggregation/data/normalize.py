from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-axis normalization constants over collection.collect's recorded "
        "trajectories, for scaling (x, y) waypoints to [-1, 1] before feeding them to the model."
    )
    parser.add_argument(
        "dir",
        help="run directory (or parent of several run directories) containing batch_*.npz files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="path to save the constants JSON (default: <dir>/trajectory_norm.json)",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=None,
        metavar="P",
        help="use the [P, 100-P] percentile of each axis instead of the true min/max, so a "
        "handful of extreme-curvature frames don't compress the common case into a sliver of "
        "[-1, 1]. Points outside the range will land slightly outside [-1, 1] and should be "
        "clipped by the caller.",
    )
    return parser.parse_args()


def load_trajectories(root: Path, key: str = "trajectory") -> np.ndarray:
    batch_paths = sorted(root.rglob("batch_*.npz"))
    if not batch_paths:
        raise FileNotFoundError(f"No batch_*.npz files found under {root}")
    trajectories = []
    for path in batch_paths:
        with np.load(path) as data:
            if key not in data:
                return np.empty((0,))
            trajectories.append(data[key])
    return np.concatenate(trajectories)


def trajectory_norm_constants(
    trajectory: np.ndarray, clip_percentile: float | None = None
) -> dict[str, float]:
    """trajectory: (S, N, 2) local-frame waypoints (x forward, y right; see trajectory.py).
    Returns a single (x_min, x_max, y_min, y_max) pair applied uniformly across every waypoint
    index - one scale per axis, not one per index - so the normalization is a rigid affine map
    of the (x, y) plane: a trajectory's geometric shape is preserved, and a normalized point can
    be un-normalized without knowing which waypoint index it came from."""
    x, y = trajectory[..., 0], trajectory[..., 1]
    if clip_percentile is None:
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
    else:
        x_min, x_max = (
            float(v) for v in np.percentile(x, [clip_percentile, 100 - clip_percentile])
        )
        y_min, y_max = (
            float(v) for v in np.percentile(y, [clip_percentile, 100 - clip_percentile])
        )
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def normalize(trajectory: np.ndarray, constants: dict[str, float]) -> np.ndarray:
    """Maps (x, y) meters into [-1, 1] per axis: norm = 2 * (v - min) / (max - min) - 1."""
    out = np.empty_like(trajectory)
    out[..., 0] = (
        2
        * (trajectory[..., 0] - constants["x_min"])
        / (constants["x_max"] - constants["x_min"])
        - 1
    )
    out[..., 1] = (
        2
        * (trajectory[..., 1] - constants["y_min"])
        / (constants["y_max"] - constants["y_min"])
        - 1
    )
    return out


def denormalize(normalized: np.ndarray, constants: dict[str, float]) -> np.ndarray:
    """Inverse of normalize(): maps [-1, 1] back to (x, y) meters."""
    out = np.empty_like(normalized)
    out[..., 0] = (normalized[..., 0] + 1) / 2 * (
        constants["x_max"] - constants["x_min"]
    ) + constants["x_min"]
    out[..., 1] = (normalized[..., 1] + 1) / 2 * (
        constants["y_max"] - constants["y_min"]
    ) + constants["y_min"]
    return out


def checkpoint_norm_constants(checkpoint: dict) -> dict[str, float]:
    """The constants a checkpoint was trained with. Checkpoints carry their own copy so they run
    without the training data; older ones fall back to <data.dir>/trajectory_norm.json."""
    if "norm_constants" in checkpoint:
        return checkpoint["norm_constants"]
    path = Path(checkpoint["config"]["data"]["dir"]) / "trajectory_norm.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint has no embedded norm_constants and {path} does not exist"
        )
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    root = Path(args.dir)
    trajectory = load_trajectories(root)
    constants = trajectory_norm_constants(trajectory, args.clip_percentile)

    print(
        f"Loaded {len(trajectory)} trajectories ({trajectory.shape[1]} waypoints each) from {root}\n"
    )
    label = (
        "true min/max"
        if args.clip_percentile is None
        else f"{args.clip_percentile:g}/{100 - args.clip_percentile:g} percentile"
    )
    print(
        f"Spatial trajectory normalization constants ({label}, x forward / y right, meters):"
    )
    print(f"  x: [{constants['x_min']:+.3f}, {constants['x_max']:+.3f}]")
    print(f"  y: [{constants['y_min']:+.3f}, {constants['y_max']:+.3f}]")

    renormalized = normalize(trajectory, constants)
    print(
        f"\nApplied to the loaded data: "
        f"x in [{renormalized[..., 0].min():+.3f}, {renormalized[..., 0].max():+.3f}], "
        f"y in [{renormalized[..., 1].min():+.3f}, {renormalized[..., 1].max():+.3f}]"
        + (
            " (in range by construction)"
            if args.clip_percentile is None
            else " (outliers land outside [-1, 1] - clip them)"
        )
    )

    temporal = load_trajectories(root, "temporal_trajectory")
    if temporal.size:
        tconst = trajectory_norm_constants(temporal, args.clip_percentile)
        constants.update(
            {
                "tx_min": tconst["x_min"],
                "tx_max": tconst["x_max"],
                "ty_min": tconst["y_min"],
                "ty_max": tconst["y_max"],
            }
        )
        print(f"\nTemporal trajectory normalization constants ({label}, meters):")
        print(f"  x: [{tconst['x_min']:+.3f}, {tconst['x_max']:+.3f}]")
        print(f"  y: [{tconst['y_min']:+.3f}, {tconst['y_max']:+.3f}]")
    else:
        print(
            "\nNo temporal_trajectory found in this run - skipping temporal constants."
        )

    out_path = Path(args.out) if args.out else root / "trajectory_norm.json"
    out_path.write_text(json.dumps(constants, indent=2) + "\n")
    print(f"\nSaved normalization constants to {out_path}")


if __name__ == "__main__":
    main()
