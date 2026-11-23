from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np

from ..config import PipelineConfig
from .visualize import PygameViewer

_RIG_NAMES = ("front", "left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay batch_*.npz files recorded by collect.py")
    parser.add_argument("dir", help="recording directory containing batch_*.npz files")
    parser.add_argument("--fps", type=float, default=20.0, help="playback rate; matches collect.py's default 20Hz (1/fixed_delta_seconds)")
    parser.add_argument("--loop", action="store_true", help="restart from the first batch after the last one plays")
    return parser.parse_args()


def _iter_samples(batch_paths: list[Path]):
    for path in batch_paths:
        with np.load(path) as data:
            main_img, bev_seg, trajectory, command = (
                data["main_img"],
                data["bev_seg"],
                data["trajectory"],
                data["command"],
            )
            # Temporal ("speed") trajectory - absent from pre-temporal runs; the viewer just
            # skips the overlay when it's None.
            temporal = data["temporal_trajectory"] if "temporal_trajectory" in data else None
            for i in range(len(main_img)):
                yield main_img[i], bev_seg[i], trajectory[i], int(command[i]), (None if temporal is None else temporal[i])


def iter_batches(batch_paths: list[Path]):
    """Regroups the flat per-rig sample stream back into per-tick
    {rig_name: (img, bev, traj, command, temporal)} dicts - collect.py writes samples in a fixed
    round-robin front/left/right order every
    tick regardless of batch-file boundaries, so grouping every 3 consecutive samples in the
    concatenated stream recovers that."""
    samples = _iter_samples(batch_paths)
    while True:
        group = list(itertools.islice(samples, len(_RIG_NAMES)))
        if not group:
            return
        yield dict(zip(_RIG_NAMES, group))


def main() -> None:
    args = parse_args()
    batch_paths = sorted(Path(args.dir).glob("batch_*.npz"))
    if not batch_paths:
        raise FileNotFoundError(f"No batch_*.npz files found in {args.dir}")

    cfg = PipelineConfig()
    main_rig = cfg.main_camera.rigs[0]
    viewer = PygameViewer(
        cfg.target_image.size,
        main_camera_xy=(main_rig.transform.x, main_rig.transform.y),
        bev_xy=(cfg.bev.transform.x, cfg.bev.transform.y),
        bev_size_m=cfg.bev.size_m,
    )

    frame_delay = 1.0 / args.fps
    try:
        while True:
            for rig_frames in iter_batches(batch_paths):
                if not viewer.render(rig_frames):
                    return
                time.sleep(frame_delay)
            if not args.loop:
                break
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
