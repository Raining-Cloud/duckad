from __future__ import annotations

import argparse
from pathlib import Path

import carla
import cv2
import pygame

from ..aggregation.command import COMMAND_NAMES, Command
from ..aggregation.config import TargetImageConfig
from ..aggregation.imaging import letterbox
from .config import EvalConfig
from .inference import TrajectoryPredictor
from .visualize import EvalViewer

DISPLAY_SIZE = 256
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
MAX_CHECKPOINTS = 4


def find_images(lab_dir: Path) -> list[Path]:
    return sorted(p for ext in IMAGE_EXTENSIONS for p in lab_dir.glob(ext))


def checkpoint_label(checkpoint: str) -> str:
    path = Path(checkpoint)
    return path.parent.name or path.name


def parse_args() -> argparse.Namespace:
    defaults = EvalConfig()
    parser = argparse.ArgumentParser(
        description="Compare DuckAD checkpoints on real lab images"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help=f"checkpoint paths to compare; maximum {MAX_CHECKPOINTS}",
    )
    parser.add_argument(
        "--lab-dir", default="data/lab_images", help="input image directory"
    )
    parser.add_argument(
        "--lookahead-index",
        type=int,
        default=defaults.lookahead_index,
        help="highlighted trajectory point index",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=defaults.device,
        help="inference device",
    )
    parser.add_argument(
        "--export",
        metavar="OUT.json",
        default="domain_gap_trajectories.json",
        help="trajectory export file",
    )
    return parser.parse_args()


def _export_records(
    predictors: list[tuple[str, TrajectoryPredictor]],
    predictions: list[tuple],
    image_name: str,
    command: int,
) -> list[dict]:
    return [
        {
            "image": image_name,
            "command": COMMAND_NAMES[int(command)],
            "checkpoint": label,
            "trajectory": [
                [round(float(x), 4), round(float(y), 4)] for x, y in trajectory
            ],
            "temporal": None
            if temporal is None
            else [[round(float(x), 4), round(float(y), 4)] for x, y in temporal],
            "predicted_command": None
            if pred_cmd is None
            else COMMAND_NAMES[int(pred_cmd)],
        }
        for (label, _), (trajectory, temporal, _, pred_cmd) in zip(
            predictors, predictions
        )
    ]


def main() -> None:
    args = parse_args()
    device = args.device

    if len(args.checkpoints) > MAX_CHECKPOINTS:
        raise SystemExit(
            f"at most {MAX_CHECKPOINTS} checkpoints can be stacked, got {len(args.checkpoints)}"
        )

    lab_dir = Path(args.lab_dir)
    images = find_images(lab_dir)
    if not images:
        raise SystemExit(
            f"no images ({', '.join(IMAGE_EXTENSIONS)}) found in {lab_dir}"
        )

    predictors = [
        (checkpoint_label(cp), TrajectoryPredictor(cp, device))
        for cp in args.checkpoints
    ]
    viewer = EvalViewer(DISPLAY_SIZE, DISPLAY_SIZE, rows=len(predictors))
    dummy_control = carla.VehicleControl()

    index = 0
    command = int(Command.DEFAULT)
    dirty = True
    predictions: list[tuple] = []
    clock = pygame.time.Clock()

    exported: list[dict] = []
    print(
        f"{len(images)} images | {len(predictors)} checkpoint(s) | SPACE next, BACKSPACE prev, arrows set command, E export frame -> {args.export}, ESC/Q quit"
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    index = (index + 1) % len(images)
                    dirty = True
                elif event.key == pygame.K_BACKSPACE:
                    index = (index - 1) % len(images)
                    dirty = True
                elif event.key == pygame.K_LEFT:
                    command, dirty = int(Command.LEFT), True
                elif event.key == pygame.K_RIGHT:
                    command, dirty = int(Command.RIGHT), True
                elif event.key == pygame.K_UP:
                    command, dirty = int(Command.STRAIGHT), True
                elif event.key == pygame.K_DOWN:
                    command, dirty = int(Command.DEFAULT), True
                elif event.key == pygame.K_e and predictions:
                    import json

                    exported.extend(
                        _export_records(
                            predictors, predictions, images[index].name, command
                        )
                    )
                    Path(args.export).write_text(
                        json.dumps(
                            {
                                "lab_dir": str(args.lab_dir),
                                "checkpoints": [label for label, _ in predictors],
                                "records": exported,
                            },
                            indent=2,
                        )
                    )
                    print(
                        f"exported {images[index].name} cmd={COMMAND_NAMES[command].upper()} "
                        f"({len(exported)} records total) -> {args.export}"
                    )

        if not running:
            break

        if dirty:
            image_bgr = cv2.imread(str(images[index]), cv2.IMREAD_COLOR)

            main_display = cv2.resize(
                letterbox(image_bgr, TargetImageConfig()), (DISPLAY_SIZE, DISPLAY_SIZE)
            )

            predictions = [
                predictor.predict(image_bgr, command) for _, predictor in predictors
            ]
            pygame.display.set_caption(
                f"DuckAD domain gap [{index + 1}/{len(images)}] {images[index].name}  cmd={COMMAND_NAMES[command].upper()}"
            )
            dirty = False

        for row, (
            (label, _predictor),
            (trajectory, temporal, bev_idx, cmd),
        ) in enumerate(zip(predictors, predictions)):
            target_index = min(args.lookahead_index, len(trajectory) - 1)
            if not viewer.render(
                main_display,
                bev_idx,
                trajectory,
                target_index,
                dummy_control,
                0.0,
                command,
                command,
                cmd,
                temporal_xy=temporal,
                row=row,
                row_label=label,
            ):
                running = False
                break
        clock.tick(30)

    viewer.close()


if __name__ == "__main__":
    main()
