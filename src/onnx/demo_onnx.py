"""Standalone example script for model loading and deployment.

Usage:
    python -m src.onnx.demo_onnx model.onnx frame.jpg
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import onnxruntime as ort
import pygame

IMG_SIZE = 224
DEFAULT_TEMPORAL_DT = 0.25
LOOKAHEAD_INDEX = 2
PANEL = 256
HUD_H = 40
TRAJ_RANGE_M = 10.0

COMMAND_NAMES = ["default", "left", "straight", "right"]
KEY_TO_COMMAND = {"left": 1, "right": 3, "up": 2, "down": 0}
BEV_COLORS_BGR = np.array(
    [
        (30, 30, 30),
        (0, 215, 255),
        (255, 255, 255),
        (60, 60, 60),
        (30, 30, 200),
        (255, 120, 60),
        (0, 140, 255),
        (255, 0, 255),
    ],
    dtype=np.uint8,
)


def letterbox(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized

    return canvas


def denorm(points: np.ndarray, c: dict, keys: tuple[str, str, str, str]) -> np.ndarray:
    lo = np.array([c[keys[0]], c[keys[1]]])
    span = np.array([c[keys[2]] - c[keys[0]], c[keys[3]] - c[keys[1]]])

    return (points + 1.0) * 0.5 * span + lo


class OnnxModel:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map

        if "trajectory_norm" not in meta:
            raise SystemExit(
                "ONNX has no embedded norm stats; re-export with src.onnx.export_onnx"
            )
        
        self.norm = json.loads(meta["trajectory_norm"])
        self.dt = float(meta.get("temporal_dt", DEFAULT_TEMPORAL_DT))

    def predict(self, bgr: np.ndarray, command: int):
        letterboxed = letterbox(bgr)
        image_input = letterboxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = dict(
            zip(
                (o.name for o in self.session.get_outputs()),
                self.session.run(
                    None,
                    {"image": image_input, "command": np.array([command], np.int64)},
                ),
            )
        )
        spatial = denorm(
            out["spatial"][0], self.norm, ("x_min", "y_min", "x_max", "y_max")
        )
        temporal = denorm(
            out["temporal"][0], self.norm, ("tx_min", "ty_min", "tx_max", "ty_max")
        )
        bev_idx = out["bev"][0].argmax(0)
        action = int(out["command_logits"][0].argmax())
        speed_kmh = float(np.linalg.norm(temporal[0]) / self.dt * 3.6)

        return letterboxed, spatial, temporal, bev_idx, action, speed_kmh


class Viewer:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("monospace", 16)
        self.display = pygame.display.set_mode((PANEL * 3, PANEL + HUD_H))
        pygame.display.set_caption("DuckAD ONNX demo")

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        scale = PANEL / (2 * TRAJ_RANGE_M)
        cx, cy = PANEL // 2, PANEL - 15

        return int(cx + y * scale), int(cy - x * scale)

    def render(
        self, letterboxed_bgr, spatial, temporal, bev_idx, action, speed_kmh, command
    ):
        self.display.fill((0, 0, 0))

        hud = (
            f"predicted action: {COMMAND_NAMES[action].upper():8s} "
            f"predicted speed: {speed_kmh:5.1f} km/h   fed: {COMMAND_NAMES[command].upper()}"
        )
        self.display.blit(self.font.render(hud, True, (255, 255, 255)), (8, 12))

        img_surf = pygame.surfarray.make_surface(
            letterboxed_bgr[:, :, ::-1].swapaxes(0, 1)
        )
        self.display.blit(pygame.transform.scale(img_surf, (PANEL, PANEL)), (0, HUD_H))
        bev_surf = pygame.surfarray.make_surface(
            BEV_COLORS_BGR[bev_idx][:, :, ::-1].swapaxes(0, 1)
        )
        self.display.blit(
            pygame.transform.scale(bev_surf, (PANEL, PANEL)), (PANEL, HUD_H)
        )

        traj = pygame.Surface((PANEL, PANEL))
        traj.fill((20, 20, 20))

        for x, y in temporal:
            pygame.draw.circle(traj, (0, 200, 255), self._to_px(x, y), 2)

        target = min(LOOKAHEAD_INDEX, len(spatial) - 1)

        for i, (x, y) in enumerate(spatial):
            color = (255, 255, 0) if i == target else (255, 140, 0)
            pygame.draw.circle(traj, color, self._to_px(x, y), 5 if i == target else 3)

        pygame.draw.circle(traj, (255, 255, 255), (PANEL // 2, PANEL - 15), 4)
        self.display.blit(traj, (PANEL * 2, HUD_H))

        pygame.display.flip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "onnx_model", help="path to the exported .onnx model (see src.onnx.export_onnx)"
    )
    parser.add_argument(
        "image", help="path to a .jpg/.png frame to run through the model"
    )
    args = parser.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read image {args.image!r}")

    model = OnnxModel(args.onnx_model)
    viewer = Viewer()
    command = 0
    viewer.render(*model.predict(bgr, command), command)

    print("arrows set command (LEFT/RIGHT/UP=straight, DOWN=default), ESC/Q to quit")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif pygame.key.name(event.key) in KEY_TO_COMMAND:
                    command = KEY_TO_COMMAND[pygame.key.name(event.key)]
                    viewer.render(*model.predict(bgr, command), command)
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
