from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..aggregation.command import Command
from ..aggregation.config import BEVConfig, TargetImageConfig, TrajectoryConfig
from ..aggregation.data.normalize import checkpoint_norm_constants, denormalize
from ..aggregation.imaging import BEV_NUM_CLASSES, letterbox
from ..model import DuckAD


class TrajectoryPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str):
        self.device = device
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        self.norm_constants = checkpoint_norm_constants(checkpoint)

        c = self.norm_constants
        self.temporal_constants = (
            {
                "x_min": c["tx_min"],
                "x_max": c["tx_max"],
                "y_min": c["ty_min"],
                "y_max": c["ty_max"],
            }
            if "tx_min" in c
            else None
        )

        self.temporal_dt = (checkpoint["config"].get("data") or {}).get("temporal_dt")

        model_cfg = checkpoint["config"].get("model") or {}
        self.model = DuckAD(
            num_waypoints=TrajectoryConfig().num_points,
            bev_classes=BEV_NUM_CLASSES,
            bev_size=BEVConfig().image_size,
            head=model_cfg.get("head", "gru"),
            pretrained=False,
        ).to(device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @torch.no_grad()
    def predict(
        self, main_bgr: np.ndarray, command: int = int(Command.DEFAULT)
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int | None]:
        img = letterbox(main_bgr, TargetImageConfig())
        tensor = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        command_tensor = torch.full(
            (1,), int(command), dtype=torch.long, device=self.device
        )

        with torch.autocast(device_type=self.device, enabled=self.device == "cuda"):
            pred_wp, pred_temporal, pred_bev, pred_cmd = self.model(
                tensor, command_tensor
            )

        trajectory = denormalize(pred_wp.float().cpu().numpy(), self.norm_constants)[0]
        temporal_trajectory = (
            denormalize(pred_temporal.float().cpu().numpy(), self.temporal_constants)[0]
            if self.temporal_constants is not None
            else None
        )
        bev_class_idx = (
            pred_bev.argmax(1)[0].cpu().numpy() if pred_bev is not None else None
        )
        predicted_command = (
            int(pred_cmd.argmax(1).item()) if pred_cmd is not None else None
        )
        return trajectory, temporal_trajectory, bev_class_idx, predicted_command
