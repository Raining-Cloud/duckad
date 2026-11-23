from __future__ import annotations

import carla
import numpy as np
import pygame

from ..aggregation.command import COMMAND_NAMES, Command


class EvalViewer:
    BEV_DISPLAY_SIZE = 256
    TRAJ_PLOT_SIZE = 256
    TRAJ_METERS_RANGE = 10.0

    MANUAL_STEER = 0.6
    MANUAL_REVERSE_THROTTLE = 0.5

    _COMMAND_KEYS = {
        pygame.K_LEFT: Command.LEFT,
        pygame.K_RIGHT: Command.RIGHT,
        pygame.K_UP: Command.STRAIGHT,
        pygame.K_DOWN: Command.DEFAULT,
    }

    def __init__(self, main_width: int, main_height: int, rows: int = 1):

        pygame.init()
        pygame.font.init()
        self.main_width = main_width
        self.rows = rows
        self.font = pygame.font.SysFont("monospace", 16)
        width = main_width + self.BEV_DISPLAY_SIZE + self.TRAJ_PLOT_SIZE
        self._row_height = max(main_height, self.BEV_DISPLAY_SIZE, self.TRAJ_PLOT_SIZE)
        self.display = pygame.display.set_mode((width, self._row_height * rows))
        pygame.display.set_caption("lee2e eval")
        self._command = int(Command.DEFAULT)
        self._keys_down = dict.fromkeys(self._COMMAND_KEYS, False)
        self._manual = False
        self._space_down = False

    def get_command(self) -> int:
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        for key, command in self._COMMAND_KEYS.items():
            down = keys[key]
            if down and not self._keys_down[key]:
                self._command = int(command)
            self._keys_down[key] = down
        return self._command

    def manual_control(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        space = keys[pygame.K_SPACE]
        if space and not self._space_down:
            self._manual = not self._manual
            pygame.display.set_caption(
                "lee2e eval [MANUAL: WASD]" if self._manual else "lee2e eval"
            )
        self._space_down = space
        if not self._manual:
            return None
        forward, back = keys[pygame.K_w], keys[pygame.K_s]
        throttle = 1.0 if forward else (self.MANUAL_REVERSE_THROTTLE if back else 0.0)
        reverse = back and not forward

        brake = 0.0 if (forward or back) else 1.0
        steer = (self.MANUAL_STEER if keys[pygame.K_d] else 0.0) - (
            self.MANUAL_STEER if keys[pygame.K_a] else 0.0
        )
        return (throttle, steer, brake, reverse)

    def render(
        self,
        main_img_bgr: np.ndarray,
        bev_class_idx: np.ndarray | None,
        trajectory_xy: np.ndarray,
        target_index: int,
        control: carla.VehicleControl,
        speed_kmh: float,
        command: int = int(Command.DEFAULT),
        executed_command: int = int(Command.DEFAULT),
        predicted_command: int | None = None,
        target_speed_kmh: float | None = None,
        temporal_xy: np.ndarray | None = None,
        selected_temporal_index: int | None = None,
        row: int = 0,
        row_label: str | None = None,
    ) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        y0 = row * self._row_height

        main_rgb = main_img_bgr[:, :, ::-1]
        self.display.blit(
            pygame.surfarray.make_surface(main_rgb.swapaxes(0, 1)), (0, y0)
        )

        if bev_class_idx is not None:
            bev_bgr = np.array(
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
            )[bev_class_idx]
            bev_surface = pygame.surfarray.make_surface(
                bev_bgr[:, :, ::-1].swapaxes(0, 1)
            )
            bev_surface = pygame.transform.scale(
                bev_surface, (self.BEV_DISPLAY_SIZE, self.BEV_DISPLAY_SIZE)
            )
            self.display.blit(bev_surface, (self.main_width, y0))

        traj_surface = pygame.Surface((self.TRAJ_PLOT_SIZE, self.TRAJ_PLOT_SIZE))
        traj_surface.fill((20, 20, 20))
        scale = self.TRAJ_PLOT_SIZE / (2 * self.TRAJ_METERS_RANGE)

        cx, cy = self.TRAJ_PLOT_SIZE // 2, self.TRAJ_PLOT_SIZE - 15

        def to_px(x: float, y: float) -> tuple[int, int]:
            return int(cx + y * scale), int(cy - x * scale)

        if temporal_xy is not None:
            for i, (x, y) in enumerate(temporal_xy):
                is_selected = i == selected_temporal_index
                color = (255, 0, 255) if is_selected else (0, 200, 255)
                pygame.draw.circle(
                    traj_surface, color, to_px(x, y), 5 if is_selected else 2
                )

        for i, (x, y) in enumerate(trajectory_xy):
            is_target = i == target_index
            color = (255, 255, 0) if is_target else (255, 140, 0)
            pygame.draw.circle(traj_surface, color, to_px(x, y), 5 if is_target else 3)
        pygame.draw.circle(traj_surface, (255, 255, 255), (cx, cy), 4)
        self.display.blit(traj_surface, (self.main_width + self.BEV_DISPLAY_SIZE, y0))

        pred_str = (
            COMMAND_NAMES[predicted_command].upper()
            if predicted_command is not None
            else "--"
        )

        speed_str = f"speed={speed_kmh:5.1f}"
        if target_speed_kmh is not None:
            speed_str += f"->{target_speed_kmh:4.1f}"
        speed_str += "km/h"
        hud = (
            f"cmd={COMMAND_NAMES[command].upper():8s} exec={COMMAND_NAMES[executed_command].upper():8s} "
            f"pred={pred_str:8s} "
            f"{speed_str}  steer={control.steer:+.2f}  throttle={control.throttle:.2f}  brake={control.brake:.2f}"
        )
        self.display.blit(
            self.font.render(hud, True, (255, 255, 255), (0, 0, 0)), (4, y0 + 4)
        )

        if row_label is not None:
            label_surf = self.font.render(row_label, True, (255, 255, 255), (0, 0, 0))
            self.display.blit(label_surf, (self.main_width + 4, y0 + 4))

        pygame.display.flip()
        return True

    def close(self) -> None:
        pygame.quit()
