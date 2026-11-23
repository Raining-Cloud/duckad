from __future__ import annotations

import carla
import numpy as np
import pygame

from ..command import COMMAND_COLORS_BGR, Command
from ..imaging import DUCKIETOWN_LABELS

_BACKGROUND_COLOR_BGR = (30, 30, 30)
_CLASS_COLORS_BGR = {
    "center_lane": (0, 215, 255),
    "side_lane": (255, 255, 255),
    "asphalt": (60, 60, 60),
    "stop_lane": (30, 30, 200),
    "sign": (255, 120, 60),
    "bot": (0, 140, 255),
    "duck": (255, 0, 255),
}


def colorize_duckietown_labels(label_img: np.ndarray) -> np.ndarray:
    ids = label_img[0] if label_img.ndim == 3 else label_img
    colors = np.full((*ids.shape, 3), _BACKGROUND_COLOR_BGR, dtype=np.uint8)
    for name, tag_id in DUCKIETOWN_LABELS.items():
        colors[ids == tag_id] = _CLASS_COLORS_BGR[name]
    return colors


def draw_rig_markers(
    world: carla.World,
    vehicle_transform: carla.Transform,
    main_camera_transform: carla.Transform,
    bev_camera_transform: carla.Transform,
    bev_size_m: float,
    life_time: float,
) -> None:
    debug = world.debug
    debug.draw_point(
        main_camera_transform.location,
        size=0.12,
        color=carla.Color(0, 255, 0),
        life_time=life_time,
    )
    debug.draw_point(
        bev_camera_transform.location,
        size=0.12,
        color=carla.Color(30, 120, 255),
        life_time=life_time,
    )

    ground_point = carla.Location(
        bev_camera_transform.location.x,
        bev_camera_transform.location.y,
        vehicle_transform.location.z,
    )
    debug.draw_point(
        ground_point, size=0.12, color=carla.Color(0, 255, 255), life_time=life_time
    )

    forward = vehicle_transform.get_forward_vector()
    right = vehicle_transform.get_right_vector()
    half = bev_size_m / 2.0
    corners = [
        ground_point + forward * half + right * half,
        ground_point + forward * half - right * half,
        ground_point - forward * half - right * half,
        ground_point - forward * half + right * half,
    ]
    rect_color = carla.Color(0, 255, 255)
    for i in range(4):
        debug.draw_line(
            corners[i],
            corners[(i + 1) % 4],
            thickness=0.05,
            color=rect_color,
            life_time=life_time,
        )


def draw_trajectory_markers(
    world: carla.World,
    vehicle_transform: carla.Transform,
    trajectory_world: list[carla.Location],
    life_time: float,
) -> None:
    debug = world.debug
    debug.draw_point(
        vehicle_transform.location,
        size=0.1,
        color=carla.Color(255, 255, 255),
        life_time=life_time,
    )
    for location in trajectory_world:
        debug.draw_point(
            location, size=0.08, color=carla.Color(255, 140, 0), life_time=life_time
        )


class PygameViewer:
    BEV_DISPLAY_SIZE = 256
    TRAJ_PLOT_SIZE = 256
    ROW_ORDER = ("left", "front", "right")

    def __init__(
        self,
        main_size: int,
        main_camera_xy: tuple[float, float],
        bev_xy: tuple[float, float],
        bev_size_m: float,
    ):
        pygame.init()
        self.main_size = main_size
        self.main_camera_xy = main_camera_xy
        self.bev_xy = bev_xy
        self.bev_size_m = bev_size_m
        self.traj_range_m = bev_size_m * 0.8
        self.row_height = max(main_size, self.BEV_DISPLAY_SIZE, self.TRAJ_PLOT_SIZE)
        width = main_size + self.BEV_DISPLAY_SIZE + self.TRAJ_PLOT_SIZE
        height = self.row_height * len(self.ROW_ORDER)
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption("aggregation preview")

    TEMPORAL_COLOR_BGR = (0, 255, 255)

    def render(self, rig_frames: dict[str, tuple]) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        for row, name in enumerate(self.ROW_ORDER):
            if name in rig_frames:
                self._render_row(*rig_frames[name], y_offset=row * self.row_height)

        pygame.display.flip()
        return True

    def _render_row(
        self,
        main_img_bgr: np.ndarray,
        bev_labels: np.ndarray,
        trajectory_xy: np.ndarray,
        command: int,
        temporal_xy: np.ndarray | None = None,
        *,
        y_offset: int,
    ) -> None:
        main_rgb = main_img_bgr[:, :, ::-1]
        self.display.blit(
            pygame.surfarray.make_surface(main_rgb.swapaxes(0, 1)), (0, y_offset)
        )

        bev_bgr = colorize_duckietown_labels(bev_labels)
        bev_rgb = bev_bgr[:, :, ::-1]
        bev_surface = pygame.surfarray.make_surface(bev_rgb.swapaxes(0, 1))
        bev_surface = pygame.transform.scale(
            bev_surface, (self.BEV_DISPLAY_SIZE, self.BEV_DISPLAY_SIZE)
        )
        self.display.blit(bev_surface, (self.main_size, y_offset))

        traj_surface = pygame.Surface((self.TRAJ_PLOT_SIZE, self.TRAJ_PLOT_SIZE))
        traj_surface.fill((20, 20, 20))
        scale = self.TRAJ_PLOT_SIZE / (2 * self.traj_range_m)
        cx, cy = self.TRAJ_PLOT_SIZE // 2, self.TRAJ_PLOT_SIZE - 15

        def to_px(x: float, y: float) -> tuple[int, int]:
            return int(cx + y * scale), int(cy - x * scale)

        bev_x, bev_y = self.bev_xy
        half = self.bev_size_m / 2.0
        rect_corners = [
            to_px(bev_x + half, bev_y + half),
            to_px(bev_x + half, bev_y - half),
            to_px(bev_x - half, bev_y - half),
            to_px(bev_x - half, bev_y + half),
        ]
        pygame.draw.polygon(traj_surface, (0, 200, 200), rect_corners, width=2)
        pygame.draw.circle(traj_surface, (30, 120, 255), to_px(bev_x, bev_y), 4)

        pygame.draw.circle(traj_surface, (0, 255, 0), to_px(*self.main_camera_xy), 4)

        dot_color = COMMAND_COLORS_BGR[Command(command)][::-1]
        for x, y in trajectory_xy:
            pygame.draw.circle(traj_surface, dot_color, to_px(x, y), 3)

        if temporal_xy is not None:
            temporal_rgb = self.TEMPORAL_COLOR_BGR[::-1]
            prev = None
            for x, y in temporal_xy:
                point = to_px(x, y)
                if prev is not None:
                    pygame.draw.line(traj_surface, temporal_rgb, prev, point, 1)
                pygame.draw.circle(traj_surface, temporal_rgb, point, 2)
                prev = point

        pygame.draw.circle(traj_surface, (255, 255, 255), (cx, cy), 4)
        self.display.blit(
            traj_surface, (self.main_size + self.BEV_DISPLAY_SIZE, y_offset)
        )

    def close(self) -> None:
        pygame.quit()
