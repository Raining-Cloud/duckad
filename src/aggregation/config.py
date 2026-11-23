from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

IMAGENET_DIR = "imagenet-val"  # Background augmentation images.
CACHE_SIZE = 1000  # Images kept in memory.


@dataclass
class TransformConfig:
    """Pose relative to the parent (vehicle origin), in CARLA's left-handed meters/degrees."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class VehicleConfig:
    blueprint_id: str = "vehicle.duckietown.duckiebot"
    # Keep the ego chassis out of recorded camera images.
    hide_ego_mesh: bool = True
    max_speed_kmh: float = 18


@dataclass
class CameraRigConfig:
    """Front fisheye camera used by the center and recovery rigs."""

    name: str = "front"
    role_name: str = "front"
    transform: TransformConfig = field(
        default_factory=lambda: TransformConfig(x=1.62, y=0.0, z=2.13, pitch=-17.0)
    )
    image_size_x: int = 640  # VGA
    image_size_y: int = 480

    camera_model: str = "equidistant"
    diagonal_fov_deg: float = 145.0

    @property
    def fov_deg(self) -> float:
        """Horizontal FOV derived from the diagonal equidistant FOV."""
        half_diag_rad = math.radians(self.diagonal_fov_deg / 2)
        r_diag = math.hypot(self.image_size_x / 2, self.image_size_y / 2)
        focal_px = r_diag / half_diag_rad
        return 2 * math.degrees((self.image_size_x / 2) / focal_px)


_SIDE_CAMERA_OFFSET_M = 1.4
_SIDE_CAMERA_JITTER_M = 0.1


def _side_rig(name: str, base: TransformConfig, y: float) -> CameraRigConfig:
    return CameraRigConfig(name=name, role_name=name, transform=replace(base, y=y))


def _default_rigs() -> list[CameraRigConfig]:
    center = CameraRigConfig()
    left_y = -_SIDE_CAMERA_OFFSET_M + random.uniform(
        -_SIDE_CAMERA_JITTER_M, _SIDE_CAMERA_JITTER_M
    )
    right_y = _SIDE_CAMERA_OFFSET_M + random.uniform(
        -_SIDE_CAMERA_JITTER_M, _SIDE_CAMERA_JITTER_M
    )
    return [
        center,
        _side_rig("left", center.transform, left_y),
        _side_rig("right", center.transform, right_y),
    ]


@dataclass
class MainCameraConfig:
    rigs: list[CameraRigConfig] = field(default_factory=_default_rigs)


@dataclass
class TargetImageConfig:
    """Output format expected by the MobileNetV2 head."""

    size: int = 224
    pad_color: int = 0


@dataclass
class BEVConfig:
    """Top-down semantic camera covering the road ahead."""

    image_size: int = 64
    transform: TransformConfig = field(
        default_factory=lambda: TransformConfig(x=8.5, z=60.0, pitch=-90.0)
    )
    size_m: float = 12.1  # Square ground footprint in meters.

    @property
    def fov_deg(self) -> float:
        """FOV needed to cover `size_m` from the configured height."""
        return 2 * math.degrees(math.atan(self.size_m / (2 * self.transform.z)))


def bev_config_for_rig(bev: BEVConfig, rig: CameraRigConfig) -> BEVConfig:
    """Align the BEV camera with a front-camera rig."""
    return replace(
        bev, transform=replace(bev.transform, y=bev.transform.y + rig.transform.y)
    )


@dataclass
class TrajectoryConfig:
    num_points: int = 10
    spacing_m: float = 1.0
    start_offset_m: float = 1.0

    # Time-spaced future positions encode the speed profile.
    temporal_num_points: int = 10
    temporal_hz: float = 3.33  # Hz

    def temporal_stride_ticks(self, fixed_delta_seconds: float) -> int:
        """Simulation ticks between temporal points."""
        stride = round((1.0 / self.temporal_hz) / fixed_delta_seconds)
        if stride < 1:
            raise ValueError(
                f"temporal_hz={self.temporal_hz} exceeds the sim rate "
                f"({1 / fixed_delta_seconds:g} Hz); the temporal stride rounds to <1 tick"
            )
        return stride

    def temporal_dt(self, fixed_delta_seconds: float) -> float:
        """Actual seconds between temporal points."""
        return self.temporal_stride_ticks(fixed_delta_seconds) * fixed_delta_seconds

    def temporal_horizon_ticks(self, fixed_delta_seconds: float) -> int:
        """Future ticks required to complete a temporal trajectory."""
        return self.temporal_num_points * self.temporal_stride_ticks(
            fixed_delta_seconds
        )


@dataclass
class DuckieConfig:
    """Placement settings for decorative or obstacle duckies."""

    enabled: bool = False
    obstacle: bool = False
    min_separation_m: float = 2
    min_lateral_offset_m: float = 0.0
    max_lateral_offset_m: float = 0.0
    z_offset_m: float = 0.05
    sample_distance_m: float = 0.25
    spawn_probability: float = 1.0
    seed: int | None = None
    wait_timeout_s: float = 4.0


@dataclass
class StorageConfig:
    out_dir: str = "data"
    batch_size: int = 256  # Samples per NPZ shard.


@dataclass
class PipelineConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    main_camera: MainCameraConfig = field(default_factory=MainCameraConfig)
    target_image: TargetImageConfig = field(default_factory=TargetImageConfig)
    bev: BEVConfig = field(default_factory=BEVConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    duckies: DuckieConfig = field(default_factory=DuckieConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    fixed_delta_seconds: float = 0.05
    host: str = "127.0.0.1"
    port: int = 2000
    map_name: str | None = None
    hdri_preset: str | None = "airport"
    junction_safe_distance_m: float | None = None
