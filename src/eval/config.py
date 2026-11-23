from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PIDGains:
    k_p: float
    k_i: float
    k_d: float


@dataclass
class ControlLimits:
    max_throttle: float = 0.75
    max_brake: float = 0.3
    max_steer: float = 0.8
    max_steer_delta: float = 0.1


@dataclass
class EvalConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    map_name: str | None = "duckietown_02"
    hdri_preset: str | None = "airport"

    checkpoint: str = ""
    device: str = "cuda"

    fixed_delta_seconds: float = 0.05
    duration_s: float = 60.0

    inference_hz: float = 3.0
    lookahead_index: int = 2

    use_model_speed: bool = True
    target_speed_kmh: float = 10.0
    max_speed_kmh: float = 15.0

    stall_timeout_s: float = 2.0
    stall_speed_eps_kmh: float = 0.5
    stall_push_interval_s: float = 0.5

    stall_creep_step_kmh: float = 4.0
    stall_creep_max_kmh: float = 4.0

    lateral_pid: PIDGains = field(
        default_factory=lambda: PIDGains(k_p=1.5, k_i=0.05, k_d=0.2)
    )
    longitudinal_pid: PIDGains = field(
        default_factory=lambda: PIDGains(k_p=1.0, k_i=0.05, k_d=0.0)
    )

    speed_pid: PIDGains = field(
        default_factory=lambda: PIDGains(k_p=1.0, k_i=0.02, k_d=0.1)
    )
    limits: ControlLimits = field(default_factory=ControlLimits)

    steer_noise_std: float = 0.0

    steer_noise_step: float = 0.0

    junction: bool = False
    max_junctions: int = 0
    departure_threshold_m: float = 0.35
    abort_threshold_m: float = 3.0
    progress_stall_timeout_s: float = 8.0
    progress_stall_min_m: float = 0.5
    lap_radius_m: float = 3.0
    max_duration_s: float = 600.0

    traffic_count: int = 0

    duckies: bool = False
    duckie_obstacle: bool = False
    duckie_min_separation_m: float = 20.0
    duckie_lateral_offset: tuple[float, float] = (3.0, 5.0)

    visualize: bool = False
