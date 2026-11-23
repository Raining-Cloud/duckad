from __future__ import annotations

import math
from collections import deque

import carla
import numpy as np

from .config import ControlLimits, PIDGains


def temporal_segment_speeds_kmh(
    temporal_xy: np.ndarray, temporal_dt: float
) -> np.ndarray:
    pts = np.vstack(([0.0, 0.0], temporal_xy))
    return np.linalg.norm(np.diff(pts, axis=0), axis=1) / temporal_dt * 3.6


def lookahead_temporal_index(
    spatial_xy: np.ndarray, temporal_xy: np.ndarray, lookahead_index: int
) -> int:
    target = spatial_xy[min(lookahead_index, len(spatial_xy) - 1)]
    return int(np.argmin(np.linalg.norm(temporal_xy - target, axis=1)))


class StallAwareSpeedSelector:
    def __init__(
        self,
        lookahead_index: int,
        temporal_dt: float,
        dt: float,
        stall_timeout_s: float,
        stall_speed_eps_kmh: float,
        push_interval_s: float,
        creep_step_kmh: float,
        creep_max_kmh: float,
    ):
        self._lookahead_index = lookahead_index
        self._temporal_dt = temporal_dt
        self._dt = dt
        self._stall_timeout_s = stall_timeout_s
        self._stall_speed_eps_kmh = stall_speed_eps_kmh
        self._push_interval_s = push_interval_s
        self._creep_step_kmh = creep_step_kmh
        self._creep_max_kmh = creep_max_kmh
        self._stall_time_s = 0.0

    def select(
        self, spatial_xy: np.ndarray, temporal_xy: np.ndarray, measured_speed_kmh: float
    ) -> tuple[float, int]:
        seg_speeds_kmh = temporal_segment_speeds_kmh(temporal_xy, self._temporal_dt)
        base_index = lookahead_temporal_index(
            spatial_xy, temporal_xy, self._lookahead_index
        )
        last_index = len(temporal_xy) - 1

        if measured_speed_kmh > self._stall_speed_eps_kmh:
            self._stall_time_s = 0.0
        else:
            self._stall_time_s += self._dt

        stalled_for_s = self._stall_time_s - self._stall_timeout_s
        steps = (
            0
            if stalled_for_s <= 0.0
            else int(stalled_for_s / self._push_interval_s) + 1
        )
        index = min(base_index + steps, last_index)
        target_kmh = float(seg_speeds_kmh[index])

        overflow_steps = max(0, (base_index + steps) - last_index)
        if overflow_steps > 0:
            creep_kmh = min(overflow_steps * self._creep_step_kmh, self._creep_max_kmh)
            target_kmh = max(target_kmh, creep_kmh)
        return target_kmh, index


class _PID:
    def __init__(self, gains: PIDGains, dt: float, buffer_len: int = 10):
        self._gains = gains
        self._dt = dt
        self._errors: deque[float] = deque(maxlen=buffer_len)

    def _update(self, error: float) -> float:
        self._errors.append(error)
        derivative = (
            (self._errors[-1] - self._errors[-2]) / self._dt
            if len(self._errors) >= 2
            else 0.0
        )
        integral = sum(self._errors) * self._dt
        return (
            self._gains.k_p * error
            + self._gains.k_d * derivative
            + self._gains.k_i * integral
        )


class LongitudinalPID(_PID):
    def step(self, current_speed_kmh: float, target_speed_kmh: float) -> float:
        return float(
            np.clip(self._update(target_speed_kmh - current_speed_kmh), -1.0, 1.0)
        )


class LateralPID(_PID):
    def step(self, target_local_xy: tuple[float, float]) -> float:
        x, y = target_local_xy
        heading_error = math.atan2(y, x)
        return float(np.clip(self._update(heading_error), -1.0, 1.0))


class TrajectoryPIDController:
    def __init__(
        self,
        lateral_gains: PIDGains,
        longitudinal_gains: PIDGains,
        dt: float,
        limits: ControlLimits,
    ):
        self._lateral = LateralPID(lateral_gains, dt)
        self._longitudinal = LongitudinalPID(longitudinal_gains, dt)
        self._limits = limits
        self._past_steer = 0.0

    def step(
        self,
        current_speed_kmh: float,
        target_speed_kmh: float,
        target_local_xy: tuple[float, float],
    ) -> carla.VehicleControl:
        accel = self._longitudinal.step(current_speed_kmh, target_speed_kmh)
        steer = self._lateral.step(target_local_xy)

        delta = self._limits.max_steer_delta
        steer = max(self._past_steer - delta, min(self._past_steer + delta, steer))
        steer = float(np.clip(steer, -self._limits.max_steer, self._limits.max_steer))
        self._past_steer = steer

        control = carla.VehicleControl()
        control.steer = steer
        if accel >= 0.0:
            control.throttle = min(accel, self._limits.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel, self._limits.max_brake)
        return control
