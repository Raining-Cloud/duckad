from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import carla
import numpy as np

from ..aggregation.command import Command
from .config import EvalConfig
from .controller import StallAwareSpeedSelector, TrajectoryPIDController
from .inference import TrajectoryPredictor

if TYPE_CHECKING:
    from ..aggregation.carla.duckie_agent import AgentDriver
    from ..aggregation.carla.world import SimHandle
    from ..aggregation.config import PipelineConfig


@dataclass
class DriveStep:
    control: carla.VehicleControl
    path_local: np.ndarray
    target_index: int
    target_speed_kmh: float
    executed_command: int
    predicted_command: int | None
    bev_idx: np.ndarray | None
    temporal_local: np.ndarray | None
    selected_temporal_index: int | None


class DrivingAgent:
    def __init__(
        self,
        predictor: TrajectoryPredictor,
        controller: TrajectoryPIDController,
        speed_selector: StallAwareSpeedSelector,
        cfg: EvalConfig,
        ticks_per_inference: int,
        rng: random.Random | None = None,
    ):
        self._predictor = predictor
        self._controller = controller
        self._speed_selector = speed_selector
        self._cfg = cfg
        self._ticks_per_inference = ticks_per_inference
        self._rng = rng or random.Random()

        self._steer_noise_std = cfg.steer_noise_std

        self._path_local: np.ndarray | None = None
        self._temporal_local: np.ndarray | None = None
        self._bev_idx: np.ndarray | None = None
        self._predicted_command: int | None = None
        self._executed_command = int(Command.DEFAULT)
        self._selected_temporal_index: int | None = None
        self._target_speed_kmh = cfg.target_speed_kmh

    @property
    def steer_noise_std(self) -> float:
        return self._steer_noise_std

    @steer_noise_std.setter
    def steer_noise_std(self, std: float) -> None:
        self._steer_noise_std = std

    def step(
        self, tick_i: int, main_bgr: np.ndarray, command: int, speed_kmh: float
    ) -> DriveStep:
        if tick_i % self._ticks_per_inference == 0:
            (
                self._path_local,
                self._temporal_local,
                self._bev_idx,
                self._predicted_command,
            ) = self._predictor.predict(main_bgr, command)
            self._executed_command = command
            if self._temporal_local is not None:
                model_speed_kmh, self._selected_temporal_index = (
                    self._speed_selector.select(
                        self._path_local, self._temporal_local, speed_kmh
                    )
                )
                if self._cfg.use_model_speed:
                    self._target_speed_kmh = min(
                        self._cfg.max_speed_kmh, model_speed_kmh
                    )

        target_index = min(self._cfg.lookahead_index, len(self._path_local) - 1)
        control = self._controller.step(
            speed_kmh, self._target_speed_kmh, tuple(self._path_local[target_index])
        )

        control.steer = max(
            -1.0, min(1.0, control.steer + self._rng.gauss(0.0, self._steer_noise_std))
        )
        return DriveStep(
            control=control,
            path_local=self._path_local,
            target_index=target_index,
            target_speed_kmh=self._target_speed_kmh,
            executed_command=self._executed_command,
            predicted_command=self._predicted_command,
            bev_idx=self._bev_idx,
            temporal_local=self._temporal_local,
            selected_temporal_index=self._selected_temporal_index,
        )


def build_agent(
    predictor: TrajectoryPredictor,
    cfg: EvalConfig,
    fixed_delta_seconds: float,
    rng: random.Random | None = None,
) -> tuple[DrivingAgent, int]:
    from ..aggregation.config import TrajectoryConfig

    ticks_per_inference = max(1, round(1.0 / (cfg.inference_hz * fixed_delta_seconds)))

    longitudinal_gains = cfg.speed_pid if cfg.use_model_speed else cfg.longitudinal_pid
    controller = TrajectoryPIDController(
        cfg.lateral_pid, longitudinal_gains, fixed_delta_seconds, cfg.limits
    )

    temporal_dt = getattr(
        predictor, "temporal_dt", None
    ) or TrajectoryConfig().temporal_dt(fixed_delta_seconds)
    inference_period = ticks_per_inference * fixed_delta_seconds
    speed_selector = StallAwareSpeedSelector(
        cfg.lookahead_index,
        temporal_dt,
        inference_period,
        cfg.stall_timeout_s,
        cfg.stall_speed_eps_kmh,
        cfg.stall_push_interval_s,
        cfg.stall_creep_step_kmh,
        cfg.stall_creep_max_kmh,
    )

    agent = DrivingAgent(
        predictor, controller, speed_selector, cfg, ticks_per_inference, rng
    )
    return agent, ticks_per_inference


def build_traffic_driver(
    handle: SimHandle,
    cfg: PipelineConfig,
    count: int,
    rng: random.Random | None = None,
) -> AgentDriver | None:
    if count <= 0:
        return None

    from ..aggregation.carla.duckie_agent import AgentDriver, build_global_route_planner
    from ..aggregation.carla.world import spawn_traffic

    carla_map = handle.world.get_map()
    driver = AgentDriver(
        carla_map,
        {},
        grp=build_global_route_planner(carla_map),
        rng=rng,
    )
    for vehicle in spawn_traffic(handle, cfg, count):
        driver.add_traffic(vehicle, cfg.vehicle.max_speed_kmh)
    return driver
