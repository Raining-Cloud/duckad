from __future__ import annotations

import math
from dataclasses import dataclass

from ..aggregation.command import Command


@dataclass
class ClosedLoopSummary:
    laps_completed: int
    lane_departures: int
    distance_m: float
    duration_s: float
    cte_mean_m: float
    cte_rms_m: float
    cte_max_m: float

    junctions_evaluated: int
    junctions_correct: int
    failed: bool
    failure_reason: str | None

    def as_dict(self) -> dict:
        return {
            "laps_completed": self.laps_completed,
            "lane_departures": self.lane_departures,
            "distance_m": round(self.distance_m, 3),
            "duration_s": round(self.duration_s, 3),
            "cte_mean_m": round(self.cte_mean_m, 4),
            "cte_rms_m": round(self.cte_rms_m, 4),
            "cte_max_m": round(self.cte_max_m, 4),
            "junctions_evaluated": self.junctions_evaluated,
            "junctions_correct": self.junctions_correct,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
        }


class ClosedLoopMetrics:
    def __init__(
        self,
        departure_threshold_m: float,
        abort_threshold_m: float,
        lap_radius_m: float,
        stall_timeout_s: float,
        stall_min_progress_m: float,
        turn_threshold_deg: float = 45.0,
    ):
        self._departure_threshold_m = departure_threshold_m
        self._abort_threshold_m = abort_threshold_m
        self._lap_radius_m = lap_radius_m
        self._stall_timeout_s = stall_timeout_s
        self._stall_min_progress_m = stall_min_progress_m

        self._turn_threshold_deg = turn_threshold_deg

        self._start_xy: tuple[float, float] | None = None
        self._prev_xy: tuple[float, float] | None = None
        self._distance_m = 0.0
        self._duration_s = 0.0

        self._cte_sum = 0.0
        self._cte_sq_sum = 0.0
        self._cte_max = 0.0
        self._n = 0

        self._lane_departures = 0
        self._out_of_lane = False

        self._laps = 0
        self._left_start = False

        self._distance_at_progress = 0.0
        self._time_since_progress = 0.0

        self._in_junction = False
        self._junction_entry_yaw: float | None = None
        self._junction_entry_command: int | None = None
        self._junctions_evaluated = 0
        self._junctions_correct = 0

        self._failed = False
        self._failure_reason: str | None = None

    def update(
        self,
        x: float,
        y: float,
        lane_center: tuple[float, float] | None,
        dt: float,
        is_junction: bool = False,
        yaw_deg: float | None = None,
        command: int | None = None,
    ) -> None:
        if self._failed:
            return

        if self._start_xy is None:
            self._start_xy = (x, y)
        if self._prev_xy is not None:
            self._distance_m += math.dist((x, y), self._prev_xy)
        self._prev_xy = (x, y)
        self._duration_s += dt

        cte = None if lane_center is None else math.dist((x, y), lane_center)
        if cte is not None:
            self._cte_sum += cte
            self._cte_sq_sum += cte * cte
            self._cte_max = max(self._cte_max, cte)
            self._n += 1

        self._update_laps(x, y)
        self._update_junction(is_junction, yaw_deg, command)

        out_of_lane = cte is None or cte > self._departure_threshold_m
        if out_of_lane and not self._out_of_lane:
            self._lane_departures += 1
        self._out_of_lane = out_of_lane

        if cte is not None and cte > self._abort_threshold_m:
            self._fail("off_route")
            return

        if self._distance_m - self._distance_at_progress >= self._stall_min_progress_m:
            self._distance_at_progress = self._distance_m
            self._time_since_progress = 0.0
        else:
            self._time_since_progress += dt
            if self._time_since_progress > self._stall_timeout_s:
                self._fail("stalled")

    def _update_laps(self, x: float, y: float) -> None:
        d_start = math.dist((x, y), self._start_xy)
        if d_start > self._lap_radius_m:
            self._left_start = True
        elif self._left_start and d_start <= self._lap_radius_m:
            self._laps += 1
            self._left_start = False

    def _update_junction(
        self, is_junction: bool, yaw_deg: float | None, command: int | None
    ) -> None:
        if yaw_deg is None:
            return
        if is_junction and not self._in_junction:
            self._in_junction = True
            self._junction_entry_yaw = yaw_deg
            self._junction_entry_command = command
        elif not is_junction and self._in_junction:
            self._in_junction = False
            entry_command = self._junction_entry_command
            if entry_command in (int(Command.LEFT), int(Command.RIGHT)):
                delta = (yaw_deg - self._junction_entry_yaw + 180.0) % 360.0 - 180.0
                if delta >= self._turn_threshold_deg:
                    executed = int(Command.RIGHT)
                elif delta <= -self._turn_threshold_deg:
                    executed = int(Command.LEFT)
                else:
                    executed = int(Command.STRAIGHT)
                self._junctions_evaluated += 1
                if executed == entry_command:
                    self._junctions_correct += 1

    def _fail(self, reason: str) -> None:
        self._failed = True
        self._failure_reason = reason

    def failed(self) -> bool:
        return self._failed

    @property
    def laps_completed(self) -> int:
        return self._laps

    @property
    def junctions_evaluated(self) -> int:
        return self._junctions_evaluated

    def summary(self) -> ClosedLoopSummary:
        mean = self._cte_sum / self._n if self._n else 0.0
        rms = math.sqrt(self._cte_sq_sum / self._n) if self._n else 0.0
        return ClosedLoopSummary(
            laps_completed=self._laps,
            lane_departures=self._lane_departures,
            distance_m=self._distance_m,
            duration_s=self._duration_s,
            junctions_evaluated=self._junctions_evaluated,
            junctions_correct=self._junctions_correct,
            cte_mean_m=mean,
            cte_rms_m=rms,
            cte_max_m=self._cte_max,
            failed=self._failed,
            failure_reason=self._failure_reason,
        )
