from __future__ import annotations

import carla
import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import TrajectoryConfig

_WAYPOINT_STEP_M = 0.5


def _nearest_index(waypoints: list[carla.Waypoint], location: carla.Location) -> int:
    distances = [location.distance(wp.transform.location) for wp in waypoints]
    return int(np.argmin(distances))


def trajectory_horizon_m(cfg: TrajectoryConfig) -> float:
    return cfg.start_offset_m + (cfg.num_points - 1) * cfg.spacing_m


def sample_future_trajectory(
    carla_map: carla.Map,
    vehicle_transform: carla.Transform,
    cfg: TrajectoryConfig,
    reference_transform: carla.Transform | None = None,
    anchor_waypoints: list[carla.Waypoint] | None = None,
) -> np.ndarray:
    reference_transform = reference_transform or vehicle_transform
    total_needed_m = trajectory_horizon_m(cfg)
    arc_length_to_collect = total_needed_m * 1.5 + 2.0

    remaining_anchors: list[carla.Waypoint] | None = None
    if anchor_waypoints:
        start_index = _nearest_index(anchor_waypoints, vehicle_transform.location)
        remaining_anchors = anchor_waypoints[start_index:]

    waypoint = (
        remaining_anchors[0]
        if remaining_anchors
        else carla_map.get_waypoint(vehicle_transform.location, project_to_road=True)
    )
    arc_lengths = [0.0]
    xs = [waypoint.transform.location.x]
    ys = [waypoint.transform.location.y]
    walked = 0.0
    anchor_i = 1
    while walked < arc_length_to_collect:
        if remaining_anchors is not None and anchor_i < len(remaining_anchors):
            next_waypoint = remaining_anchors[anchor_i]
            anchor_i += 1
            step = waypoint.transform.location.distance(
                next_waypoint.transform.location
            )
            waypoint = next_waypoint
        else:
            candidates = waypoint.next(_WAYPOINT_STEP_M)
            if not candidates:
                break
            waypoint = candidates[0]
            step = _WAYPOINT_STEP_M
        walked += step
        arc_lengths.append(walked)
        xs.append(waypoint.transform.location.x)
        ys.append(waypoint.transform.location.y)

    arc_lengths = np.asarray(arc_lengths, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    finite = np.isfinite(arc_lengths) & np.isfinite(xs) & np.isfinite(ys)
    arc_lengths, xs, ys = arc_lengths[finite], xs[finite], ys[finite]
    if arc_lengths.size:
        strictly_increasing = np.concatenate([[True], np.diff(arc_lengths) > 1e-6])
        arc_lengths, xs, ys = (
            arc_lengths[strictly_increasing],
            xs[strictly_increasing],
            ys[strictly_increasing],
        )

    if arc_lengths.size < 2:
        if arc_lengths.size >= 1:
            local = reference_transform.inverse_transform(
                carla.Location(float(xs[0]), float(ys[0]), 0.0)
            )
            return np.tile(
                np.array([local.x, local.y], dtype=np.float32), (cfg.num_points, 1)
            )
        points = np.zeros((cfg.num_points, 2), dtype=np.float32)
        points[:, 0] = cfg.start_offset_m + np.arange(cfg.num_points) * cfg.spacing_m
        return points

    spline_x = PchipInterpolator(arc_lengths, xs)
    spline_y = PchipInterpolator(arc_lengths, ys)

    s_dense = np.arange(arc_lengths[0], arc_lengths[-1], 0.01)
    x_dense, y_dense = spline_x(s_dense), spline_y(s_dense)
    seg_lengths = np.hypot(np.diff(x_dense), np.diff(y_dense))
    cum_l2 = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    target_l2 = cfg.start_offset_m + np.arange(cfg.num_points) * cfg.spacing_m
    x_world = np.interp(target_l2, cum_l2, x_dense)
    y_world = np.interp(target_l2, cum_l2, y_dense)

    points = np.empty((cfg.num_points, 2), dtype=np.float32)
    for i in range(cfg.num_points):
        local = reference_transform.inverse_transform(
            carla.Location(float(x_world[i]), float(y_world[i]), 0.0)
        )
        points[i] = (local.x, local.y)
    return points


def trajectory_turn_and_lateral(
    trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    directions = np.diff(trajectory, axis=1)
    headings = np.arctan2(directions[:, :, 1], directions[:, :, 0])
    wrapped_delta = np.angle(np.exp(1j * (headings[:, -1] - headings[:, 0])))
    turn_deg = np.degrees(wrapped_delta)
    lateral_m = trajectory[:, -1, 1]
    return turn_deg, lateral_m
