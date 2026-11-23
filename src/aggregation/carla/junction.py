from __future__ import annotations

import random

import carla

from ..command import Command

_STEP_M = 1.0
_YAW_TURN_THRESHOLD_DEG = 25.0


def _walk_to_junction(
    start_wp: carla.Waypoint, max_lookahead_m: float
) -> tuple[list[carla.Waypoint], carla.Waypoint] | None:
    wp = start_wp
    approach_path: list[carla.Waypoint] = []
    walked = 0.0
    while walked < max_lookahead_m:
        next_wps = wp.next(_STEP_M)
        if not next_wps:
            return None
        if any(n.is_junction for n in next_wps):
            return approach_path, wp
        wp = next_wps[0]
        approach_path.append(wp)
        walked += _STEP_M
    return None


def available_junction_actions(
    start_wp: carla.Waypoint, max_lookahead_m: float
) -> tuple[dict[Command, list[carla.Waypoint]], int]:
    found = _walk_to_junction(start_wp, max_lookahead_m)
    if found is None:
        return {}, 0
    approach_path, entrance_wp = found
    entrance_index = len(approach_path)
    start_yaw = entrance_wp.transform.rotation.yaw

    actions: dict[Command, list[carla.Waypoint]] = {}
    queue: list[tuple[carla.Waypoint, bool, list[carla.Waypoint]]] = [
        (entrance_wp, False, [])
    ]
    visited: set[tuple[int, int, float]] = set()
    steps = 0

    while queue and steps < 200:
        current_wp, entered, path = queue.pop(0)
        steps += 1

        wp_id = (current_wp.road_id, current_wp.lane_id, round(current_wp.s, 1))
        if wp_id in visited:
            continue
        visited.add(wp_id)

        if not entered and current_wp.is_junction:
            entered = True

        if entered and not current_wp.is_junction:
            exit_yaw = current_wp.transform.rotation.yaw
            diff = (exit_yaw - start_yaw) % 360
            if diff > 180:
                diff -= 360

            if diff < -_YAW_TURN_THRESHOLD_DEG:
                command = Command.LEFT
            elif diff > _YAW_TURN_THRESHOLD_DEG:
                command = Command.RIGHT
            else:
                command = Command.STRAIGHT

            if command not in actions:
                ext_path = list(path)
                ext_wp = current_wp
                for _ in range(15):
                    next_wps = ext_wp.next(_STEP_M)
                    if not next_wps:
                        break
                    ext_wp = next_wps[0]
                    ext_path.append(ext_wp)
                actions[command] = approach_path + ext_path
            continue

        for next_wp in current_wp.next(_STEP_M):
            queue.append((next_wp, entered, path + [next_wp]))

    return actions, entrance_index


def junction_spawn_transforms(
    carla_map: carla.Map,
    safe_distance_m: float,
    jitter_m: float = 0.0,
    rng: random.Random | None = None,
    points_per_lane: int = 1,
    min_distance_m: float | None = None,
) -> list[carla.Transform]:
    rng = rng or random.Random()
    candidates = [
        wp
        for wp in carla_map.generate_waypoints(5.0)
        if not wp.is_junction and wp.lane_type == carla.LaneType.Driving
    ]

    entrances: dict[tuple[int, int], carla.Waypoint] = {}
    for wp in candidates:
        found = _walk_to_junction(wp, 100.0)
        if found is None:
            continue
        _, entrance_wp = found
        entrances[(entrance_wp.road_id, entrance_wp.lane_id)] = entrance_wp

    transforms: list[carla.Transform] = []
    lo = safe_distance_m if min_distance_m is None else min_distance_m
    hi = safe_distance_m + jitter_m
    for entrance_wp in entrances.values():
        for _ in range(points_per_lane):
            distance = rng.uniform(lo, hi) if hi > lo else lo
            back = entrance_wp.previous(distance)
            if not back or back[0].is_junction:
                continue
            transform = back[0].transform
            transform.location.z += 0.3
            transforms.append(transform)

    return transforms
