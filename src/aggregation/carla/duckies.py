from __future__ import annotations

import math
import random

import carla
import numpy as np

from ..config import DuckieConfig

_RUBBERDUCKIE_PROP = "static.prop.rubberduckie"


def _build_transform(
    wp: carla.Waypoint, lateral_offset: float, z_offset: float, rng: random.Random
) -> carla.Transform:
    rotation = carla.Rotation(
        pitch=wp.transform.rotation.pitch,
        yaw=wp.transform.rotation.yaw + rng.uniform(0.0, 360.0),
        roll=wp.transform.rotation.roll,
    )
    transform = carla.Transform(
        carla.Location(
            x=wp.transform.location.x,
            y=wp.transform.location.y,
            z=wp.transform.location.z + z_offset,
        ),
        rotation,
    )
    if lateral_offset != 0.0:
        right = wp.transform.get_right_vector()
        transform.location.x += right.x * lateral_offset
        transform.location.y += right.y * lateral_offset
        transform.location.z += right.z * lateral_offset
    return transform


def _nearest_dist_2d(location: carla.Location, points_xy) -> float:
    if len(points_xy) == 0:
        return float("inf")
    diff = np.asarray(points_xy, dtype=np.float32) - np.array(
        [location.x, location.y], dtype=np.float32
    )
    return float(np.sqrt((diff**2).sum(axis=1)).min())


def place_duckies(
    client: carla.Client, world: carla.World, cfg: DuckieConfig, run_index: int = 0
) -> list[carla.Actor]:
    rng = random.Random(None if cfg.seed is None else cfg.seed + run_index)
    carla_map = world.get_map()
    duckie_bp = world.get_blueprint_library().find(_RUBBERDUCKIE_PROP)

    waypoints = [
        wp
        for wp in carla_map.generate_waypoints(cfg.sample_distance_m)
        if wp.lane_type == carla.LaneType.Driving
        and not (cfg.obstacle and wp.is_junction)
        and math.isfinite(wp.transform.location.x)
        and math.isfinite(wp.transform.location.y)
    ]
    road_xy = np.array(
        [(wp.transform.location.x, wp.transform.location.y) for wp in waypoints],
        dtype=np.float32,
    )
    rng.shuffle(waypoints)

    transforms: list[carla.Transform] = []
    accepted_xy: list[tuple[float, float]] = []
    for wp in waypoints:
        offset = (
            0.0
            if cfg.obstacle
            else rng.uniform(cfg.min_lateral_offset_m, cfg.max_lateral_offset_m)
        )
        transform = _build_transform(wp, offset, cfg.z_offset_m, rng)
        loc = transform.location
        clears_lanes = (
            cfg.obstacle or _nearest_dist_2d(loc, road_xy) >= cfg.min_lateral_offset_m
        )
        if (
            clears_lanes
            and _nearest_dist_2d(loc, accepted_xy) >= cfg.min_separation_m
            and rng.random() < cfg.spawn_probability
        ):
            transforms.append(transform)
            accepted_xy.append((loc.x, loc.y))

    print(
        f"[duckies] {len(transforms)} positions generated from {len(waypoints)} drivable waypoints"
    )

    responses = client.apply_batch_sync(
        [carla.command.SpawnActor(duckie_bp, t) for t in transforms],
        True,
    )
    actor_ids = [r.actor_id for r in responses if not r.error]
    failed = sum(1 for r in responses if r.error)
    if failed:
        print(f"[duckies] {failed} spawn failures (collisions / out-of-bounds)")

    spawned = list(world.get_actors(actor_ids))
    print(f"[duckies] spawned {len(spawned)} duckies")
    return spawned


def cleanup_duckies(client: carla.Client, world: carla.World) -> int:
    actors = [a for a in world.get_actors() if a.type_id == _RUBBERDUCKIE_PROP]
    if not actors:
        return 0
    responses = client.apply_batch_sync(
        [carla.command.DestroyActor(a) for a in actors], True
    )
    destroyed = sum(1 for r in responses if not r.error)
    print(f"[duckies] destroyed {destroyed} duckie props")
    return destroyed
