from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import carla

from ..config import PipelineConfig
from .junction import junction_spawn_transforms


@dataclass
class SimHandle:
    client: carla.Client
    world: carla.World
    vehicle: carla.Vehicle
    original_settings: carla.WorldSettings

    spawn_points: list[carla.Transform]

    traffic_spawn_points: list[carla.Transform]

    traffic: list[carla.Vehicle] = field(default_factory=list)


def _apply_hdri(world: carla.World, cfg: PipelineConfig, presets: list[str]) -> None:
    if cfg.hdri_preset is not None:
        if cfg.hdri_preset not in presets:
            raise RuntimeError(
                f"HDRI preset {cfg.hdri_preset!r} not available on this map; options are {presets}"
            )
        world.set_hdri_preset(cfg.hdri_preset)


def _hdri_presets(world: carla.World) -> list[str]:
    try:
        return list(world.get_hdri_presets())
    except RuntimeError:
        return []


def _resolve_spawn_points(
    world: carla.World, cfg: PipelineConfig
) -> list[carla.Transform]:
    if cfg.junction_safe_distance_m is None:
        return world.get_map().get_spawn_points()
    spawn_points = junction_spawn_transforms(
        world.get_map(), cfg.junction_safe_distance_m
    )
    if not spawn_points:
        raise RuntimeError(
            f"No junction-approach spawn points found at safe distance {cfg.junction_safe_distance_m}m on this map"
        )
    return spawn_points


def _spawn_vehicle(
    world: carla.World,
    cfg: PipelineConfig,
    spawn_points: list[carla.Transform],
    hidden: bool = False,
) -> carla.Vehicle:
    blueprint = world.get_blueprint_library().find(cfg.vehicle.blueprint_id)
    vehicle = None
    for spawn_point in random.sample(spawn_points, len(spawn_points)):
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            break
    if vehicle is None:
        raise RuntimeError("Could not spawn duckiebot at any spawn point")

    if hidden:
        vehicle.set_hidden_in_game(True)

    return vehicle


def _settle(world: carla.World, cfg: PipelineConfig) -> None:
    settle_ticks = round(2.0 / cfg.fixed_delta_seconds)
    for _ in range(settle_ticks):
        world.tick()


def connect_and_spawn(cfg: PipelineConfig, hide_ego: bool = False) -> SimHandle:
    client = carla.Client(cfg.host, cfg.port)

    client.set_timeout(60.0)
    world = client.load_world(cfg.map_name) if cfg.map_name else client.get_world()

    hdri_presets = _hdri_presets(world) if cfg.hdri_preset is not None else []
    _apply_hdri(world, cfg, hdri_presets)

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.fixed_delta_seconds
    world.apply_settings(settings)

    spawn_points = _resolve_spawn_points(world, cfg)

    if cfg.junction_safe_distance_m is None:
        traffic_spawn_points = spawn_points
    else:
        traffic_spawn_points = (
            junction_spawn_transforms(
                world.get_map(),
                cfg.junction_safe_distance_m,
                jitter_m=20.0,
                points_per_lane=3,
                min_distance_m=4.0,
            )
            or world.get_map().get_spawn_points()
        )
    vehicle = _spawn_vehicle(world, cfg, spawn_points, hidden=hide_ego)

    _settle(world, cfg)

    return SimHandle(
        client,
        world,
        vehicle,
        original_settings,
        spawn_points,
        traffic_spawn_points,
    )


def respawn_vehicle(
    handle: SimHandle,
    cfg: PipelineConfig,
    hide_ego: bool = False,
) -> None:
    if handle.vehicle.is_alive:
        handle.vehicle.destroy()
    handle.vehicle = _spawn_vehicle(
        handle.world,
        cfg,
        handle.spawn_points,
        hidden=hide_ego,
    )
    _settle(handle.world, cfg)


def _lead_spawn_point(handle: SimHandle) -> carla.Transform | None:
    ego_wp = handle.world.get_map().get_waypoint(handle.vehicle.get_location())
    ahead = ego_wp.next(random.uniform(6.0, 12.0))
    if not ahead or ahead[0].is_junction:
        return None
    transform = ahead[0].transform
    transform.location.z += 0.3
    return transform


def _select_separated(
    spawn_points: list[carla.Transform],
    seed_xy: list[tuple[float, float]],
    min_separation_m: float,
) -> list[carla.Transform]:
    kept: list[carla.Transform] = []
    kept_xy: list[tuple[float, float]] = list(seed_xy)
    for sp in spawn_points:
        x, y = sp.location.x, sp.location.y
        if all(math.hypot(x - kx, y - ky) >= min_separation_m for kx, ky in kept_xy):
            kept.append(sp)
            kept_xy.append((x, y))
    return kept


def spawn_traffic(
    handle: SimHandle,
    cfg: PipelineConfig,
    count: int,
    min_separation_m: float = 5.0,
) -> list[carla.Vehicle]:
    if count <= 0:
        return []
    blueprint = handle.world.get_blueprint_library().find(cfg.vehicle.blueprint_id)
    SpawnActor = carla.command.SpawnActor

    ego_loc = handle.vehicle.get_transform().location
    shuffled = random.sample(
        handle.traffic_spawn_points, len(handle.traffic_spawn_points)
    )

    if random.random() < 0.5:
        lead = _lead_spawn_point(handle)
        if lead is not None:
            shuffled.insert(0, lead)
    candidates = _select_separated(shuffled, [(ego_loc.x, ego_loc.y)], min_separation_m)

    actor_ids: list[int] = []
    cursor = 0
    while len(actor_ids) < count and cursor < len(candidates):
        chunk = candidates[cursor : cursor + (count - len(actor_ids))]
        cursor += len(chunk)
        batch = [SpawnActor(blueprint, sp) for sp in chunk]
        for response in handle.client.apply_batch_sync(batch, True):
            if not response.error:
                actor_ids.append(response.actor_id)

    spawned = list(handle.world.get_actors(actor_ids))
    handle.traffic.extend(spawned)
    if spawned:
        _settle(handle.world, cfg)
    if len(spawned) < count:
        print(
            f"[traffic] only spawned {len(spawned)}/{count} bots "
            f"(map can't fit more at {min_separation_m:g}m spacing)"
        )
    else:
        print(f"[traffic] spawned {len(spawned)} traffic bots")
    return spawned


def _release_and_destroy(actor: carla.Actor | None) -> None:
    if actor is None:
        return
    try:
        if not actor.is_alive:
            return
        actor.destroy()
    except RuntimeError:
        pass


def despawn_traffic(handle: SimHandle) -> None:
    for bot in handle.traffic:
        _release_and_destroy(bot)
    handle.traffic.clear()


def shutdown(handle: SimHandle) -> None:
    for bot in handle.traffic:
        _release_and_destroy(bot)
    handle.traffic.clear()
    _release_and_destroy(handle.vehicle)
    handle.world.apply_settings(handle.original_settings)
