from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

from ..carla.duckie_agent import AgentDriver, build_global_route_planner
from ..carla.duckies import cleanup_duckies, place_duckies
from ..carla.sensors import (
    SensorSync,
    rig_ground_transform,
    spawn_bev_camera,
    spawn_main_camera,
    spawn_main_segmentation_camera,
)
from ..carla.world import (
    connect_and_spawn,
    despawn_traffic,
    respawn_vehicle,
    shutdown,
    spawn_traffic,
)
from ..config import (
    DuckieConfig,
    PipelineConfig,
    StorageConfig,
    TrajectoryConfig,
    bev_config_for_rig,
)
from ..imaging import (
    background_mask,
    bev_semantic_labels,
    carla_image_to_bgr,
    letterbox,
)
from ..trajectory import sample_future_trajectory, trajectory_horizon_m
from .recorder import SegmentBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Carla duckiebot training data"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="recording time per location in seconds",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=1,
        help="number of recording locations; respawns between segments",
    )
    parser.add_argument(
        "--junction",
        action="store_true",
        help="spawn near junctions",
    )
    parser.add_argument(
        "--junction-safe-distance",
        type=float,
        default=15.0,
        help="spawn distance before junctions in meters",
    )
    parser.add_argument(
        "--out",
        default=StorageConfig().out_dir,
        help="output root; a <map>_<hdri>_<timestamp> run folder is created inside",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=StorageConfig().batch_size,
        help="number of samples per output file",
    )
    parser.add_argument(
        "--temporal-hz",
        type=float,
        default=TrajectoryConfig().temporal_hz,
        help="trajectory sampling rate in Hz",
    )
    parser.add_argument(
        "--temporal-points",
        type=int,
        default=TrajectoryConfig().temporal_num_points,
        help="number of trajectory points",
    )
    parser.add_argument(
        "--duckies",
        action="store_true",
        help="place duckies beside the road",
    )
    parser.add_argument(
        "--duckie-obstacle",
        action="store_true",
        help="place duckies on the road as obstacles",
    )
    parser.add_argument(
        "--duckie-min-separation",
        type=float,
        default=DuckieConfig().min_separation_m,
        help="minimum distance between duckies in meters",
    )
    parser.add_argument(
        "--duckie-lateral-offset",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(
            DuckieConfig().min_lateral_offset_m,
            DuckieConfig().max_lateral_offset_m,
        ),
        help="minimum and maximum distance from lane center in meters",
    )
    parser.add_argument(
        "--duckie-seed",
        type=int,
        default=None,
        help="random seed for duckie placement",
    )
    parser.add_argument(
        "--duckie-wait-timeout",
        type=float,
        default=DuckieConfig().wait_timeout_s,
        help="seconds to stop before removing an obstacle duckie",
    )
    parser.add_argument(
        "--traffic",
        type=int,
        default=0,
        metavar="N",
        help="number of additional traffic vehicles",
    )
    parser.add_argument(
        "--map",
        default="duckietown_02",
        help="CARLA map name",
    )
    parser.add_argument(
        "--hdri-preset",
        default=None,
        help="HDRI lighting preset; omit for the map's default backdrop",
    )
    parser.add_argument(
        "--camera-yaw-jitter",
        type=float,
        default=0.0,
        metavar="DEG",
        help="random camera yaw range in degrees; 0 disables",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="open a pygame preview window"
    )
    parser.add_argument(
        "--visualize-server",
        action="store_true",
        help="draw camera, BEV and trajectory markers in CARLA; disables recording",
    )
    return parser.parse_args()


def _spawn_segment_sensors(handle, cfg, rigs, yaw_jitter_deg=0.0):
    rig_yaw = {
        rig.name: (
            random.uniform(-yaw_jitter_deg, yaw_jitter_deg) if yaw_jitter_deg else 0.0
        )
        for rig in rigs
    }
    main_sensors = {
        rig.name: spawn_main_camera(
            handle.world, handle.vehicle, rig, yaw_offset=rig_yaw[rig.name]
        )
        for rig in rigs
    }
    bev_sensors = {
        rig.name: spawn_bev_camera(
            handle.world,
            handle.vehicle,
            bev_config_for_rig(cfg.bev, rig),
            yaw_offset=rig_yaw[rig.name],
        )
        for rig in rigs
    }
    seg_sensors = {
        rig.name: spawn_main_segmentation_camera(
            handle.world, handle.vehicle, rig, yaw_offset=rig_yaw[rig.name]
        )
        for rig in rigs
    }
    sync = SensorSync(
        handle.world,
        {
            **{f"main_{name}": sensor for name, sensor in main_sensors.items()},
            **{f"bev_{name}": sensor for name, sensor in bev_sensors.items()},
            **{f"seg_{name}": sensor for name, sensor in seg_sensors.items()},
        },
    )
    return main_sensors, bev_sensors, sync, rig_yaw


def _pad_future(handle, buffer, horizon_ticks, driver):
    for _ in range(horizon_ticks):
        try:
            driver.control_tick()
        except RuntimeError:
            break
        handle.world.tick()
        if not handle.vehicle.is_alive:
            break
        loc = handle.vehicle.get_transform().location
        buffer.add_ego(loc.x, loc.y, loc.z)


def _capture_tick(
    args,
    cfg,
    handle,
    carla_map,
    sync,
    rigs,
    main_rig,
    main_sensors,
    bev_sensors,
    rig_yaw,
    buffer,
    viewer,
    driver,
    tick,
):
    driver.control_tick()
    frame, data = sync.tick()

    vehicle_transform = handle.vehicle.get_transform()
    velocity = handle.vehicle.get_velocity()
    speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    if buffer is not None:
        loc = vehicle_transform.location
        buffer.add_ego(loc.x, loc.y, loc.z)

    command, anchor_waypoints = driver.label(handle.vehicle)

    rig_frames = {}
    for rig in rigs:
        main_bgr = carla_image_to_bgr(data[f"main_{rig.name}"])
        main_img = letterbox(main_bgr, cfg.target_image)

        bev_bgr = carla_image_to_bgr(data[f"bev_{rig.name}"])
        bev_seg = bev_semantic_labels(bev_bgr)

        seg_bgr = carla_image_to_bgr(data[f"seg_{rig.name}"])
        main_seg = letterbox(
            background_mask(seg_bgr), cfg.target_image, cv2.INTER_NEAREST
        )

        reference_transform = rig_ground_transform(
            vehicle_transform, rig, yaw_offset=rig_yaw[rig.name]
        )
        trajectory = sample_future_trajectory(
            carla_map,
            vehicle_transform,
            cfg.trajectory,
            reference_transform,
            anchor_waypoints,
        )

        if buffer is not None:
            buffer.add(
                tick=tick,
                frame=frame,
                main_img=main_img,
                main_seg=main_seg,
                bev_seg=bev_seg,
                trajectory=trajectory,
                speed=speed,
                command=command,
                ref_inv_matrix=np.array(reference_transform.get_inverse_matrix()),
            )

        rig_frames[rig.name] = (main_img, bev_seg, trajectory, command, None)

        if args.visualize_server:
            from .visualize import draw_rig_markers, draw_trajectory_markers

            draw_rig_markers(
                handle.world,
                vehicle_transform,
                main_sensors[rig.name].get_transform(),
                bev_sensors[rig.name].get_transform(),
                cfg.bev.size_m,
                life_time=cfg.fixed_delta_seconds * 2,
            )
            if rig is main_rig:
                trajectory_world = [
                    reference_transform.transform(
                        carla.Location(float(x), float(y), 0.0)
                    )
                    for x, y in trajectory
                ]
                draw_trajectory_markers(
                    handle.world,
                    vehicle_transform,
                    trajectory_world,
                    life_time=cfg.fixed_delta_seconds * 2,
                )

    viewer_open = viewer is None or viewer.render(rig_frames)
    return vehicle_transform, viewer_open


_DUCKIE_SETTLE_TICKS = 10


def _respawn_duckies(handle, cfg, run_index):
    if not cfg.duckies.enabled:
        return
    cleanup_duckies(handle.client, handle.world)
    place_duckies(handle.client, handle.world, cfg.duckies, run_index=run_index)
    for _ in range(_DUCKIE_SETTLE_TICKS):
        handle.world.tick()


def _respawn_agent_traffic(handle, cfg, args, agent_driver):
    despawn_traffic(handle)
    agent_driver.reset_traffic()
    if args.traffic <= 0:
        return
    bots = spawn_traffic(handle, cfg, args.traffic)
    for bot in bots:
        agent_driver.add_traffic(bot, cfg.vehicle.max_speed_kmh)


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(host=args.host, port=args.port)
    cfg.map_name = args.map
    cfg.hdri_preset = args.hdri_preset
    cfg.junction_safe_distance_m = (
        args.junction_safe_distance if args.junction else None
    )
    cfg.duckies.enabled = args.duckies or args.duckie_obstacle
    cfg.duckies.obstacle = args.duckie_obstacle
    cfg.duckies.min_separation_m = args.duckie_min_separation
    cfg.duckies.min_lateral_offset_m, cfg.duckies.max_lateral_offset_m = (
        args.duckie_lateral_offset
    )
    cfg.duckies.seed = args.duckie_seed
    cfg.duckies.wait_timeout_s = args.duckie_wait_timeout
    cfg.storage.out_dir = args.out
    cfg.storage.batch_size = args.batch_size
    cfg.trajectory.temporal_hz = args.temporal_hz
    cfg.trajectory.temporal_num_points = args.temporal_points
    if args.visualize:
        cfg.vehicle.hide_ego_mesh = False

    capture_time = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    hdri_tag = cfg.hdri_preset or "defaulthdri"
    run_name = f"{cfg.map_name or 'defaultmap'}_{hdri_tag}_{capture_time}"
    cfg.storage.out_dir = str(Path(cfg.storage.out_dir) / run_name)

    rigs = cfg.main_camera.rigs
    main_rig = rigs[0]

    record = not (args.visualize or args.visualize_server)

    horizon_ticks = cfg.trajectory.temporal_horizon_ticks(cfg.fixed_delta_seconds)
    temporal_dt = cfg.trajectory.temporal_dt(cfg.fixed_delta_seconds)
    if record:
        print(
            f"temporal trajectory: {cfg.trajectory.temporal_num_points} points @ "
            f"{cfg.trajectory.temporal_hz}Hz -> dt={temporal_dt:.3f}s, "
            f"{horizon_ticks} future-pad ticks/segment; "
            f"{cfg.storage.batch_size} samples per output file"
        )

    hide_ego = cfg.vehicle.hide_ego_mesh

    handle = connect_and_spawn(cfg, hide_ego=hide_ego)
    sync = None
    buffer = (
        SegmentBuffer(
            cfg.storage,
            temporal_num_points=cfg.trajectory.temporal_num_points,
            temporal_stride_ticks=cfg.trajectory.temporal_stride_ticks(
                cfg.fixed_delta_seconds
            ),
            temporal_dt=temporal_dt,
            file_suffix=capture_time,
        )
        if record
        else None
    )
    viewer = None
    try:
        if args.visualize:
            from .visualize import PygameViewer

            viewer = PygameViewer(
                cfg.target_image.size,
                main_camera_xy=(main_rig.transform.x, main_rig.transform.y),
                bev_xy=(cfg.bev.transform.x, cfg.bev.transform.y),
                bev_size_m=cfg.bev.size_m,
            )

        carla_map = handle.world.get_map()
        command_counts: dict = {}

        grp = build_global_route_planner(carla_map)
        agent_driver = AgentDriver(
            carla_map,
            command_counts,
            grp=grp,
            rng=random.Random(),
            report_distance_m=trajectory_horizon_m(cfg.trajectory),
            duckie_wait_timeout=cfg.duckies.wait_timeout_s,
        )

        def make_driver():
            agent_driver.set_ego(
                handle.vehicle, target_speed_kmh=cfg.vehicle.max_speed_kmh
            )
            return agent_driver

        num_ticks = int(args.duration / cfg.fixed_delta_seconds)
        for segment in range(args.segments):
            if segment > 0:
                respawn_vehicle(handle, cfg, hide_ego=hide_ego)

            _respawn_duckies(handle, cfg, segment)
            _respawn_agent_traffic(handle, cfg, args, agent_driver)
            driver = make_driver()
            main_sensors, bev_sensors, sync, rig_yaw = _spawn_segment_sensors(
                handle, cfg, rigs, args.camera_yaw_jitter
            )

            viewer_closed = False
            vehicle_lost = False
            for tick in range(num_ticks):
                try:
                    _, viewer_open = _capture_tick(
                        args,
                        cfg,
                        handle,
                        carla_map,
                        sync,
                        rigs,
                        main_rig,
                        main_sensors,
                        bev_sensors,
                        rig_yaw,
                        buffer,
                        viewer,
                        driver,
                        tick,
                    )
                except RuntimeError:
                    if handle.vehicle.is_alive:
                        raise
                    print(
                        f"segment {segment}: vehicle despawned mid-run at tick {tick} "
                        f"(drove into a dead end?) - starting a new segment"
                    )
                    vehicle_lost = True
                    break
                if not viewer_open:
                    viewer_closed = True
                    break

            sync.destroy()
            sync = None
            if buffer is not None:
                if not vehicle_lost:
                    _pad_future(handle, buffer, horizon_ticks, driver)
                buffer.end_segment()
            if viewer_closed:
                break
    finally:
        if buffer is not None:
            dropped = buffer.close()
            if dropped:
                print(
                    f"dropped {dropped} trailing sample(s) "
                    f"(no complete future, or fewer than {args.batch_size} samples remain)"
                )
        if sync is not None:
            sync.destroy()
        if viewer is not None:
            viewer.close()
        if cfg.duckies.enabled:
            cleanup_duckies(handle.client, handle.world)
        shutdown(handle)


if __name__ == "__main__":
    main()
