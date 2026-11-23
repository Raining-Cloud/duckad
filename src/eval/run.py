from __future__ import annotations

import argparse
import math

import carla

from ..aggregation.carla.duckies import cleanup_duckies, place_duckies
from ..aggregation.carla.sensors import SensorSync, spawn_main_camera
from ..aggregation.carla.world import connect_and_spawn, shutdown
from ..aggregation.command import Command
from ..aggregation.config import DuckieConfig, PipelineConfig
from ..aggregation.imaging import carla_image_to_bgr
from .config import EvalConfig
from .driver import build_agent, build_traffic_driver
from .inference import TrajectoryPredictor


def parse_args() -> argparse.Namespace:

    defaults = EvalConfig()
    parser = argparse.ArgumentParser(
        description="Drive a duckiebot in CARLA from a trained DuckAD checkpoint"
    )
    parser.add_argument(
        "checkpoint",
        help="model checkpoint path",
    )
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--map", default=defaults.map_name, help="CARLA map name")
    parser.add_argument(
        "--hdri-preset",
        default=defaults.hdri_preset,
        help="HDRI lighting preset",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=defaults.duration_s,
        help="driving time in seconds",
    )
    parser.add_argument(
        "--inference-hz",
        type=float,
        default=defaults.inference_hz,
        help="inference rate in Hz",
    )
    parser.add_argument(
        "--use-model-speed",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_model_speed,
        help="use the model-predicted target speed",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=defaults.target_speed_kmh,
        help="fallback target speed in km/h",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=defaults.max_speed_kmh,
        help="maximum target speed in km/h",
    )
    parser.add_argument(
        "--lookahead-index",
        type=int,
        default=defaults.lookahead_index,
        help="steering target waypoint index",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=defaults.device,
        help="inference device",
    )
    parser.add_argument(
        "--steer-noise",
        type=float,
        default=defaults.steer_noise_std,
        help="steering noise standard deviation; 0 disables",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=defaults.stall_timeout_s,
        help="seconds before stall recovery starts",
    )
    parser.add_argument(
        "--stall-creep-step",
        type=float,
        default=defaults.stall_creep_step_kmh,
        help="stall creep increase in km/h",
    )
    parser.add_argument(
        "--stall-creep-max",
        type=float,
        default=defaults.stall_creep_max_kmh,
        help="maximum stall creep speed in km/h",
    )
    parser.add_argument(
        "--traffic",
        type=int,
        default=defaults.traffic_count,
        metavar="N",
        help="number of additional traffic vehicles",
    )
    parser.add_argument(
        "--duckies",
        action="store_true",
        default=defaults.duckies,
        help="place duckies beside the road",
    )
    parser.add_argument(
        "--duckie-obstacle",
        action="store_true",
        default=defaults.duckie_obstacle,
        help="place duckies on the road as obstacles",
    )
    parser.add_argument(
        "--duckie-min-separation",
        type=float,
        default=defaults.duckie_min_separation_m,
        help="minimum distance between duckies in meters",
    )
    parser.add_argument(
        "--duckie-lateral-offset",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=defaults.duckie_lateral_offset,
        help="minimum and maximum distance from lane center in meters",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=defaults.visualize,
        help="open a live pygame preview window",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    cfg = EvalConfig()
    cfg.checkpoint = args.checkpoint
    cfg.host = args.host
    cfg.port = args.port
    cfg.map_name = args.map
    cfg.hdri_preset = args.hdri_preset
    cfg.duration_s = args.duration
    cfg.inference_hz = args.inference_hz
    cfg.use_model_speed = args.use_model_speed
    cfg.target_speed_kmh = args.target_speed
    cfg.max_speed_kmh = args.max_speed
    cfg.lookahead_index = args.lookahead_index
    cfg.device = args.device
    cfg.steer_noise_std = args.steer_noise
    cfg.stall_timeout_s = args.stall_timeout
    cfg.stall_creep_step_kmh = args.stall_creep_step
    cfg.stall_creep_max_kmh = args.stall_creep_max
    cfg.traffic_count = args.traffic
    cfg.duckies = args.duckies or args.duckie_obstacle
    cfg.duckie_obstacle = args.duckie_obstacle
    cfg.duckie_min_separation_m = args.duckie_min_separation
    cfg.duckie_lateral_offset = tuple(args.duckie_lateral_offset)
    cfg.visualize = args.visualize
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    device = cfg.device

    predictor = TrajectoryPredictor(cfg.checkpoint, device)

    pcfg = PipelineConfig(host=cfg.host, port=cfg.port)
    pcfg.map_name = cfg.map_name
    pcfg.hdri_preset = cfg.hdri_preset
    pcfg.fixed_delta_seconds = cfg.fixed_delta_seconds
    rig = pcfg.main_camera.rigs[0]

    num_ticks = int(cfg.duration_s / pcfg.fixed_delta_seconds)

    agent, _ = build_agent(predictor, cfg, pcfg.fixed_delta_seconds)

    handle = connect_and_spawn(pcfg)
    traffic_driver = build_traffic_driver(handle, pcfg, cfg.traffic_count)

    ducks_placed = False
    if cfg.duckies:
        place_duckies(
            handle.client,
            handle.world,
            DuckieConfig(
                enabled=True,
                obstacle=cfg.duckie_obstacle,
                min_separation_m=cfg.duckie_min_separation_m,
                min_lateral_offset_m=cfg.duckie_lateral_offset[0],
                max_lateral_offset_m=cfg.duckie_lateral_offset[1],
            ),
        )
        ducks_placed = True
        for _ in range(10):
            handle.world.tick()

    camera = spawn_main_camera(handle.world, handle.vehicle, rig)
    sync = SensorSync(handle.world, {"front": camera})

    viewer = None
    if cfg.visualize:
        from .visualize import EvalViewer

        viewer = EvalViewer(rig.image_size_x, rig.image_size_y)

    try:
        for tick_i in range(num_ticks):
            if traffic_driver is not None:
                traffic_driver.control_tick()
            _, data = sync.tick()
            main_bgr = carla_image_to_bgr(data["front"])
            command = (
                viewer.get_command() if viewer is not None else int(Command.DEFAULT)
            )

            velocity = handle.vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            drive = agent.step(tick_i, main_bgr, command, speed_kmh)

            control = drive.control
            if viewer is not None:
                manual = viewer.manual_control()
                if manual is not None:
                    throttle, steer, brake, reverse = manual
                    control = carla.VehicleControl(
                        throttle=throttle, steer=steer, brake=brake, reverse=reverse
                    )
            handle.vehicle.apply_control(control)

            if viewer is not None and not viewer.render(
                main_bgr,
                drive.bev_idx,
                drive.path_local,
                drive.target_index,
                drive.control,
                speed_kmh,
                command,
                drive.executed_command,
                drive.predicted_command,
                drive.target_speed_kmh,
                drive.temporal_local,
                drive.selected_temporal_index,
            ):
                break
    finally:
        sync.destroy()
        if viewer is not None:
            viewer.close()
        if ducks_placed:
            cleanup_duckies(handle.client, handle.world)
        shutdown(handle)


if __name__ == "__main__":
    main()
