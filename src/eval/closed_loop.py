from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

import carla

from ..aggregation.carla.sensors import SensorSync, spawn_main_camera
from ..aggregation.carla.world import connect_and_spawn, shutdown
from ..aggregation.command import Command
from ..aggregation.config import PipelineConfig
from ..aggregation.imaging import carla_image_to_bgr
from .config import EvalConfig
from .driver import build_agent, build_traffic_driver
from .inference import TrajectoryPredictor
from .metrics import ClosedLoopMetrics, ClosedLoopSummary


def parse_args() -> argparse.Namespace:
    defaults = EvalConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate DuckAD checkpoints in closed-loop CARLA driving"
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        help="model checkpoint paths to evaluate",
    )
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--map", default=defaults.map_name, help="CARLA map name")
    parser.add_argument(
        "--hdri-preset", default=defaults.hdri_preset, help="HDRI lighting preset"
    )
    parser.add_argument("--inference-hz", type=float, default=defaults.inference_hz)
    parser.add_argument(
        "--use-model-speed",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_model_speed,
    )
    parser.add_argument("--target-speed", type=float, default=defaults.target_speed_kmh)
    parser.add_argument("--max-speed", type=float, default=defaults.max_speed_kmh)
    parser.add_argument("--lookahead-index", type=int, default=defaults.lookahead_index)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=defaults.device)
    parser.add_argument(
        "--steer-noise",
        type=float,
        default=defaults.steer_noise_std,
        help="steering noise standard deviation; 0 disables",
    )
    parser.add_argument(
        "--steer-noise-step",
        type=float,
        default=defaults.steer_noise_step,
        help="steering-noise increase per completed lap or junction",
    )
    parser.add_argument(
        "--traffic",
        type=int,
        default=defaults.traffic_count,
        metavar="N",
        help="number of additional traffic vehicles",
    )
    parser.add_argument(
        "--junction",
        action="store_true",
        default=defaults.junction,
        help="evaluate LEFT and RIGHT junction commands",
    )
    parser.add_argument(
        "--max-junctions",
        type=int,
        default=defaults.max_junctions,
        help="junctions to evaluate; 0 disables the limit",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=defaults.max_duration_s,
        help="maximum simulated seconds per run",
    )
    parser.add_argument(
        "--departure-threshold",
        type=float,
        default=defaults.departure_threshold_m,
        help="lane-departure threshold in meters",
    )
    parser.add_argument(
        "--abort-threshold",
        type=float,
        default=defaults.abort_threshold_m,
        help="run-abort distance from lane in meters",
    )
    parser.add_argument(
        "--lap-radius",
        type=float,
        default=defaults.lap_radius_m,
        help="lap-counting radius in meters",
    )
    parser.add_argument(
        "--stall-timeout",
        type=float,
        default=defaults.progress_stall_timeout_s,
        help="seconds without progress before aborting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="seed list or range; overrides --seed",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="summary JSON output file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="open a live pygame preview window",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, checkpoint: str) -> EvalConfig:
    cfg = EvalConfig()
    cfg.checkpoint = checkpoint
    cfg.host = args.host
    cfg.port = args.port
    cfg.map_name = args.map
    cfg.hdri_preset = args.hdri_preset
    cfg.inference_hz = args.inference_hz
    cfg.use_model_speed = args.use_model_speed
    cfg.target_speed_kmh = args.target_speed
    cfg.max_speed_kmh = args.max_speed
    cfg.lookahead_index = args.lookahead_index
    cfg.device = args.device
    cfg.steer_noise_std = args.steer_noise
    cfg.steer_noise_step = args.steer_noise_step
    cfg.traffic_count = args.traffic
    cfg.junction = args.junction
    cfg.max_junctions = args.max_junctions
    cfg.max_duration_s = args.max_duration
    cfg.departure_threshold_m = args.departure_threshold
    cfg.abort_threshold_m = args.abort_threshold
    cfg.lap_radius_m = args.lap_radius
    cfg.progress_stall_timeout_s = args.stall_timeout
    cfg.visualize = args.visualize
    return cfg


def _road_state(
    carla_map: carla.Map, location: carla.Location
) -> tuple[tuple[float, float] | None, bool]:
    wp = carla_map.get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if wp is None:
        return None, False
    return (wp.transform.location.x, wp.transform.location.y), wp.is_junction


def run_one(cfg: EvalConfig, seed: int, device: str) -> ClosedLoopSummary:
    random.seed(seed)
    predictor = TrajectoryPredictor(cfg.checkpoint, device)

    pcfg = PipelineConfig(host=cfg.host, port=cfg.port)
    pcfg.map_name = cfg.map_name
    pcfg.hdri_preset = cfg.hdri_preset
    pcfg.fixed_delta_seconds = cfg.fixed_delta_seconds
    rig = pcfg.main_camera.rigs[0]

    agent, _ = build_agent(
        predictor, cfg, pcfg.fixed_delta_seconds, rng=random.Random(seed)
    )
    metrics = ClosedLoopMetrics(
        cfg.departure_threshold_m,
        cfg.abort_threshold_m,
        cfg.lap_radius_m,
        cfg.progress_stall_timeout_s,
        cfg.progress_stall_min_m,
    )

    handle = connect_and_spawn(pcfg)
    traffic_driver = build_traffic_driver(
        handle, pcfg, cfg.traffic_count, rng=random.Random(seed)
    )

    carla_map = handle.world.get_map()
    camera = spawn_main_camera(handle.world, handle.vehicle, rig)
    sync = SensorSync(handle.world, {"front": camera})

    viewer = None
    if cfg.visualize:
        from .visualize import EvalViewer

        viewer = EvalViewer(rig.image_size_x, rig.image_size_y)

    num_ticks = int(cfg.max_duration_s / pcfg.fixed_delta_seconds)
    dt = pcfg.fixed_delta_seconds
    direction_rng = random.Random(seed)
    command = (
        direction_rng.choice((int(Command.LEFT), int(Command.RIGHT)))
        if cfg.junction
        else int(Command.DEFAULT)
    )
    loop_index = 0
    try:
        for tick_i in range(num_ticks):
            if traffic_driver is not None:
                traffic_driver.control_tick()
            _, data = sync.tick()
            main_bgr = carla_image_to_bgr(data["front"])

            velocity = handle.vehicle.get_velocity()
            speed_kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            drive = agent.step(tick_i, main_bgr, command, speed_kmh)
            handle.vehicle.apply_control(drive.control)

            transform = handle.vehicle.get_transform()
            loc = transform.location
            lane_center, is_junction = _road_state(carla_map, loc)
            metrics.update(
                loc.x,
                loc.y,
                lane_center,
                dt,
                is_junction=is_junction,
                yaw_deg=transform.rotation.yaw,
                command=command,
            )

            index = (
                metrics.junctions_evaluated if cfg.junction else metrics.laps_completed
            )
            if index > loop_index:
                loop_index = index
                if cfg.steer_noise_step:
                    agent.steer_noise_std = (
                        cfg.steer_noise_std + loop_index * cfg.steer_noise_step
                    )
                label = "junction" if cfg.junction else "lap"
                if cfg.junction:
                    command = direction_rng.choice(
                        (int(Command.LEFT), int(Command.RIGHT))
                    )
                print(
                    f"  {label} {loop_index} @ {metrics.summary().distance_m:.0f} m -> "
                    f"steer noise {agent.steer_noise_std:.3f}, next command {Command(command).name}"
                )
                if (
                    cfg.junction
                    and cfg.max_junctions
                    and loop_index >= cfg.max_junctions
                ):
                    print(f"  route complete: {loop_index} junctions evaluated")
                    break

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
            if metrics.failed():
                break
    finally:
        sync.destroy()
        if viewer is not None:
            viewer.close()
        shutdown(handle)

    return metrics.summary()


def _peak_noise(args: argparse.Namespace, summary: ClosedLoopSummary) -> float:
    index = summary.junctions_evaluated if args.junction else summary.laps_completed
    return args.steer_noise + index * args.steer_noise_step


def _parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError(f"no seeds parsed from {spec!r}")
    return seeds


@dataclass
class RunRecord:
    name: str
    checkpoint: str
    seed: int
    summary: ClosedLoopSummary
    peak_noise: float


def _print_runs_table(records: list[RunRecord]) -> None:
    header = f"{'checkpoint':<28} {'seed':>4} {'laps':>4} {'departs':>7} {'junc_ok':>8} {'dist_m':>8} {'time_s':>7} {'cte_mean':>8} {'cte_rms':>8} {'cte_max':>8} {'pk_noise':>8}  outcome"
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        s = r.summary
        outcome = s.failure_reason if s.failed else "survived"
        junc = f"{s.junctions_correct}/{s.junctions_evaluated}"
        print(
            f"{r.name:<28} {r.seed:>4d} {s.laps_completed:>4d} {s.lane_departures:>7d} {junc:>8} {s.distance_m:>8.1f} {s.duration_s:>7.1f} "
            f"{s.cte_mean_m:>8.3f} {s.cte_rms_m:>8.3f} {s.cte_max_m:>8.3f} {r.peak_noise:>8.3f}  {outcome}"
        )


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def _aggregate(records: list[RunRecord]) -> dict:
    ss = [r.summary for r in records]
    junc_correct = sum(s.junctions_correct for s in ss)
    junc_evaluated = sum(s.junctions_evaluated for s in ss)
    agg: dict = {
        "n": len(records),
        "failures": sum(1 for s in ss if s.failed),
        "junctions_correct": junc_correct,
        "junctions_evaluated": junc_evaluated,
        "junction_success": round(junc_correct / junc_evaluated, 4)
        if junc_evaluated
        else None,
    }
    fields = {
        "laps_completed": [float(s.laps_completed) for s in ss],
        "lane_departures": [float(s.lane_departures) for s in ss],
        "distance_m": [s.distance_m for s in ss],
        "duration_s": [s.duration_s for s in ss],
        "cte_mean_m": [s.cte_mean_m for s in ss],
        "cte_rms_m": [s.cte_rms_m for s in ss],
        "cte_max_m": [s.cte_max_m for s in ss],
        "peak_steer_noise_std": [r.peak_noise for r in records],
    }
    for key, values in fields.items():
        mean, std = _mean_std(values)
        agg[key] = {"mean": round(mean, 4), "std": round(std, 4)}
    return agg


def _print_aggregate_table(names: list[str], aggs: list[dict]) -> None:
    header = f"{'checkpoint':<28} {'runs':>4} {'laps':>11} {'departs':>12} {'junc_ok':>11} {'dist_m':>15} {'pk_noise':>14} {'cte_mean':>14} {'fails':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name, a in zip(names, aggs):

        def ms(key: str, prec: int) -> str:
            return f"{a[key]['mean']:.{prec}f}±{a[key]['std']:.{prec}f}"

        rate = a["junction_success"]
        junc = f"{a['junctions_correct']}/{a['junctions_evaluated']}" + (
            f" {rate:.0%}" if rate is not None else ""
        )
        print(
            f"{name:<28} {a['n']:>4d} {ms('laps_completed', 1):>11} {ms('lane_departures', 0):>12} {junc:>11} "
            f"{ms('distance_m', 0):>15} {ms('peak_steer_noise_std', 3):>14} {ms('cte_mean_m', 3):>14} "
            f"{a['failures']:>4d}/{a['n']:<2d}"
        )


def main() -> None:
    args = parse_args()
    device = args.device
    seeds = _parse_seeds(args.seeds) if args.seeds else [args.seed]

    names = [Path(c).parent.name or Path(c).stem for c in args.checkpoints]
    records: list[RunRecord] = []
    aggs: list[dict] = []
    for checkpoint, name in zip(args.checkpoints, names):
        ckpt_records: list[RunRecord] = []
        for seed in seeds:
            print(f"\n=== {name} | seed {seed} ({checkpoint}) ===")
            cfg = build_config(args, checkpoint)
            summary = run_one(cfg, seed, device)
            print(json.dumps(summary.as_dict(), indent=2))
            rec = RunRecord(name, checkpoint, seed, summary, _peak_noise(args, summary))
            ckpt_records.append(rec)
            records.append(rec)
        aggs.append(_aggregate(ckpt_records))

    _print_runs_table(records)
    if len(seeds) > 1:
        print(f"\nmean over {len(seeds)} seeds {seeds}:")
        _print_aggregate_table(names, aggs)

    if args.json_out:
        payload = {
            "seeds": seeds,
            "steer_noise_std": args.steer_noise,
            "steer_noise_step": args.steer_noise_step,
            "runs": [
                {
                    "checkpoint": r.checkpoint,
                    "name": r.name,
                    "seed": r.seed,
                    "peak_steer_noise_std": round(r.peak_noise, 4),
                    **r.summary.as_dict(),
                }
                for r in records
            ],
            "aggregate": [{"name": n, **a} for n, a in zip(names, aggs)],
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
