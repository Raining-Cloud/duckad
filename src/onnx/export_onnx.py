from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import onnx

from ..aggregation.command import Command
from ..aggregation.config import BEVConfig, TargetImageConfig, TrajectoryConfig
from ..aggregation.data.normalize import checkpoint_norm_constants
from ..aggregation.imaging import BEV_NUM_CLASSES
from ..model import DuckAD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained DuckAD checkpoint to ONNX. The waypoint head (gru/mlp) is read "
        "from the checkpoint's own saved config, so the exported graph matches how it was trained. "
        "The model takes (image, command) and returns (spatial, temporal, bev, command_logits) - the "
        "same forward as eval.inference.TrajectoryPredictor, minus the letterbox/denormalize pre/post "
        "processing (do those around the ONNX graph, as inference.py does around the torch model)."
    )
    parser.add_argument(
        "checkpoint",
        help="path to a train.py checkpoint (e.g. checkpoints/<run>/last.pt)",
    )
    parser.add_argument(
        "-o", "--out", help="output .onnx path (default: <checkpoint dir>/model.onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=15,
        help="ONNX opset version (default: 15, the max the robot's onnxruntime 1.11 loads)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cpu",
        help="device to trace on; cpu is fine and most portable",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="mark the batch dimension as dynamic. Off by default: a fixed batch of 1 is baked in, "
        "which is what the duckiebot runs and keeps the per-command gather simple.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location=args.device, weights_only=False
    )
    head = (checkpoint["config"].get("model") or {}).get("head", "gru")

    model = DuckAD(
        num_waypoints=TrajectoryConfig().num_points,
        bev_classes=BEV_NUM_CLASSES,
        bev_size=BEVConfig().image_size,
        head=head,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    size = TargetImageConfig().size
    dummy_image = torch.zeros(1, 3, size, size, device=args.device)
    dummy_command = torch.full(
        (1,), int(Command.DEFAULT), dtype=torch.long, device=args.device
    )

    io_names = ["image", "command", "spatial", "temporal", "bev", "command_logits"]
    dynamic_axes = (
        {name: {0: "batch"} for name in io_names} if args.dynamic_batch else None
    )

    out_path = (
        Path(args.out) if args.out else Path(args.checkpoint).parent / "model.onnx"
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_image, dummy_command),
        str(out_path),
        input_names=["image", "command"],
        output_names=["spatial", "temporal", "bev", "command_logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    metadata = {"trajectory_norm": json.dumps(checkpoint_norm_constants(checkpoint))}
    temporal_dt = (checkpoint["config"].get("data") or {}).get("temporal_dt")
    if temporal_dt is not None:
        metadata["temporal_dt"] = str(float(temporal_dt))

    model_proto = onnx.load(str(out_path))
    for key, value in metadata.items():
        entry = model_proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model_proto, str(out_path))

    print(
        f"exported head={head} model -> {out_path} "
        f"(opset {args.opset}, {'dynamic' if args.dynamic_batch else 'fixed=1'} batch)\n"
        f"embedded metadata: {', '.join(metadata)}"
    )


if __name__ == "__main__":
    main()
