from __future__ import annotations

import argparse
import functools
import json
import pickle
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..aggregation.command import COMMAND_COLORS_BGR, COMMAND_NAMES, Command
from ..aggregation.config import TrajectoryConfig
from ..aggregation.data.augment import build_train_transform
from ..aggregation.data.background import ImageNetBackgroundCache
from ..aggregation.data.dataset import BatchedTrajectoryDataset
from ..aggregation.data.normalize import denormalize
from ..aggregation.imaging import BEV_BACKGROUND_INDEX
from ..model import DuckAD, loss_duckad

_BEV_COLORS_HEX = [
    "#1E1E1E",
    "#FFD700",
    "#FFFFFF",
    "#3C3C3C",
    "#C81E1E",
    "#3C78FF",
    "#FF8C00",
    "#FF00FF",
]

_COMMAND_COLORS_RGB = {
    cmd: tuple(c / 255 for c in reversed(bgr))
    for cmd, bgr in COMMAND_COLORS_BGR.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DuckAD on collect.py's recorded batches"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/train_gru.yaml",
        help="path to a YAML training config",
    )

    return parser.parse_args()


def resolve_runtime(raw: dict) -> dict:
    runtime = raw.get("runtime", {})

    device = runtime.get("device")
    if device not in {"cuda", "cpu"}:
        raise ValueError("runtime.device must be either 'cuda' or 'cpu'")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    runname = runtime.get("runname")
    if not isinstance(runname, str) or not runname.strip():
        raise ValueError("runtime.runname must be a non-empty string")
    if Path(runname).name != runname:
        raise ValueError("runtime.runname must be a directory name, not a path")
    checkpoint_dir = Path("checkpoints") / f"{runname}-{timestamp}"

    return {
        "seed": runtime.get("seed", 0),
        "device": device,
        "checkpoint_dir": checkpoint_dir,
        "runname": runname,
        "init_weights": runtime.get("init_weights"),
    }


def perturb_default_commands(command: torch.Tensor, prob: float) -> torch.Tensor:
    """
    Randomly switch some DEFAULT model commands to a turn while keeping the straight target,
    so every head learns lane following outside junctions
    """
    is_default = command == Command.DEFAULT
    random_turn = torch.randint(
        Command.LEFT, Command.RIGHT + 1, command.shape, device=command.device
    )
    swap = is_default & (torch.rand(command.shape, device=command.device) < prob)

    return torch.where(swap, random_turn, command)


class ImageNetStopAug:
    """
    Replace some DEFAULT samples with ImageNet images and zero-velocity targets,
    teaching the model to stop outside familiar road scenes
    """

    def __init__(
        self, prob, norm_constants, num_waypoints, num_temporal, size=224, seed=0
    ):
        self.prob = prob
        self.bev_bg_index = BEV_BACKGROUND_INDEX
        self.cache = ImageNetBackgroundCache(size=size, seed=seed)

        def _norm(arr, xmin, xmax, ymin, ymax):
            out = np.empty_like(arr, dtype=np.float32)
            out[:, 0] = 2.0 * (arr[:, 0] - xmin) / (xmax - xmin) - 1.0
            out[:, 1] = 2.0 * (arr[:, 1] - ymin) / (ymax - ymin) - 1.0
            return out

        nc = norm_constants
        cfg = TrajectoryConfig()
        raw_x = cfg.start_offset_m + np.arange(num_waypoints) * cfg.spacing_m
        traj = np.stack([raw_x, np.zeros(num_waypoints)], axis=1)
        self.forward_traj = torch.from_numpy(
            _norm(traj, nc["x_min"], nc["x_max"], nc["y_min"], nc["y_max"])
        )
        tzero = np.zeros((num_temporal, 2), dtype=np.float32)
        self.zero_temporal = torch.from_numpy(
            _norm(tzero, nc["tx_min"], nc["tx_max"], nc["ty_min"], nc["ty_max"])
        )

    def to(self, device):
        self.forward_traj = self.forward_traj.to(device)
        self.zero_temporal = self.zero_temporal.to(device)
        return self

    def apply(self, main_img, trajectory, gt_temporal, bev_seg, gt_cmd, model_cmd):
        sel = (gt_cmd == Command.DEFAULT) & (
            torch.rand(gt_cmd.shape, device=gt_cmd.device) < self.prob
        )
        k = int(sel.sum())
        if k == 0:
            return
        imgs = np.stack([self.cache.sample() for _ in range(k)])
        imgs = (
            torch.from_numpy(imgs)
            .permute(0, 3, 1, 2)
            .to(main_img.device, main_img.dtype)
            .div_(255.0)
        )
        main_img[sel] = imgs
        trajectory[sel] = self.forward_traj
        gt_temporal[sel] = self.zero_temporal
        bev_seg[sel] = self.bev_bg_index
        gt_cmd[sel] = int(Command.DEFAULT)
        model_cmd[sel] = int(Command.DEFAULT)


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def sample_indices_per_command(
    dataset: BatchedTrajectoryDataset, k: int, seed: int
) -> list[int]:
    by_command: dict[int, list[int]] = {int(c): [] for c in Command}
    offset = 0

    for path in dataset.batch_paths:
        with np.load(path) as data:
            commands = data["command"]

        for local_idx, cmd in enumerate(commands):
            by_command[int(cmd)].append(offset + local_idx)

        offset += len(commands)

    rng = random.Random(seed)
    indices = []

    for cmd in Command:
        pool = by_command[int(cmd)]
        indices.extend(rng.sample(pool, min(k, len(pool))))

    return indices


def build_optimizer(model, lr_backbone: float, lr_head: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.param_groups(lr_backbone, lr_head))


def run_epoch(
    model,
    loader,
    loss_fn,
    device,
    use_mixed_precision,
    optimizer=None,
    scaler=None,
    desc="train",
    command_perturb_prob=0.0,
    stop_aug=None,
):
    train = optimizer is not None
    model.train(train)

    totals = {
        "total": 0.0,
        "wp": 0.0,
        "temporal": 0.0,
        "bev": 0.0,
        "bev_acc": 0.0,
        "cmd": 0.0,
        "cmd_acc": 0.0,
    }
    wp_by_cmd_sum = {int(c): 0.0 for c in Command}
    wp_by_cmd_count = {int(c): 0 for c in Command}
    n_samples = 0
    progress = tqdm(
        loader, desc=desc, leave=False, unit="batch", disable=None, mininterval=5.0
    )
    with torch.set_grad_enabled(train):
        for batch in progress:
            main_img = batch["main_img"].to(device, non_blocking=True)
            bev_seg = batch["bev_seg"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)
            bsz = main_img.size(0)

            with torch.autocast(device_type=device, enabled=use_mixed_precision):
                gt_cmd = batch["command"].to(device, non_blocking=True)
                gt_temporal = batch["temporal_trajectory"].to(device, non_blocking=True)
                model_cmd = perturb_default_commands(gt_cmd, command_perturb_prob)

                if stop_aug is not None:
                    stop_aug.apply(
                        main_img, trajectory, gt_temporal, bev_seg, gt_cmd, model_cmd
                    )
                
                pred_wp, pred_temporal, pred_bev, pred_cmd = model(main_img, model_cmd)
                loss, l_wp, l_bev, l_cmd, l_temporal = loss_fn(
                    pred_wp,
                    pred_bev,
                    trajectory,
                    bev_seg,
                    pred_cmd,
                    gt_cmd,
                    pred_temporal,
                    gt_temporal,
                )

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            bev_acc = (pred_bev.argmax(1) == bev_seg).float().mean()
            cmd_acc = (pred_cmd.argmax(1) == gt_cmd).float().mean()
            totals["total"] += loss.item() * bsz
            totals["wp"] += l_wp.item() * bsz
            totals["temporal"] += l_temporal.item() * bsz
            totals["bev"] += l_bev.item() * bsz
            totals["bev_acc"] += float(bev_acc) * bsz
            totals["cmd"] += l_cmd.item() * bsz
            totals["cmd_acc"] += float(cmd_acc) * bsz

            with torch.no_grad():
                wp_ps = (pred_wp - trajectory).abs().float().mean(dim=(1, 2))
                tmp_ps = (pred_temporal - gt_temporal).abs().float().mean(dim=(1, 2))
                waypoint_ps = wp_ps + tmp_ps

                for c in wp_by_cmd_sum:
                    mask = gt_cmd == c
                    n_c = int(mask.sum())
                    if n_c:
                        wp_by_cmd_sum[c] += float(waypoint_ps[mask].sum())
                        wp_by_cmd_count[c] += n_c

            n_samples += bsz

            progress.set_postfix(
                total=totals["total"] / n_samples, bev_acc=totals["bev_acc"] / n_samples
            )

    metrics = {k: v / n_samples for k, v in totals.items()}
    metrics["wp_by_command"] = {
        c: (wp_by_cmd_sum[c] / wp_by_cmd_count[c]) if wp_by_cmd_count[c] else None
        for c in wp_by_cmd_sum
    }
    metrics["count_by_command"] = dict(wp_by_cmd_count)

    return metrics


def save_curve_data(history: list[dict], runname: str | None, out_path: Path) -> None:
    with open(out_path, "wb") as f:
        pickle.dump({"runname": runname, "history": history}, f)


def plot_curve(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] + 1 for h in history]
    fig, ax = plt.subplots()
    ax.plot(epochs, [h["train"]["total"] for h in history], label="train")

    if history[0]["test"] is not None:
        ax.plot(epochs, [h["test"]["total"] for h in history], label="test")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def plot_qualitative(
    model,
    dataset,
    indices,
    norm_constants,
    device,
    use_mixed_precision,
    out_path: Path,
) -> None:
    samples = [dataset[i] for i in indices]
    main_img = torch.stack([s["main_img"] for s in samples]).to(device)

    model.eval()

    with (
        torch.no_grad(),
        torch.autocast(device_type=device, enabled=use_mixed_precision),
    ):
        command = torch.stack([s["command"] for s in samples]).to(device)
        model_command = perturb_default_commands(command, 1.0)
        pred_wp, pred_temporal, pred_bev, pred_cmd = model(main_img, model_command)
        pred_cmd_idx = pred_cmd.argmax(1).cpu().numpy()
        model_command_idx = model_command.cpu().numpy()

    pred_wp = pred_wp.float().cpu().numpy()
    pred_bev_idx = pred_bev.argmax(1).cpu().numpy()

    pred_temporal = pred_temporal.float().cpu().numpy()
    temporal_constants = {
        "x_min": norm_constants["tx_min"],
        "x_max": norm_constants["tx_max"],
        "y_min": norm_constants["ty_min"],
        "y_max": norm_constants["ty_max"],
    }

    cmap = ListedColormap(_BEV_COLORS_HEX)
    x_margin = 0.1 * (norm_constants["x_max"] - norm_constants["x_min"])
    y_margin = 0.1 * (norm_constants["y_max"] - norm_constants["y_min"])
    forward_lim = (
        norm_constants["x_min"] - x_margin,
        norm_constants["x_max"] + x_margin,
    )
    lateral_lim = (
        norm_constants["y_min"] - y_margin,
        norm_constants["y_max"] + y_margin,
    )
    tx_margin = 0.1 * (temporal_constants["x_max"] - temporal_constants["x_min"])
    ty_margin = 0.1 * (temporal_constants["y_max"] - temporal_constants["y_min"])
    t_forward_lim = (
        temporal_constants["x_min"] - tx_margin,
        temporal_constants["x_max"] + tx_margin,
    )
    t_lateral_lim = (
        temporal_constants["y_min"] - ty_margin,
        temporal_constants["y_max"] + ty_margin,
    )

    fig, axes = plt.subplots(
        len(samples), 4, figsize=(12, 3 * len(samples)), squeeze=False
    )

    for row, sample in enumerate(samples):
        img_rgb = sample["main_img"].numpy()[::-1].transpose(1, 2, 0)
        gt_bev_idx = sample["bev_seg"].numpy()
        gt_wp_m = denormalize(sample["trajectory"].numpy()[None], norm_constants)[0]
        pred_wp_m = denormalize(pred_wp[row][None], norm_constants)[0]

        ax_img, ax_bev, ax_traj, ax_temp = axes[row]
        ax_img.imshow(img_rgb)
        ax_img.axis("off")

        bev_panel = np.concatenate([gt_bev_idx, pred_bev_idx[row]], axis=1)
        ax_bev.imshow(bev_panel, cmap=cmap, vmin=0, vmax=len(_BEV_COLORS_HEX) - 1)
        ax_bev.axvline(gt_bev_idx.shape[1] - 0.5, color="white", linewidth=1)
        ax_bev.axis("off")

        gt_cmd_i = Command(int(sample["command"]))
        pred_cmd_i = Command(int(pred_cmd_idx[row]))
        model_cmd_i = Command(int(model_command_idx[row]))
        gt_color, pred_color = (
            _COMMAND_COLORS_RGB[gt_cmd_i],
            _COMMAND_COLORS_RGB[pred_cmd_i],
        )
        gt_label, pred_label = (
            f"gt ({COMMAND_NAMES[gt_cmd_i]})",
            f"pred ({COMMAND_NAMES[pred_cmd_i]})",
        )

        ax_traj.plot(gt_wp_m[:, 1], gt_wp_m[:, 0], "o-", color=gt_color, label=gt_label)
        ax_traj.plot(
            pred_wp_m[:, 1], pred_wp_m[:, 0], "o-", color=pred_color, label=pred_label
        )
        ax_traj.scatter([0], [0], color="black", marker="*", zorder=5)
        ax_traj.set_xlim(*lateral_lim)
        ax_traj.set_ylim(*forward_lim)
        ax_traj.set_aspect("equal", adjustable="box")

        if model_cmd_i != gt_cmd_i:
            ax_traj.text(
                0.02,
                0.98,
                f"fed: {COMMAND_NAMES[model_cmd_i]}",
                transform=ax_traj.transAxes,
                fontsize=6,
                va="top",
                color=_COMMAND_COLORS_RGB[model_cmd_i],
            )
        
        ax_traj.legend(fontsize=6, loc="upper right")

        gt_temp_m = denormalize(
            sample["temporal_trajectory"].numpy()[None], temporal_constants
        )[0]
        pred_temp_m = denormalize(pred_temporal[row][None], temporal_constants)[0]
        ax_temp.plot(gt_temp_m[:, 1], gt_temp_m[:, 0], "o-", color="grey", label="gt")
        ax_temp.plot(
            pred_temp_m[:, 1], pred_temp_m[:, 0], "o-", color="cyan", label="pred"
        )
        ax_temp.scatter([0], [0], color="black", marker="*", zorder=5)
        ax_temp.set_xlim(*t_lateral_lim)
        ax_temp.set_ylim(*t_forward_lim)
        ax_temp.set_aspect("equal", adjustable="box")
        ax_temp.legend(fontsize=6, loc="upper right")

        if row == 0:
            ax_img.set_title("front cam")
            ax_bev.set_title("BEV: gt | pred")
            ax_traj.set_title("trajectory (m)")
            ax_temp.set_title("temporal (m)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text())
    data_cfg = raw.get("data", {})
    optim_cfg = raw.get("optim", {})
    model_cfg = raw.get("model", {})
    runtime = resolve_runtime(raw)

    loss_fn = loss_duckad
    head = model_cfg.get("head", "gru")
    pretrained = model_cfg.get("pretrained", True)
    print(f"model=duckad head={head} pretrained={pretrained}")

    torch.manual_seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    random.seed(runtime["seed"])
    device = runtime["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is 'cuda', but CUDA is not available")
    use_mixed_precision = device == "cuda"
    print(f"device={device}")

    train_dir = Path(data_cfg["dir"])
    normalization_path = train_dir / "trajectory_norm.json"
    norm_constants = json.loads(normalization_path.read_text())
    background_cache = None
    if data_cfg.get("augment") and data_cfg.get("background_aug"):
        background_cache = ImageNetBackgroundCache(size=data_cfg.get("image_size", 224))
    train_transform = (
        build_train_transform(background_cache) if data_cfg.get("augment") else None
    )
    if train_transform is not None:
        print(
            f"train-image augmentation: on (background replacement: {'on' if background_cache else 'off'})"
        )
    train_dataset = BatchedTrajectoryDataset(
        train_dir,
        normalization_constants=norm_constants,
        transform=train_transform,
    )
    raw.setdefault("data", {})["temporal_dt"] = train_dataset.temporal_dt

    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )

    test_loader = None
    test_dataset = None
    qual_indices = []
    test_dir = data_cfg.get("test_dir")
    if test_dir:
        print(f"test set: {test_dir} (normalized with {normalization_path})")
        test_dataset = BatchedTrajectoryDataset(
            test_dir, normalization_constants=norm_constants
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
        )

    model = DuckAD(
        num_waypoints=train_dataset.num_waypoints,
        bev_classes=train_dataset.bev_classes,
        bev_size=train_dataset.bev_size,
        head=head,
        pretrained=pretrained,
    ).to(device)
    print(f"modelsize={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if test_dataset is not None:
        qual_indices = sample_indices_per_command(
            test_dataset, k=2, seed=runtime["seed"]
        )

    optimizer = build_optimizer(
        model, optim_cfg.get("lr_backbone", 1e-5), optim_cfg.get("lr_head", 1e-4)
    )
    epochs = optim_cfg.get("epochs", 30)
    command_perturb_prob = optim_cfg.get("command_perturb_prob", 0.5)
    stop_prob = optim_cfg.get("imagenet_stop_prob", 0.0)
    stop_aug = None
    if stop_prob > 0:
        stop_aug = ImageNetStopAug(
            stop_prob,
            norm_constants,
            train_dataset.num_waypoints,
            train_dataset.num_temporal_points,
            size=data_cfg.get("image_size", 224),
            seed=runtime["seed"],
        ).to(device)
        print(f"imagenet stop-augmentation: on (prob={stop_prob} of DEFAULT samples)")
    train_bev = optim_cfg.get("train_bev", True)
    loss_fn = loss_fn if train_bev else functools.partial(loss_fn, bev_weight=0.0)
    if not train_bev:
        print(
            "BEV training disabled (optim.train_bev=false): bev_head left at init, noisy BEV at inference"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=use_mixed_precision)

    checkpoint_dir = runtime["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir = checkpoint_dir / "epochs"
    epochs_dir.mkdir(exist_ok=True)

    if runtime["init_weights"]:
        init = torch.load(
            runtime["init_weights"], map_location=device, weights_only=False
        )
        model.load_state_dict(init["model"])
        print(
            f"initialized model weights from {runtime['init_weights']} (fresh optimizer/schedule)"
        )

    history = []

    for epoch in range(epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            use_mixed_precision,
            optimizer=optimizer,
            scaler=scaler,
            desc=f"epoch {epoch + 1}/{epochs} [train]",
            command_perturb_prob=command_perturb_prob,
            stop_aug=stop_aug,
        )
        test_metrics = (
            run_epoch(
                model,
                test_loader,
                loss_fn,
                device,
                use_mixed_precision,
                desc=f"epoch {epoch + 1}/{epochs} [test]",
                command_perturb_prob=1.0,
            )
            if test_loader is not None
            else None
        )
        scheduler.step()

        lrs = [group["lr"] for group in optimizer.param_groups]
        line = (
            f"epoch {epoch + 1}/{epochs}  "
            f"train: total={train_metrics['total']:.4f} wp={train_metrics['wp']:.4f} "
            f"tmp={train_metrics['temporal']:.4f} "
            f"bev={train_metrics['bev']:.4f} acc={train_metrics['bev_acc']:.3f} "
            f"cmd_acc={train_metrics['cmd_acc']:.3f}"
        )
        if test_metrics is not None:
            line += (
                f"  test: total={test_metrics['total']:.4f} wp={test_metrics['wp']:.4f} "
                f"tmp={test_metrics['temporal']:.4f} "
                f"bev={test_metrics['bev']:.4f} acc={test_metrics['bev_acc']:.3f} "
                f"cmd_acc={test_metrics['cmd_acc']:.3f}"
            )
        line += f"  lr(backbone/head)={lrs[0]:.2e}/{lrs[-1]:.2e}"
        print(line)

        history.append({"epoch": epoch, "train": train_metrics, "test": test_metrics})

        if test_dataset is not None:
            plot_qualitative(
                model,
                test_dataset,
                qual_indices,
                norm_constants,
                device,
                use_mixed_precision,
                checkpoint_dir / "qualitative" / f"epoch_{epoch + 1:03d}.png",
            )

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "config": raw,
            "norm_constants": norm_constants,
        }
        torch.save(ckpt, checkpoint_dir / "last.pt")
        torch.save(ckpt, epochs_dir / f"epoch_{epoch + 1:03d}.pt")

        save_curve_data(
            history, runtime["runname"], checkpoint_dir / "training_curve.pkl"
        )

    if history:
        plot_curve(history, checkpoint_dir / "training_curve.png")
        save_curve_data(
            history, runtime["runname"], checkpoint_dir / "training_curve.pkl"
        )
        print(
            f"saved epoch checkpoints + last.pt + training curves to {checkpoint_dir}"
        )


if __name__ == "__main__":
    main()
