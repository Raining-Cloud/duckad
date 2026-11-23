"""Generate the ablation configs: 5 cells x N seeds (default 1).

Cells (one factor off the full baseline: GRU + BEV + aug + background swap + pretrained):
    base        full model (baseline)
    mlp         MLP waypoint head instead of the autoregressive GRU
    nopretrain  MobileNet trained from scratch
    nobg        ImageNet background drop-ins off (photometric aug still on)
    noaug       all image augmentation off

    python scripts/ablation/gen_configs.py --seeds 8743 --num-workers 5
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml

CELLS = {
    "base":       {},
    "mlp":        {"model": {"head": "mlp"}},
    "nopretrain": {"model": {"pretrained": False}},
    "nobg":       {"data": {"background_aug": False}},
    "noaug":      {"data": {"augment": False, "background_aug": False}},
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default="configs/train_gru.yaml", help="config to derive the cells from")
    p.add_argument("--out-dir", default="configs/ablation")
    p.add_argument("--seeds", type=int, nargs="+", default=[8743])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=5, help="dataloader workers PER RUN")
    args = p.parse_args()

    base = yaml.safe_load(Path(args.base).read_text())
    os.makedirs(args.out_dir, exist_ok=True)

    # A missing test_dir would crash every run at startup, so drop it rather than fail 20 runs deep.
    test_dir = base.get("data", {}).get("test_dir")
    if test_dir and not os.path.isdir(test_dir):
        print(f"WARNING: test_dir '{test_dir}' not found -> setting test_dir: null")
        base["data"]["test_dir"] = None

    names = []
    for cell, patch in CELLS.items():
        for seed in args.seeds:
            name = f"{cell}-s{seed}"
            c = copy.deepcopy(base)
            c["optim"]["epochs"] = args.epochs
            c["data"]["num_workers"] = args.num_workers
            c["runtime"]["seed"] = seed
            c["runtime"]["runname"] = name
            for section, kv in patch.items():
                c.setdefault(section, {}).update(kv)
            Path(args.out_dir, f"{name}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
            names.append(name)

    # stdout is consumed by run_ablation.sh as the run list - keep it to bare names.
    print("\n".join(names))


if __name__ == "__main__":
    main()
