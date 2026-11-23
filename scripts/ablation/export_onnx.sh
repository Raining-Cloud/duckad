#!/usr/bin/env bash
#
# Export every ablation checkpoint to ONNX, named after its cell. Run from the repo root:
#     ./scripts/ablation/export_onnx.sh
#     ABL_SEEDS="8743 1553" ABL_ONNX_DIR=exports ./scripts/ablation/export_onnx.sh
#
# Writes <out dir>/<cell>.onnx (base.onnx, mlp.onnx, ...) from each run's last.pt. With several
# seeds per cell the name carries the seed too, so runs don't overwrite each other.
set -euo pipefail

SEEDS="${ABL_SEEDS:-8743}"
CKPT_ROOT="${ABL_CKPTS:-checkpoints}"
OUT_DIR="${ABL_ONNX_DIR:-checkpoints/onnx}"
MULTI_SEED=$([ "$(echo $SEEDS | wc -w)" -gt 1 ] && echo 1 || echo 0)

mkdir -p "$OUT_DIR"
exported=0
for seed in $SEEDS; do
  for dir in "$CKPT_ROOT"/*-s"$seed"-*/; do
    [ -f "$dir/last.pt" ] || { echo ">>> skip $dir (no last.pt)"; continue; }
    cell="$(basename "$dir")"; cell="${cell%%-s$seed-*}"
    name="$cell"; [ "$MULTI_SEED" = 1 ] && name="$cell-s$seed"
    echo ">>> $name  <-  $dir"
    uv run python -m src.onnx.export_onnx "$dir/last.pt" -o "$OUT_DIR/$name.onnx"
    exported=$(( exported + 1 ))
  done
done

echo ">>> exported $exported model(s) to $OUT_DIR/"
