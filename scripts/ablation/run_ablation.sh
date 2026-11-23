#!/usr/bin/env bash
#
# Ablation study: 5 cells x 1 seed = 5 runs, 15 epochs. Run from the repo root:
#     ./scripts/ablation/run_ablation.sh
#     ABL_SEEDS="1 2 3" ABL_EPOCHS=10 ABL_WORKERS=6 ABL_CONCURRENCY=4 ./scripts/ablation/run_ablation.sh
#
# Each run writes checkpoints/<cell>-s<seed>-<timestamp>/ with epochs/, last.pt and
# training_curve.pkl.
set -euo pipefail

SEEDS="${ABL_SEEDS:-8743}"                       # add seeds for mean +/- std across runs
EPOCHS="${ABL_EPOCHS:-15}"
WORKERS="${ABL_WORKERS:-5}"
CORES="$(nproc)"
# Augmentation-bound on CPU, so concurrency follows cores, not GPUs.
CONCURRENCY="${ABL_CONCURRENCY:-$(( CORES / (WORKERS + 1) ))}"
[ "$CONCURRENCY" -lt 1 ] && CONCURRENCY=1
LOG_ROOT="${ABL_LOGS:-logs/ablation}"
TRAIN_DIR=data/final
TEST_DIR=data/test

NUM_GPUS="${ABL_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
[ "${NUM_GPUS:-0}" -lt 1 ] && NUM_GPUS=1

echo ">>> cores=$CORES workers/run=$WORKERS concurrent=$CONCURRENCY epochs=$EPOCHS gpus=$NUM_GPUS"
echo ">>> seeds: $SEEDS"

# --- preflight ---
[ -d "$TRAIN_DIR" ] || { echo "ERROR: $TRAIN_DIR missing - run ./scripts/setup/fetch_data.sh"; exit 1; }
if [ "$(find imagenet-val -name '*.JPEG' 2>/dev/null | wc -l)" -lt 100 ]; then
  echo "ERROR: imagenet-val/ has no *.JPEG - background- and stop-augmentation would crash."
  echo "    ./scripts/setup/fetch_imagenet_kaggle.sh"
  exit 1
fi
if [ ! -f "$TRAIN_DIR/trajectory_norm.json" ]; then
  echo ">>> fitting normalization constants on $TRAIN_DIR ..."
  uv run python -u -m src.aggregation.data.normalize "$TRAIN_DIR" --clip-percentile 1
fi

# --- configs ---
mkdir -p "$LOG_ROOT"
RUNS="$(uv run python scripts/ablation/gen_configs.py --seeds $SEEDS --epochs "$EPOCHS" --num-workers "$WORKERS")"
TOTAL="$(echo "$RUNS" | wc -l)"
echo ">>> generated $TOTAL configs in configs/ablation/"

# --- pre-warm the memmap cache ---
# Must happen once, single-process: parallel runs would race building the same ~41GB cache and a
# late starter can mmap a half-written one.
echo ">>> pre-warming the memmap cache (slow, once) ..."
TRAIN_DIR="$TRAIN_DIR" TEST_DIR="$TEST_DIR" uv run python -u - <<'PY'
import json
import os
from src.aggregation.data.dataset import BatchedTrajectoryDataset as Dataset

train_dir, test_dir = os.environ["TRAIN_DIR"], os.environ["TEST_DIR"]
norm = json.loads(open(f"{train_dir}/trajectory_norm.json").read())
Dataset(train_dir, normalization_constants=norm)
if os.path.isdir(test_dir):
    Dataset(test_dir, normalization_constants=norm)
print(">>> memmap cache ready")
PY

# --- launch (rolling pool: start a new run as soon as a slot frees) ---
started=0
for name in $RUNS; do
  while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
  gpu=$(( started % NUM_GPUS ))
  started=$(( started + 1 ))
  echo ">>> [$started/$TOTAL] start $name on gpu $gpu  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python -u -m src.training.train "configs/ablation/$name.yaml" \
      > "$LOG_ROOT/$name.log" 2>&1 &
done
wait

echo ">>> ALL $TOTAL RUNS DONE $(date +%H:%M:%S) - logs in $LOG_ROOT/"
