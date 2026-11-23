#!/usr/bin/env bash
#
# Download the training data (+ held-out test set) from the Hugging Face Hub and rebuild the
# npz shards the training code reads (see scripts/datasets/import_hf.py):
#     train split (~12GB) -> data/final/  (214,200 frames)
#     test split  (~0.7GB) -> data/test/   (10,200 frames, held-out map06/airport set)
#
# The ~41GB memmap cache is deliberately NOT part of the download - the dataset rebuilds it on
# first load, so make sure the node has ~70GB of disk.
#
#     ./scripts/setup/fetch_data.sh
#     DATA_HF_REPO=<user>/<dataset> ./scripts/setup/fetch_data.sh
set -euo pipefail

REPO="${DATA_HF_REPO:-pamasan/duckad-data}"

shards() { find "$1" -name 'batch_*.npz' 2>/dev/null | wc -l; }

if [ "$(shards data/final)" -gt 0 ]; then
  echo ">>> data/final already present ($(shards data/final) shards) - skipping download."
else
  command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not found - see https://docs.astral.sh/uv/"; exit 1; }

  echo ">>> downloading $REPO from the Hugging Face Hub and converting to npz shards ..."
  uv run --no-project --python 3.12 \
      --with huggingface_hub --with pyarrow --with pillow --with "numpy>=2" \
      scripts/datasets/import_hf.py --repo "$REPO" --train-dir data/final --test-dir data/test
fi

# The Hub dataset does not carry the normalization constants; training needs them.
if [ ! -f data/final/trajectory_norm.json ]; then
  echo ">>> fitting normalization constants on data/final ..."
  uv run python -u -m src.aggregation.data.normalize data/final --clip-percentile 1
fi

# --- verify ------------------------------------------------------------------------------------
echo "=============================================================="
n="$(shards data/final)"
echo " data/final: $n shards"
[ "$n" -gt 0 ] || { echo " ERROR: data/final has no shards"; exit 1; }

[ -f data/final/trajectory_norm.json ] || { echo " ERROR: data/final/trajectory_norm.json missing"; exit 1; }
echo " trajectory_norm.json: present"

t="$(shards data/test)"
if [ "$t" -gt 0 ]; then
  echo " data/test: $t shards (per-epoch cross-map test loss enabled)"
else
  echo " data/test: absent -> configs get test_dir: null (no held-out signal!)"
fi
echo " free disk: $(df -h . | tail -1 | awk '{print $4}')  (need ~70GB: data + ~41GB memmap cache)"
echo "=============================================================="
