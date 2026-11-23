#!/usr/bin/env bash
#
# Closed-loop junction eval of every ablation cell on the held-out map+lighting combo.
#
# A route = one eval seed: seeded spawn point + seeded LEFT/RIGHT command sequence, ended after
# 3 junctions - so all checkpoints face the identical routes. Needs a running CARLA server.
#     ./scripts/ablation/eval_junctions.sh
#     ABL_CELLS="base mlp" STEER_NOISE=0.15 ./scripts/ablation/eval_junctions.sh
set -euo pipefail

SEEDS="${ABL_SEEDS:-8743}"                       # model (training) seeds
EVAL_SEEDS="${EVAL_SEEDS:-0-2}"                  # eval seeds = the routes
CELLS="${ABL_CELLS:-base mlp nopretrain nobg noaug}"
CHECKPOINT_ROOT="checkpoints"
MAP="${EVAL_MAP:-duckietown_06}"
HDRI="${EVAL_HDRI:-airport}"
STEER_NOISE="${STEER_NOISE:-0}"
TARGET_SPEED="${TARGET_SPEED:-}"                 # km/h; set to hold a constant speed
SUFFIX=""
[ "$STEER_NOISE" != "0" ] && SUFFIX="_noise$STEER_NOISE"
SPEED_ARGS=()
if [ -n "$TARGET_SPEED" ]; then
  SPEED_ARGS=(--no-use-model-speed --target-speed "$TARGET_SPEED")
  SUFFIX="${SUFFIX}_v$TARGET_SPEED"
fi
JSON_OUT="${JSON_OUT:-$CHECKPOINT_ROOT/eval/junctions_${MAP#duckietown_}_$HDRI$SUFFIX.json}"

CHECKPOINTS=()
for cell in $CELLS; do
  for seed in $SEEDS; do
    matches=("$CHECKPOINT_ROOT/$cell-s$seed-"*/last.pt)
    [ -e "${matches[0]}" ] || { echo "ERROR: no checkpoint matching $CHECKPOINT_ROOT/$cell-s$seed-*/last.pt"; exit 1; }
    ckpt="${matches[0]}"
    for candidate in "${matches[@]}"; do
      [ "$candidate" -nt "$ckpt" ] && ckpt="$candidate"
    done
    CHECKPOINTS+=("$ckpt")
  done
done

mkdir -p "$(dirname "$JSON_OUT")"
echo ">>> ${#CHECKPOINTS[@]} checkpoints x routes $EVAL_SEEDS on $MAP/$HDRI -> $JSON_OUT"

uv run python -u -m src.eval.closed_loop "${CHECKPOINTS[@]}" \
  --map "$MAP" \
  --hdri-preset "$HDRI" \
  --junction \
  --max-junctions 3 \
  --steer-noise "$STEER_NOISE" \
  ${SPEED_ARGS[@]+"${SPEED_ARGS[@]}"} \
  --seeds "$EVAL_SEEDS" \
  --json-out "$JSON_OUT"
