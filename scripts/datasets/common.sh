#!/usr/bin/env bash
#
# Shared config + runner for the final-model data collection.
#
# IMPORTANT: run every script from the repo root, so `python -m src.aggregation.collection.collect`
# resolves (the package is imported as `src.aggregation.*`):
#
#     ./scripts/datasets/01_baseline.sh
#     ./scripts/datasets/run_all.sh          # everything, in order
#
# All conditions are driven by the rule-aware expert so the recorded
# speed/trajectory labels reflect the behaviour we want the final model to imitate
# (junction rules, stopping for on-road ducks). Every condition sweeps the same
# maps with no HDRI backdrop (the map's own default lighting).
set -euo pipefail

# --- knobs -------------------------------------------------------------------
# Maps to sweep. Override for a partial re-collection, e.g. to redo one map:
#   DATASET_MAPS="duckietown_04" ./scripts/datasets/02_traffic_only.sh
# duckietown_02 dropped from the default sweep: no junctions -> 100% DEFAULT commands (it inflated
# the default-heavy bias), and its lane-following is already covered by the straights on 04/05.
# Re-add it via the DATASET_MAPS override if wanted (its 20s duration is still in DURATION_BY_MAP).
MAPS=(${DATASET_MAPS:-duckietown_04 duckietown_05})
HDRI=defaulthdri   # label only: --hdri-preset is omitted, so the map keeps its default backdrop
SEGMENTS=10          # respawns at a fresh start point each segment -> spatial coverage
DURATION=60         # seconds of sim time per segment  (8 x 60s = 8 min / map / condition)
# Per-map overrides of DURATION (map -> seconds). duckietown_02 is a shorter loop, so 20s;
# 04 and 05 run 40s.
declare -A DURATION_BY_MAP=( [duckietown_02]=20 [duckietown_04]=30 [duckietown_05]=30 )
# Per-map extra collect flags: 04 and 05 spawn near junctions (--junction) for denser balanced
# LEFT/STRAIGHT/RIGHT command coverage. (The old map04 --junction stall is fixed: traffic no
# longer spawns onto the ego's junction-approach points but staggers randomly around them, so
# the fleet can't pile into the same intersections at t=0. Junction entry is also serialized
# by a shared reservation in DuckieBehaviorAgent.)
declare -A EXTRA_FLAGS_BY_MAP=( [duckietown_04]="--junction" [duckietown_05]="--junction" )
DATA_ROOT=data/final
# Temporal ("speed") trajectory sample rate, applied to EVERY condition. Asking 3.33Hz rounds to
# stride 6 sim ticks -> dt=0.30s (effective 3.33Hz) and a 3.0s horizon over 10 points. The wider
# spacing spreads the future points out, making stopped-vs-moving clearer, and the 3.0s horizon is
# what DUCKIE_WAIT_TIMEOUT below is tuned against.
TEMPORAL_HZ=3.33
# Camera-yaw augmentation (deg), applied to EVERY condition. Each segment, each rig's cameras are
# mounted at a random yaw in [-DEG, +DEG] about vertical and that rig's trajectory + BEV are
# expressed in the same yawed frame - so the model sees the lane/scene at a spread of apparent
# angles (robustness to camera-mount yaw / off-heading). 0 disables it (cameras vehicle-aligned).
CAMERA_YAW_JITTER=15
# Background expert-driven duckiebots for the traffic conditions. Overridable via env for a
# partial re-collect, e.g. TRAFFIC_N=2 DATASET_MAPS="duckietown_04" ./scripts/datasets/02_traffic_only.sh
TRAFFIC_N="${TRAFFIC_N:-3}"
# Per-map traffic-count overrides (map -> N), taking precedence over TRAFFIC_N. Only consulted by
# the traffic conditions (scripts that set WITH_TRAFFIC=1 - see record()). map04 runs 4 bots.
declare -A TRAFFIC_BY_MAP=( [duckietown_04]=4 )

# Roadside-duck lateral offset (MIN MAX, metres from lane centre). MIN also sets the
# clearance kept from every lane centre, so ducks stay OFF the driving line. MIN is kept
# well past the agent's 1.0m DUCKIE_LATERAL_TOLERANCE so the expert never stops for them
# (checked by driving through them: 864m driven, 0 reactions) - pure visual realism.
ROADSIDE_OFFSET=(3 5)
# [obstacle only] seconds the expert holds at an on-road duck before despawning it and driving on.
# Set just above the 3.0s temporal horizon (10 pts @ 3.33Hz) so each duck stop drives the temporal
# trajectory all the way to a FULL collapse (every point at 0 = a genuine dead stop) for ~1-2 ticks
# - teaching a true hold, not just "slow with a forward tail". Keep ducks sparse (see separation
# below) so these standstill frames stay a minority.
DUCKIE_WAIT_TIMEOUT=3.1
# [obstacle only] spacing (m) between on-road ducks = obstacle density. The 2m default paves the
# centreline (~32% stopped, near-zero cruising - measured); 20m gives mostly free-driving with
# occasional genuine stops (~3-4% stopped).
DUCKIE_OBSTACLE_SEPARATION=20
# -----------------------------------------------------------------------------

# record <condition-subdir> [extra collect flags...]
# Runs the collector once per map, writing to $DATA_ROOT/<condition>/<map>_<hdri>_<ts>/
# so each condition is its own top-level folder.
record() {
  local condition="$1"; shift
  for map in "${MAPS[@]}"; do
    local duration="${DURATION_BY_MAP[$map]:-$DURATION}"
    # shellcheck disable=SC2206  # deliberate word-split of the per-map flag string
    local map_flags=( ${EXTRA_FLAGS_BY_MAP[$map]:-} )
    # Traffic conditions invoke record with WITH_TRAFFIC=1 (instead of passing --traffic
    # themselves), so the bot count can be resolved per map here.
    if [ "${WITH_TRAFFIC:-0}" = "1" ]; then
      map_flags+=( --traffic "${TRAFFIC_BY_MAP[$map]:-$TRAFFIC_N}" )
    fi
    echo "=== [$condition] ${map} @ ${HDRI} : ${SEGMENTS}x${duration}s ${map_flags[*]} ==="
    # Guard each run: a transient CARLA stall (server unresponsive > client timeout) crashes one
    # map, but must NOT abort the remaining maps/conditions. Log it and carry on so a single hiccup
    # costs one run, not the rest of the batch.
    if ! uv run python -m src.aggregation.collection.collect \
      --map "$map" \
      --segments "$SEGMENTS" \
      --duration "$duration" \
      --temporal-hz "$TEMPORAL_HZ" \
      --camera-yaw-jitter "$CAMERA_YAW_JITTER" \
      --out "$DATA_ROOT/$condition" \
      "${map_flags[@]}" \
      "$@"; then
      echo "!!! [$condition] ${map} FAILED (exit $?) - continuing to next map" >&2
    fi
  done
}
