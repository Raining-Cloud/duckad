#!/usr/bin/env bash
# Cell (ducks, no traffic) - OBSTACLE variant: ducks on the lane centreline.
# The expert stops for them, so the speed/trajectory labels encode
# duck-stopping. (--duckie-obstacle implies --duckies and ignores lateral offset.)
source "$(dirname "$0")/common.sh"

record ducks_obstacle --duckie-obstacle \
  --duckie-min-separation "$DUCKIE_OBSTACLE_SEPARATION" \
  --duckie-wait-timeout "$DUCKIE_WAIT_TIMEOUT"
