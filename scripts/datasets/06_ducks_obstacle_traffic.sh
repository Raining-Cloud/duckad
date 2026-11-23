#!/usr/bin/env bash
# Cell (ducks + traffic) - OBSTACLE variant: on-road ducks (ego stops) + background traffic.
# The fullest scene: junction rules + vehicle yielding + duck-stopping together.
source "$(dirname "$0")/common.sh"

WITH_TRAFFIC=1 record ducks_obstacle_traffic --duckie-obstacle \
  --duckie-min-separation "$DUCKIE_OBSTACLE_SEPARATION" \
  --duckie-wait-timeout "$DUCKIE_WAIT_TIMEOUT"
