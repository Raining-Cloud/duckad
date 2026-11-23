#!/usr/bin/env bash
# Cell (ducks + traffic) - ROADSIDE variant: scenery ducks + background traffic.
source "$(dirname "$0")/common.sh"

WITH_TRAFFIC=1 record ducks_roadside_traffic \
  --duckies --duckie-lateral-offset "${ROADSIDE_OFFSET[@]}"
