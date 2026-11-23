#!/usr/bin/env bash
# Cell (ducks, no traffic) - ROADSIDE variant: ducks off the driving line as scenery.
# Visual domain realism only; the ego drives past normally (no behaviour change).
source "$(dirname "$0")/common.sh"

record ducks_roadside --duckies --duckie-lateral-offset "${ROADSIDE_OFFSET[@]}"
