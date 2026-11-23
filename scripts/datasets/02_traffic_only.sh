#!/usr/bin/env bash
# Cell (traffic, no ducks): expert ego + background traffic, no ducks.
# Decorrelates "other vehicles present" from "duck present" so the model doesn't
# entangle vehicle-yielding with duck-stopping.
source "$(dirname "$0")/common.sh"

WITH_TRAFFIC=1 record traffic_only
