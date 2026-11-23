#!/usr/bin/env bash
# Run the whole final-model collection matrix, in order. Run from the repo root:
#     ./scripts/datasets/run_all.sh
set -euo pipefail
here="$(dirname "$0")"

"$here/01_baseline.sh"
"$here/02_traffic_only.sh"
"$here/03_ducks_roadside.sh"
"$here/04_ducks_obstacle.sh"
"$here/05_ducks_roadside_traffic.sh"
"$here/06_ducks_obstacle_traffic.sh"

echo "All conditions collected under data/final/"
