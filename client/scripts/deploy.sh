#!/usr/bin/env bash
# Build and run the stack on the robot, syncing the weights first.
#
# This is the entry point to use instead of calling `dts devel build` directly: the weights now
# live on the robot's /data partition rather than in the image, and this is what keeps them in
# step. sync_models.py compares by SHA-256, so the sync is a no-op costing one round trip when
# nothing changed, and re-exports a couple of models without touching the image.
#
#   scripts/deploy.sh                 sync, build, run
#   scripts/deploy.sh --no-run        sync and build only
#   scripts/deploy.sh --sync-only     just push weights (no rebuild -- use after a re-export)
#   DUCKAD_ROBOT=192.168.1.5 scripts/deploy.sh
set -euo pipefail

ROBOT="${DUCKAD_ROBOT:-canards.local}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=1
BUILD=1

for arg in "$@"; do
    case "$arg" in
        --no-run)    RUN=0 ;;
        --sync-only) RUN=0; BUILD=0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

echo "==> syncing weights to ${ROBOT}"
python3 "${HERE}/sync_models.py" --robot "${ROBOT}"

if [ "${BUILD}" = "1" ]; then
    echo "==> dts devel build -H ${ROBOT}"
    dts devel build -H "${ROBOT}"
fi

if [ "${RUN}" = "1" ]; then
    echo "==> dts devel run -H ${ROBOT}"
    dts devel run -H "${ROBOT}"
fi
