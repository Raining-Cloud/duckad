#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------

# Free up the Nano BEFORE onnxruntime starts: a stock boot leaves the robot swapping, and the
# model needs ~1 GB. Run with --restore to bring the services back, DUCKAD_KEEP_SERVICES=1 to skip.
# Locate via rospack, NOT DT_REPO_PATH: that is baked from the template's repo name at image-build
# time, while `dts devel run` mounts this repo at /code/src/<repo>.
if [ "${DUCKAD_KEEP_SERVICES}" != "1" ]; then
    FREE_RESOURCES="$(rospack find model_node 2>/dev/null)/scripts/free_resources.py"
    if [ -f "${FREE_RESOURCES}" ]; then
        # Bring the services back on the way out, whatever ends us: a normal exit, or the SIGINT/
        # SIGTERM that `docker stop` / Ctrl-C sends. dt-register-signals only traps INT and TERM,
        # so EXIT is ours to use, and it still runs after their dt-terminate handler has fired.
        # Registered BEFORE the stop, so a run that dies mid-stop is still restored.
        trap 'echo "[free_resources] restoring services on shutdown"; \
              python3 "${FREE_RESOURCES}" --restore || true' EXIT
        python3 "${FREE_RESOURCES}" || true
    else
        echo "[free_resources] not found at ${FREE_RESOURCES} - skipping (robot keeps all services)"
    fi
fi

# Launch the full driving stack against the roscore already running on the
# robot. drive.launch (in controller_node) includes all three nodes:
#   - model_node       : runs the ONNX model on the camera feed
#   - controller_node  : turns model predictions into car commands
#   - web_interface    : FastAPI dashboard on :8090 (camera + WASD driving)
# --wait attaches to the existing master instead of starting a private one.
# NOTE: roslaunch takes ONE "<pkg> <file>" pair (or a list of files), not
# several pkg/file pairs -- hence the single combined launch file.
dt-exec roslaunch --wait controller_node drive.launch

# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
