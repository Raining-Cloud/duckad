#!/usr/bin/env python3
"""Stop the Duckietown services we don't use, to get the Nano off swap before the model loads.

  free_resources.py             stop the services (what the launcher runs)
  free_resources.py --restore   start them all again
"""
import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DOCKER_API = "http://localhost:2375"
TIMEOUT_S = 15

# Dead weight for autonomous driving. Stopping these frees ~590 MB on a stock boot.
EXPENDABLE = [
    "ros1-rosbridge-websocket",
    "driver-display",
    "driver-imu",
    "driver-tof",
    "driver-lights",
    "control-mapper",
    # Stopping these takes http://<robot>.local/ offline; SSH and dts are unaffected.
    "dashboard",
    "code-api",
    "files-api",
    "portainer",
    "device-online",
    "device-health",
]

# Do NOT add these:
#   driver-camera, driver-wheels, car-interface -> the robot's eyes and legs
#   ros-interface                               -> bridges the DTPS drivers onto ROS1 topics;
#                                                  without it our camera topic doesn't exist
#   dtps, kvstore, device-proxy                 -> the transport those drivers speak


def _mem_free_mb():
    try:
        with open("/proc/meminfo") as f:
            info = {k.strip(): int(v.split()[0]) for k, v in
                    (line.split(":", 1) for line in f)}
        swap_used = info["SwapTotal"] - info["SwapFree"]
        return info["MemAvailable"] // 1024, swap_used // 1024
    except Exception:
        return -1, -1


def _post(path):
    req = urllib.request.Request(DOCKER_API + path, method="POST")
    return urllib.request.urlopen(req, timeout=TIMEOUT_S).status


def _act(name, action):
    try:
        _post("/containers/{}/{}".format(name, action))
        return "ok"
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return "already " + ("stopped" if action == "stop" else "running")
        if exc.code == 404:
            return "not present"
        return "HTTP {}".format(exc.code)
    except Exception as exc:
        return "failed ({})".format(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true",
                        help="start the services again instead of stopping them")
    args = parser.parse_args()
    action = "start" if args.restore else "stop"

    before_avail, before_swap = _mem_free_mb()
    print("[free_resources] {} {} services (available: {} MB, swap in use: {} MB)".format(
        "starting" if args.restore else "stopping", len(EXPENDABLE), before_avail, before_swap))

    # Issued in parallel, not one after another: on shutdown this runs inside `docker stop`'s
    # grace period (~10 s by default) and a dozen sequential round-trips can overrun it, which
    # would leave half the services down. Printed in list order afterwards so the log stays stable.
    with ThreadPoolExecutor(max_workers=len(EXPENDABLE)) as pool:
        outcomes = list(pool.map(lambda n: _act(n, action), EXPENDABLE))
    for name, outcome in zip(EXPENDABLE, outcomes):
        print("[free_resources]   {:<26} {}".format(name, outcome))

    after_avail, after_swap = _mem_free_mb()
    if before_avail >= 0 and after_avail >= 0:
        print("[free_resources] available: {} -> {} MB ({:+d} MB), swap in use: {} -> {} MB".format(
            before_avail, after_avail, after_avail - before_avail, before_swap, after_swap))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:   # never let a cleanup failure stop the robot from launching
        print("[free_resources] skipped: {}".format(exc))
        sys.exit(0)
