#!/usr/bin/env python3
"""Mirror the ONNX weights onto the robot's persistent /data partition.

The weights are deliberately NOT in the Docker image. 77 MB of ONNX through the build context
over the robot's WiFi is what used to dominate `dts devel build`, and a dropped association
kills BuildKit's context stream outright ("transport is closing") with no way to resume. /data
is bind-mounted into Duckietown containers and survives image rebuilds, so the weights only
ever need to get there once; after that a build ships ~300 KB of Python.

Only what actually changed is uploaded -- files are compared by size and SHA-256, so re-running
this is cheap and safe. Weights that no longer exist locally are removed from the robot unless
--no-delete is given, so keep anything that is not a mirrored weight out of the destination.

Two transports, picked automatically:
  ssh     rsync over SSH. Preferred: it resumes a part-transferred file, which matters a great
          deal on a link that drops mid-copy.
  docker  tar over the robot's Docker API on :2375, writing through a container that already
          has /data mounted. No resume, but it needs no SSH key -- which is the situation on a
          fresh laptop. The API is unauthenticated and root-equivalent, so this is a
          trusted-LAN-only path; use ssh when you have it.

  scripts/sync_models.py                      mirror onto $DUCKAD_ROBOT (default canards.local)
  scripts/sync_models.py --robot 192.168.1.5  somewhere else
  scripts/sync_models.py --dry-run            say what would move, touch nothing
  scripts/sync_models.py --transport docker   force the API path
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(REPO_ROOT, "packages", "model_node", "assets")
DEFAULT_DEST = "/data/duckad/models"
DEFAULT_ROBOT = os.environ.get("DUCKAD_ROBOT", "canards.local")
SSH_USER = os.environ.get("DUCKAD_SSH_USER", "duckie")
DOCKER_PORT = 2375
# /data is owned by uid 1000 on a stock robot; containers write as root, so hand the files back
# afterwards or anything running unprivileged cannot read them.
DATA_UID_GID = "1000:1000"

# Long, because the link this runs over is the problem being worked around. A 13 MB model at the
# ~450 KB/s this WiFi actually sustains is ~30 s; allow an order of magnitude for a bad patch.
HTTP_TIMEOUT_S = 300
SSH_TIMEOUT_S = 30
DEFAULT_RETRIES = 3


# --------------------------------------------------------------------------- local side

def local_models(source):
    try:
        names = sorted(f for f in os.listdir(source) if f.endswith(".onnx"))
    except OSError as exc:
        raise SystemExit("cannot read {}: {}".format(source, exc))
    if not names:
        raise SystemExit("no .onnx files in {} - nothing to sync".format(source))
    out = {}
    for name in names:
        path = os.path.join(source, name)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        out[name] = (os.path.getsize(path), digest.hexdigest())
    return out


def plan(local, remote, delete):
    """(to_upload, to_delete, unchanged) -- remote maps name -> (size, sha256) or None if unknown."""
    upload, unchanged = [], []
    for name, fingerprint in sorted(local.items()):
        if remote.get(name) == fingerprint:
            unchanged.append(name)
        else:
            upload.append(name)
    stale = sorted(set(remote) - set(local)) if delete else []
    return upload, stale, unchanged


# --------------------------------------------------------------------------- ssh transport

def ssh_base(robot):
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "StrictHostKeyChecking=accept-new",
            "{}@{}".format(SSH_USER, robot)]


def ssh_run(robot, command, check=True):
    result = subprocess.run(ssh_base(robot) + [command], capture_output=True, text=True,
                            timeout=SSH_TIMEOUT_S)
    if check and result.returncode != 0:
        raise RuntimeError("ssh failed ({}): {}".format(result.returncode, result.stderr.strip()))
    return result


def ssh_available(robot):
    if shutil.which("rsync") is None:
        print("[sync] no local rsync -> falling back to the Docker API")
        return False
    try:
        probe = ssh_run(robot, "command -v rsync >/dev/null && echo RSYNC_OK", check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        print("[sync] ssh unusable ({}) -> falling back to the Docker API".format(exc))
        return False
    if probe.returncode != 0:
        print("[sync] ssh rejected us ({}) -> falling back to the Docker API".format(
            (probe.stderr or "").strip().splitlines()[-1:] or "no key?"))
        return False
    if "RSYNC_OK" not in probe.stdout:
        print("[sync] robot has no rsync -> falling back to the Docker API")
        return False
    return True


def ssh_remote_fingerprints(robot, dest):
    # One round trip for every hash: the link is the expensive part, not the Nano's CPU.
    script = ("mkdir -p {d} && cd {d} && "
              "for f in *.onnx; do [ -e \"$f\" ] || continue; "
              "printf '%s %s\\n' \"$(stat -c %s \"$f\")\" \"$(sha256sum \"$f\")\"; done").format(d=dest)
    out = ssh_run(robot, script).stdout
    return parse_fingerprints(out)


def ssh_upload(robot, source, dest, names, stale, retries):
    ssh_run(robot, "mkdir -p {}".format(dest))
    # --partial is the point of preferring this transport: a copy killed by the dongle dropping
    # resumes instead of restarting. Deletes are handled separately so that a failed upload
    # cannot take a good remote file down with it.
    command = ["rsync", "-a", "--partial", "--human-readable", "--progress",
               "--timeout=60"] + \
              [os.path.join(source, n) for n in names] + \
              ["{}@{}:{}/".format(SSH_USER, robot, dest)]
    for attempt in range(1, retries + 1):
        print("[sync] rsync attempt {}/{}: {}".format(attempt, retries, " ".join(names)))
        if subprocess.run(command).returncode == 0:
            break
        if attempt == retries:
            raise RuntimeError("rsync gave up after {} attempts".format(retries))
        time.sleep(2 * attempt)
    for name in stale:
        print("[sync] removing stale {}".format(name))
        ssh_run(robot, "rm -f {}/{}".format(dest, name))
    ssh_run(robot, "chown -R {} {} 2>/dev/null || true".format(DATA_UID_GID, dest), check=False)


# --------------------------------------------------------------------------- docker transport

def api(robot, path):
    return "http://{}:{}{}".format(robot, DOCKER_PORT, path)


def http(robot, path, method="GET", body=None, content_type=None, retries=DEFAULT_RETRIES,
         timeout=HTTP_TIMEOUT_S):
    last = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(api(robot, path), method=method, data=body)
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:                      # noqa: BLE001 - flaky link, retry anything
            last = exc
            if attempt < retries:
                print("[sync]   {} {} failed ({}), retrying".format(method, path.split("?")[0], exc))
                time.sleep(2 * attempt)
    raise RuntimeError("{} {} failed after {} attempts: {}".format(method, path, retries, last))


def docker_data_container(robot):
    """A running container with /data bind-mounted rw -- our window onto the robot's disk."""
    containers = json.loads(http(robot, "/containers/json"))
    for entry in containers:
        name = entry["Names"][0].lstrip("/")
        details = json.loads(http(robot, "/containers/{}/json".format(name)))
        for mount in details.get("Mounts") or []:
            if mount.get("Destination") == "/data" and mount.get("RW"):
                return name
    raise RuntimeError("no running container has /data mounted rw - is the robot fully booted?")


def docker_exec(robot, container, command):
    spec = json.dumps({"AttachStdout": True, "AttachStderr": True,
                       "Cmd": ["sh", "-c", command]}).encode()
    exec_id = json.loads(http(robot, "/containers/{}/exec".format(container), "POST", spec,
                              "application/json"))["Id"]
    raw = http(robot, "/exec/{}/start".format(exec_id), "POST",
               json.dumps({"Detach": False, "Tty": True}).encode(), "application/json")
    return raw.decode("utf-8", "replace")


def docker_remote_fingerprints(robot, container, dest):
    out = docker_exec(robot, container, (
        "mkdir -p {d} && cd {d} && "
        "for f in *.onnx; do [ -e \"$f\" ] || continue; "
        "printf '%s %s\\n' \"$(stat -c %s \"$f\")\" \"$(sha256sum \"$f\")\"; done").format(d=dest))
    return parse_fingerprints(out)


def docker_upload(robot, container, source, dest, names, stale):
    for name in names:
        path = os.path.join(source, name)
        buffer = io.BytesIO()
        # Docker's archive endpoint takes a tar and unpacks it at ?path=. One file per request:
        # a dropped connection then costs 13 MB of retry, not the whole set.
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            archive.add(path, arcname=name)
        payload = buffer.getvalue()
        print("[sync] uploading {} ({:.1f} MB) via Docker API".format(name, len(payload) / 1e6))
        http(robot, "/containers/{}/archive?path={}".format(container, dest),
             "PUT", payload, "application/x-tar")
    for name in stale:
        print("[sync] removing stale {}".format(name))
        docker_exec(robot, container, "rm -f {}/{}".format(dest, name))
    docker_exec(robot, container, "chown -R {} {} 2>/dev/null || true".format(DATA_UID_GID, dest))


# --------------------------------------------------------------------------- shared

def parse_fingerprints(text):
    """Lines of "<size> <sha256>  <name>" -> {name: (size, sha256)}."""
    remote = {}
    for line in text.replace("\r", "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1].isalnum() and len(parts[1]) == 64:
            remote[parts[-1]] = (int(parts[0]), parts[1])
    return remote


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default=DEFAULT_ROBOT, help="hostname or IP (default: %(default)s)")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="local .onnx directory")
    parser.add_argument("--dest", default=DEFAULT_DEST, help="directory on the robot")
    parser.add_argument("--transport", choices=("auto", "ssh", "docker"), default="auto")
    parser.add_argument("--no-delete", action="store_true",
                        help="leave weights on the robot that are gone locally")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, change nothing")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    args = parser.parse_args()

    local = local_models(args.source)
    print("[sync] {} local model(s) in {}".format(len(local), args.source))

    use_ssh = args.transport == "ssh" or (args.transport == "auto" and ssh_available(args.robot))
    container = None
    if use_ssh:
        print("[sync] transport: rsync over ssh to {}@{}".format(SSH_USER, args.robot))
        remote = ssh_remote_fingerprints(args.robot, args.dest)
    else:
        container = docker_data_container(args.robot)
        print("[sync] transport: Docker API on {}:{} via container '{}'".format(
            args.robot, DOCKER_PORT, container))
        remote = docker_remote_fingerprints(args.robot, container, args.dest)

    upload, stale, unchanged = plan(local, remote, not args.no_delete)
    moved_mb = sum(local[n][0] for n in upload) / 1e6
    print("[sync] {}:{} has {} model(s); {} unchanged, {} to upload ({:.1f} MB), {} to remove".format(
        args.robot, args.dest, len(remote), len(unchanged), len(upload), moved_mb, len(stale)))
    for name in unchanged:
        print("[sync]   = {}".format(name))
    for name in upload:
        print("[sync]   > {} ({:.1f} MB)".format(name, local[name][0] / 1e6))
    for name in stale:
        print("[sync]   x {}".format(name))

    if args.dry_run:
        print("[sync] --dry-run: stopping here")
        return 0
    if not upload and not stale:
        print("[sync] already in sync, nothing uploaded")
        return 0

    started = time.time()
    if use_ssh:
        ssh_upload(args.robot, args.source, args.dest, upload, stale, args.retries)
    else:
        docker_upload(args.robot, container, args.source, args.dest, upload, stale)
    elapsed = time.time() - started
    rate = " ({:.0f} KB/s)".format(moved_mb * 1000 / elapsed) if moved_mb and elapsed else ""
    print("[sync] done in {:.0f}s{}".format(elapsed, rate))

    # Verify rather than trust: this link corrupts nothing, but it does truncate.
    if use_ssh:
        after = ssh_remote_fingerprints(args.robot, args.dest)
    else:
        after = docker_remote_fingerprints(args.robot, container, args.dest)
    bad = [n for n, fingerprint in sorted(local.items()) if after.get(n) != fingerprint]
    if bad:
        print("[sync] FAILED verification: {}".format(", ".join(bad)))
        return 1
    print("[sync] verified {} model(s) on {}:{}".format(len(local), args.robot, args.dest))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (RuntimeError, urllib.error.URLError, subprocess.TimeoutExpired) as exc:
        sys.exit("[sync] {}".format(exc))
