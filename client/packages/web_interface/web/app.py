import asyncio
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import projection
import ros_bridge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The weights are no longer in the image; they sit on the robot's persistent /data partition so
# they never travel through the build context. This mirrors model_node.resolve_models_dir() --
# the two are separate ROS packages, so the few lines are duplicated rather than imported across
# the package boundary; change them together.
MODELS_DIR_ON_ROBOT = "/data/duckad/models"
IN_REPO_ASSETS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "model_node", "assets"))


def _resolve_models_dir():
    override = os.environ.get("DUCKAD_MODELS_DIR")
    if override:
        return override
    for candidate in (MODELS_DIR_ON_ROBOT, IN_REPO_ASSETS_DIR):
        try:
            if any(f.endswith(".onnx") for f in os.listdir(candidate)):
                return candidate
        except OSError:
            continue
    return MODELS_DIR_ON_ROBOT


# Deliberately no module-level MODELS_DIR: callers resolve per request. The dashboard comes up
# before anything guarantees a sync has happened, and _list_models() returning [] forever
# because the directory was empty at start-up is a confusing way to find that out.

STATIC_VERSION = str(int(max(
    os.path.getmtime(os.path.join(root, f))
    for root, _, files in os.walk(os.path.join(BASE_DIR, "static"))
    for f in files
)))

COMMAND_NAMES = ["default", "left", "straight", "right"]
KEY_TO_COMMAND = {"a": 1, "d": 3, "w": 2, "s": 0}

# RAM moves slowly and reading it is a parse, so cache it instead of re-reading on every state push.
SYS_POLL_S = 2.0
_mem_cache = {"at": 0.0, "value": None}

# Rate we push camera frames to the browser. The camera runs at 30 Hz, but the dashboard is a
# monitoring view, not the control path - the model only looks at ~3 Hz. Streaming all 30 fps
# costs ~600 KB/s on a link shared with every control POST, and the robot's WiFi does not have
# it to spare. 10 still reads as live video while leaving headroom for the control POSTs.
CAMERA_STREAM_HZ = 10.0

# Resolution we transmit at. This changes bytes on the wire, NOT the size on screen: #camera is
# an <img> sized entirely by CSS (width/height 100%, object-fit: contain), so the browser just
# upscales a smaller JPEG into the same viewport. Measured at 640x480 the camera's own JPEGs run
# ~36 KB, which at 10 Hz is ~364 KB/s -- at or over what this 2.4 GHz link sustains, so every
# control POST and state request queued behind a backlog of camera frames. That queueing, not
# the raw rate, is what made p95 latency 7.8x p50.
#
# Keep the aspect ratio at 4:3. The overlay canvas fills the whole viewport while the image is
# letterboxed into it by object-fit, so a different ratio would slide the picture out from under
# the trajectory drawn on top of it. Width alone is set here and the height follows from the
# source, so a camera reconfigured to another 4:3 mode still lines up.
CAMERA_STREAM_WIDTH = 320
CAMERA_STREAM_JPEG_QUALITY = 60

# One downscale per camera frame, not per connected dashboard: two browsers watching should not
# cost two encodes. Not locked across the encode itself -- a race just means the same frame is
# encoded twice, which is harmless and cheaper than serialising every viewer behind one lock.
_frame_cache = {"seq": -1, "jpeg": None}
_frame_lock = threading.Lock()
_logged_reduction = False


def _downscale(jpeg_bytes):
    """Re-encode a camera JPEG smaller. Returns the original on any failure."""
    global _logged_reduction
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jpeg_bytes
    height, width = image.shape[:2]
    if width <= CAMERA_STREAM_WIDTH:
        return jpeg_bytes
    scale = CAMERA_STREAM_WIDTH / float(width)
    resized = cv2.resize(image, (CAMERA_STREAM_WIDTH, int(round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized,
                               [int(cv2.IMWRITE_JPEG_QUALITY), CAMERA_STREAM_JPEG_QUALITY])
    if not ok:
        return jpeg_bytes
    out = encoded.tobytes()
    if not _logged_reduction:
        _logged_reduction = True
        print("[camera] streaming {}x{} q{}: {:.1f} KB -> {:.1f} KB per frame "
              "({:.0f}% less, ~{:.0f} KB/s at {:.0f} Hz)".format(
                  CAMERA_STREAM_WIDTH, resized.shape[0], CAMERA_STREAM_JPEG_QUALITY,
                  len(jpeg_bytes) / 1024.0, len(out) / 1024.0,
                  100.0 * (1 - len(out) / float(len(jpeg_bytes))),
                  len(out) * CAMERA_STREAM_HZ / 1024.0, CAMERA_STREAM_HZ))
    return out


def _stream_frame(seq, jpeg_bytes):
    with _frame_lock:
        if _frame_cache["seq"] == seq:
            return _frame_cache["jpeg"]
    out = _downscale(jpeg_bytes)
    with _frame_lock:
        _frame_cache["seq"], _frame_cache["jpeg"] = seq, out
    return out

# Floor on how fast we answer a client's request for the next state. The client paces itself by
# asking only once it has rendered the previous payload (see ws_state); this is the cap on top.
STATE_MIN_PERIOD_S = 0.1


def _read_memory():
    # Host RAM in MB: the container is not memory-namespaced, so /proc/meminfo is the real machine.
    # Returns None if it can't be read - the dashboard just shows a dash.
    now = time.monotonic()
    if _mem_cache["value"] is not None and now - _mem_cache["at"] < SYS_POLL_S:
        return _mem_cache["value"]
    try:
        with open("/proc/meminfo") as f:
            info = {k.strip(): int(v.split()[0]) for k, v in
                    (line.split(":", 1) for line in f)}
        # Same arithmetic as procps' free(1), so this agrees with `free -m` over SSH:
        # used = total - free - (buffers + page cache + reclaimable slab). NOT total - available;
        # cache is reclaimable, so counting it as used overstates pressure.
        cache = info["Cached"] + info["Buffers"] + info["SReclaimable"]
        value = {
            "total_mb": info["MemTotal"] // 1024,
            "used_mb": (info["MemTotal"] - info["MemFree"] - cache) // 1024,
            "available_mb": info["MemAvailable"] // 1024,
            "swap_used_mb": (info["SwapTotal"] - info["SwapFree"]) // 1024,
        }
    except Exception:
        value = None
    _mem_cache.update(at=now, value=value)
    return value


app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.on_event("startup")
def _startup():
    ros_bridge.start()
    projection.load()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "v": STATIC_VERSION, "vehicle": ros_bridge.VEHICLE_NAME})


class KeyPress(BaseModel):
    key: str


@app.post("/command")
def command(payload: KeyPress):
    key = payload.key.lower()
    if key not in KEY_TO_COMMAND:
        return {"status": "ignored", "key": key}
    cmd = KEY_TO_COMMAND[key]
    ros_bridge.publish_command(cmd)
    return {"status": "ok", "command": cmd, "name": COMMAND_NAMES[cmd]}


class EnabledPayload(BaseModel):
    enabled: bool


@app.post("/enabled")
def enabled(payload: EnabledPayload):
    ros_bridge.set_enabled(payload.enabled)
    return {"status": "ok", "enabled": payload.enabled}


class PidPayload(BaseModel):
    kp: float
    ki: float
    kd: float


@app.post("/pid")
def pid(payload: PidPayload):
    ros_bridge.set_lateral_pid(payload.kp, payload.ki, payload.kd)
    return {"status": "ok", "kp": payload.kp, "ki": payload.ki, "kd": payload.kd}


class TargetSpeedPayload(BaseModel):
    speed_kmh: float


@app.post("/target_speed")
def target_speed(payload: TargetSpeedPayload):
    ros_bridge.set_target_speed(payload.speed_kmh)
    return {"status": "ok", "speed_kmh": payload.speed_kmh}


class WaypointIndexPayload(BaseModel):
    index: int


@app.post("/skip_waypoints")
def skip_waypoints(payload: WaypointIndexPayload):
    ros_bridge.set_skip_waypoints(payload.index)
    return {"status": "ok", "index": payload.index}


@app.post("/max_lookat")
def max_lookat(payload: WaypointIndexPayload):
    ros_bridge.set_max_lookat(payload.index)
    return {"status": "ok", "index": payload.index}


class GoalBetaPayload(BaseModel):
    beta: float


@app.post("/goal_beta")
def goal_beta(payload: GoalBetaPayload):
    ros_bridge.set_goal_beta(payload.beta)
    return {"status": "ok", "beta": payload.beta}


class CommandBetaPayload(BaseModel):
    beta: float


@app.post("/command_beta")
def command_beta(payload: CommandBetaPayload):
    ros_bridge.set_command_beta(payload.beta)
    return {"status": "ok", "beta": payload.beta}


class SpeedBetaPayload(BaseModel):
    beta: float


@app.post("/speed_beta")
def speed_beta(payload: SpeedBetaPayload):
    ros_bridge.set_speed_beta(payload.beta)
    return {"status": "ok", "beta": payload.beta}


class UseSpeedControllerPayload(BaseModel):
    enabled: bool


@app.post("/use_speed_controller")
def use_speed_controller(payload: UseSpeedControllerPayload):
    ros_bridge.set_use_speed_controller(payload.enabled)
    return {"status": "ok", "enabled": payload.enabled}


class SteerMinSpeedPayload(BaseModel):
    speed_kmh: float


@app.post("/steer_min_speed")
def steer_min_speed(payload: SteerMinSpeedPayload):
    ros_bridge.set_steer_min_speed(payload.speed_kmh)
    return {"status": "ok", "speed_kmh": payload.speed_kmh}


def _list_models():
    models_dir = _resolve_models_dir()
    try:
        names = sorted(f for f in os.listdir(models_dir) if f.endswith(".onnx"))
        return [{"file": f, "size_mb": round(os.path.getsize(os.path.join(models_dir, f)) / 1e6, 1)}
                for f in names]
    except OSError:
        return []


@app.get("/models")
def models():
    return {"models": _list_models()}


class ModelPayload(BaseModel):
    file: str


@app.post("/model")
def model(payload: ModelPayload):
    # basename() keeps a crafted "file" from escaping the assets directory.
    name = os.path.basename(payload.file)
    if not os.path.isfile(os.path.join(_resolve_models_dir(), name)) or not name.endswith(".onnx"):
        return {"status": "unknown", "file": name}
    ros_bridge.set_model(name)
    return {"status": "ok", "file": name}


class InferenceHzPayload(BaseModel):
    hz: float


@app.post("/inference_hz")
def inference_hz(payload: InferenceHzPayload):
    ros_bridge.set_inference_hz(payload.hz)
    return {"status": "ok", "hz": payload.hz}


@app.websocket("/ws/camera")
async def ws_camera(websocket: WebSocket):
    await websocket.accept()
    period = 1.0 / CAMERA_STREAM_HZ
    last_seq = -1
    try:
        while True:
            # Only send when the frame is actually new. Without the sequence check a stalled
            # camera would still cost full bandwidth re-sending the same JPEG over and over.
            if ros_bridge.latest_jpeg is not None and ros_bridge.jpeg_seq != last_seq:
                seq, raw = ros_bridge.jpeg_seq, ros_bridge.latest_jpeg
                last_seq = seq
                # decode+resize+encode is ~10-20 ms on the Nano's A57s. Run it in a worker so it
                # does not stall the event loop, which also serves /ws/state and every POST -
                # blocking here would just move the latency we are trying to remove.
                frame = await asyncio.get_running_loop().run_in_executor(
                    None, _stream_frame, seq, raw)
                await websocket.send_bytes(frame)
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/state")
async def ws_state(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "spatial": ros_bridge.latest_spatial,
                "temporal": ros_bridge.latest_temporal,
                "spatial_cam": projection.project(ros_bridge.latest_spatial),
                "temporal_cam": projection.project(ros_bridge.latest_temporal),
                "action": ros_bridge.latest_action,
                "action_name": COMMAND_NAMES[ros_bridge.latest_action],
                "active_action": ros_bridge.latest_active_action,
                "command_probs": ros_bridge.latest_command_probs,
                "speed_kmh": ros_bridge.latest_speed_kmh,
                "commanded_speed_kmh": ros_bridge.latest_commanded_speed_kmh,
                "speed_bins": ros_bridge.latest_speed_bins,
                "temporal_dt": ros_bridge.latest_temporal_dt,
                "inference_ms": ros_bridge.latest_inference_ms,
                "command": ros_bridge.latest_command,
                "command_name": COMMAND_NAMES[ros_bridge.latest_command],
                "enabled": ros_bridge.latest_enabled,
                "lateral_pid": ros_bridge.latest_lateral_pid,
                "target_speed_kmh": ros_bridge.latest_target_speed_kmh,
                "skip_waypoints": ros_bridge.latest_skip_waypoints,
                "max_lookat": ros_bridge.latest_max_lookat,
                "goal_beta": ros_bridge.latest_goal_beta,
                "command_beta": ros_bridge.latest_command_beta,
                "speed_beta": ros_bridge.latest_speed_beta,
                "use_speed_controller": ros_bridge.latest_use_speed_controller,
                "steer_min_speed_kmh": ros_bridge.latest_steer_min_speed_kmh,
                "goal_indices": ros_bridge.latest_goal_indices,
                "inference_hz": ros_bridge.latest_inference_hz,
                "model_file": ros_bridge.latest_model_file,
                "model_busy": ros_bridge.latest_model_busy,
                "memory": _read_memory(),   # cached, refreshed every SYS_POLL_S
            })
            # Client-pulled, strictly one message in flight: the browser asks for the next
            # payload only once it has rendered this one. Pushing blindly let a slow link build a
            # backlog in the TCP buffer, so the dashboard rendered history and looked frozen.
            await websocket.receive_text()
            await asyncio.sleep(STATE_MIN_PERIOD_S)
    except WebSocketDisconnect:
        pass
