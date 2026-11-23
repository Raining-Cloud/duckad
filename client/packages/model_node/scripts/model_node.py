#!/usr/bin/env python3
import os
import sys
import threading
import time

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32, Float32, Float32MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onnx_model import OnnxModel, softmax

VEHICLE_NAME = os.environ.get("VEHICLE_NAME", "bot")
CAM_TOPIC = "/{}/camera_node/image/compressed".format(VEHICLE_NAME)
DEFAULT_MODEL = "model.onnx"

# Weights are NOT baked into the image. They live on the robot's persistent partition, which the
# Duckietown container runtime bind-mounts as /data, so they outlive both container restarts and
# image rebuilds and never travel through the build context (77 MB of ONNX over the robot's WiFi
# is what a `dts devel build` used to spend most of its time on). scripts/sync_models.sh puts
# them there and re-uploads only what changed.
# NOTE: web_interface/web/app.py resolves the same directory the same way -- keep them in step.
MODELS_DIR_ON_ROBOT = "/data/duckad/models"
IN_REPO_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "assets")


def resolve_models_dir():
    """Where to load .onnx from: /data on the robot, the checkout when running off it.

    Tested for actual weights rather than for the directory: an empty /data/duckad/models is
    what a robot that has never been synced looks like, and falling back to the checkout gives
    a developer running this outside the container something to load.
    """
    override = os.environ.get("DUCKAD_MODELS_DIR")
    if override:
        return override
    for candidate in (MODELS_DIR_ON_ROBOT, IN_REPO_ASSETS_DIR):
        try:
            if any(f.endswith(".onnx") for f in os.listdir(candidate)):
                return candidate
        except OSError:
            continue
    return MODELS_DIR_ON_ROBOT   # nothing anywhere: fail against the path we want in the error


ASSETS_DIR = resolve_models_dir()

DEFAULT_PUBLISH_RATE_HZ = 7.0
DEFAULT_COMMAND_BETA = 0.6

WORKER_POLL_S = 0.005
CAMERA_TIMEOUT_S = 3.0
WATCHDOG_PERIOD_S = 5.0


class ModelNode(object):
    # Inference runs on a dedicated thread, NOT in the camera callback. rospy services a
    # subscriber on the TCPROS receive thread, so a slow callback stops the socket from being
    # drained; at 30 Hz of JPEGs that backs up until TCP pushes back on camera_node and freezes
    # the camera for every subscriber on the robot.
    def __init__(self):
        self._command = rospy.get_param("~command", 2)
        self._model_file = rospy.get_param("~model", DEFAULT_MODEL)
        self._pending_model = None
        self._model = OnnxModel(os.path.join(ASSETS_DIR, self._model_file))
        self._model.warmup()
        self._last_predict = 0.0
        self._publish_period = 1.0 / DEFAULT_PUBLISH_RATE_HZ
        self._command_beta = DEFAULT_COMMAND_BETA
        self._smoothed_logits = None
        self._latest_msg = None
        self._lock = threading.Lock()
        self._last_frame_time = None
        self._last_warn = time.monotonic()
        self._frames_seen = 0
        self._predictions = 0

        self._pub_action = rospy.Publisher("~action", Int32, queue_size=1)
        self._pub_speed = rospy.Publisher("~speed_kmh", Float32, queue_size=1)
        self._pub_spatial = rospy.Publisher("~spatial", Float32MultiArray, queue_size=1)
        self._pub_temporal = rospy.Publisher("~temporal", Float32MultiArray, queue_size=1)
        self._pub_command_probs = rospy.Publisher("~command_probs", Float32MultiArray, queue_size=1)
        self._pub_inference_ms = rospy.Publisher("~inference_ms", Float32, queue_size=1)
        self._pub_temporal_dt = rospy.Publisher("~temporal_dt", Float32, queue_size=1, latch=True)
        self._pub_temporal_dt.publish(Float32(data=self._model.dt))
        self._pub_model_file = rospy.Publisher("~model_file", String, queue_size=1, latch=True)
        self._pub_model_file.publish(String(data=self._model_file))
        # Latched so anything that starts mid-swap still learns inference is down.
        self._pub_busy = rospy.Publisher("~busy", Bool, queue_size=1, latch=True)
        self._pub_busy.publish(Bool(data=False))

        rospy.Subscriber("~model", String, self._on_model, queue_size=1)
        rospy.Subscriber("~command", Int32, self._on_command, queue_size=1)
        rospy.Subscriber("~inference_hz", Float32, self._on_inference_hz, queue_size=1)
        rospy.Subscriber("~command_beta", Float32, self._on_command_beta, queue_size=1)
        rospy.Subscriber(CAM_TOPIC, CompressedImage, self._on_image,
                         queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)

        self._worker = threading.Thread(target=self._run, name="inference", daemon=True)
        self._worker.start()
        rospy.loginfo("model_node ready: cam_topic=%s period=%.3fs cmd=%d",
                      CAM_TOPIC, self._publish_period, self._command)

    def _on_model(self, msg):
        # Flag only. Building a session takes seconds and this is the TCPROS
        # thread; the swap itself happens in _run.
        name = os.path.basename(str(msg.data))
        if name and name != self._model_file:
            self._pending_model = name

    def _on_command(self, msg):
        command = int(msg.data)
        if command != self._command:
            self._smoothed_logits = None
        self._command = command

    def _on_command_beta(self, msg):
        self._command_beta = float(np.clip(float(msg.data), 0.05, 1.0))
        rospy.loginfo("command logits EMA beta updated: %.2f", self._command_beta)

    def _on_inference_hz(self, msg):
        hz = max(0.1, float(msg.data))
        self._publish_period = 1.0 / hz
        rospy.loginfo("inference_hz set: hz=%.3f -> period=%.3fs", hz, self._publish_period)

    def _on_image(self, msg):
        # Keep this O(1): no decode, no inference, no logging.
        with self._lock:
            self._latest_msg = msg
        self._frames_seen += 1
        self._last_frame_time = time.monotonic()

    def _run(self):
        while not rospy.is_shutdown():
            # Swapping here IS the pause: this loop is the only caller of predict().
            if self._pending_model is not None:
                self._swap_model()
                continue
            if self._model is None:
                time.sleep(WORKER_POLL_S)
                continue

            now = time.monotonic()
            if now - self._last_predict < self._publish_period:
                time.sleep(WORKER_POLL_S)
                continue

            with self._lock:
                msg, self._latest_msg = self._latest_msg, None
            if msg is None:
                self._check_camera()
                time.sleep(WORKER_POLL_S)
                continue

            self._last_predict = now
            try:
                self._predict_and_publish(msg)
            except Exception:
                rospy.logerr("inference failed", exc_info=True)
                time.sleep(self._publish_period)

    def _swap_model(self):
        name, self._pending_model = self._pending_model, None
        self._pub_busy.publish(Bool(data=True))
        # Fall back to the running weights if the new ones will not load.
        for candidate in (name, self._model_file):
            rospy.loginfo("loading weights: %s", candidate)
            try:
                # Drop the old session first: two live ORT sessions do not fit
                # in the Nano's RAM.
                self._model = None
                model = OnnxModel(os.path.join(ASSETS_DIR, candidate))
                model.warmup()
            except Exception:
                rospy.logerr("could not load %s", candidate, exc_info=True)
                continue
            self._model = model
            self._model_file = candidate
            self._smoothed_logits = None   # new norm stats, old EMA is meaningless
            self._pub_temporal_dt.publish(Float32(data=model.dt))
            self._pub_model_file.publish(String(data=candidate))
            break
        # Still "busy" if nothing loaded: inference really is down, and both the
        # dashboard and controller_node key off this flag.
        self._pub_busy.publish(Bool(data=self._model is None))

    def _check_camera(self):
        # Throttled by hand, not with logwarn_throttle: its args are evaluated on every call and
        # one of them is an XMLRPC round-trip to the master, which must not happen at poll rate.
        now = time.monotonic()
        if now - self._last_warn < WATCHDOG_PERIOD_S:
            return
        if self._last_frame_time is None:
            self._last_warn = now
            rospy.logwarn("no camera frame yet on %s (%d publisher(s) on the topic) - "
                          "camera_node down, or wedged before we subscribed?",
                          CAM_TOPIC, self._camera_publisher_count())
        elif now - self._last_frame_time > CAMERA_TIMEOUT_S:
            self._last_warn = now
            rospy.logwarn("camera stream stalled: no frame for %.1fs (%d frames seen so far)",
                          now - self._last_frame_time, self._frames_seen)

    @staticmethod
    def _camera_publisher_count():
        try:
            publishers = rospy.get_master().getSystemState()[2][0]
            return sum(len(nodes) for topic, nodes in publishers if topic == CAM_TOPIC)
        except Exception:
            return -1

    def _predict_and_publish(self, msg):
        image = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn_throttle(5.0, "dropped an undecodable camera frame")
            return

        infer_start = time.perf_counter()
        _, spatial, temporal, _bev_idx, command_logits, speed_kmh = self._model.predict(image, self._command)
        inference_ms = (time.perf_counter() - infer_start) * 1000.0

        self._smoothed_logits = command_logits if self._smoothed_logits is None else (
            self._command_beta * command_logits + (1.0 - self._command_beta) * self._smoothed_logits)
        command_probs = softmax(self._smoothed_logits)
        action = int(command_probs.argmax())

        self._pub_action.publish(Int32(data=action))
        self._pub_speed.publish(Float32(data=speed_kmh))
        self._pub_spatial.publish(Float32MultiArray(data=np.asarray(spatial, dtype=np.float32).flatten().tolist()))
        self._pub_temporal.publish(Float32MultiArray(data=np.asarray(temporal, dtype=np.float32).flatten().tolist()))
        self._pub_command_probs.publish(Float32MultiArray(data=np.asarray(command_probs, dtype=np.float32).tolist()))
        self._pub_inference_ms.publish(Float32(data=inference_ms))

        self._predictions += 1
        if self._predictions <= 3 or self._predictions % 30 == 0:
            rospy.loginfo("prediction #%d: action=%d speed=%.2fkm/h inference=%.0fms (%d frames seen)",
                          self._predictions, action, speed_kmh, inference_ms, self._frames_seen)


if __name__ == "__main__":
    rospy.init_node("model_node", anonymous=False)
    ModelNode()
    rospy.spin()
