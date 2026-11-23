import os
import threading

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, Int32MultiArray, String

VEHICLE_NAME = os.environ.get("VEHICLE_NAME", "bot")
CAM_TOPIC = "/{}/camera_node/image/compressed".format(VEHICLE_NAME)

latest_jpeg = None
jpeg_seq = 0
latest_spatial = []
latest_temporal = []
latest_active_action = 0
latest_action = 0
latest_command_probs = [0.25, 0.25, 0.25, 0.25]
latest_speed_kmh = 0.0
latest_inference_ms = 0.0
latest_commanded_speed_kmh = 0.0
latest_speed_bins = []
latest_temporal_dt = 0.25
latest_command = 2
latest_enabled = False
latest_lateral_pid = [0.55, 0.0, 0.1]
latest_target_speed_kmh = 0.22 * 3.6  # 0.22 m/s
latest_skip_waypoints = 1
latest_max_lookat = 6
latest_goal_beta = 0.95
latest_command_beta = 0.6
latest_speed_beta = 0.6
latest_use_speed_controller = True
latest_steer_min_speed_kmh = 0.0
latest_goal_indices = []
latest_inference_hz = 7.0
latest_model_file = ""
latest_model_busy = False

_command_pub = None
_enabled_pub = None
_lateral_pid_pub = None
_target_speed_pub = None
_skip_waypoints_pub = None
_max_lookat_pub = None
_goal_beta_pub = None
_command_beta_pub = None
_speed_beta_pub = None
_use_speed_controller_pub = None
_steer_min_speed_pub = None
_inference_hz_pub = None
_model_pub = None


def _on_image(msg):
    global latest_jpeg, jpeg_seq
    latest_jpeg = bytes(msg.data)
    jpeg_seq += 1


def _on_spatial(msg):
    global latest_spatial
    latest_spatial = np.asarray(msg.data, dtype=float).reshape(-1, 2).tolist()


def _on_temporal(msg):
    global latest_temporal
    latest_temporal = np.asarray(msg.data, dtype=float).reshape(-1, 2).tolist()


def _on_action(msg):
    global latest_action
    latest_action = int(msg.data)


def _on_active_action(msg):
    global latest_active_action
    latest_active_action = int(msg.data)


def _on_command_probs(msg):
    global latest_command_probs
    latest_command_probs = list(msg.data)


def _on_speed(msg):
    global latest_speed_kmh
    latest_speed_kmh = float(msg.data)


def _on_inference_ms(msg):
    global latest_inference_ms
    latest_inference_ms = float(msg.data)


def _on_goal_indices(msg):
    global latest_goal_indices
    latest_goal_indices = [int(v) for v in msg.data]


def _on_commanded_speed(msg):
    global latest_commanded_speed_kmh
    latest_commanded_speed_kmh = float(msg.data)


def _on_speed_bins(msg):
    global latest_speed_bins
    latest_speed_bins = list(msg.data)


def _on_model_file(msg):
    global latest_model_file
    latest_model_file = str(msg.data)


def _on_model_busy(msg):
    global latest_model_busy
    latest_model_busy = bool(msg.data)


def _on_temporal_dt(msg):
    global latest_temporal_dt
    latest_temporal_dt = float(msg.data)


def _spin():
    rospy.Subscriber(CAM_TOPIC, CompressedImage, _on_image,
                     queue_size=1, buff_size=2 ** 24, tcp_nodelay=True)
    rospy.Subscriber("/controller_node/active_spatial", Float32MultiArray, _on_spatial, queue_size=1)
    rospy.Subscriber("/controller_node/active_temporal", Float32MultiArray, _on_temporal, queue_size=1)
    rospy.Subscriber("/controller_node/active_action", Int32, _on_active_action, queue_size=1)
    rospy.Subscriber("/model_node/action", Int32, _on_action, queue_size=1)
    rospy.Subscriber("/model_node/command_probs", Float32MultiArray, _on_command_probs, queue_size=1)
    rospy.Subscriber("/model_node/speed_kmh", Float32, _on_speed, queue_size=1)
    rospy.Subscriber("/model_node/inference_ms", Float32, _on_inference_ms, queue_size=1)
    rospy.Subscriber("/controller_node/goal_indices", Int32MultiArray, _on_goal_indices, queue_size=1)
    rospy.Subscriber("/controller_node/commanded_speed_kmh", Float32, _on_commanded_speed, queue_size=1)
    rospy.Subscriber("/controller_node/speed_bins_kmh", Float32MultiArray, _on_speed_bins, queue_size=1)
    rospy.Subscriber("/model_node/temporal_dt", Float32, _on_temporal_dt, queue_size=1)
    rospy.Subscriber("/model_node/model_file", String, _on_model_file, queue_size=1)
    rospy.Subscriber("/model_node/busy", Bool, _on_model_busy, queue_size=1)
    rospy.spin()


def start():
    global _command_pub, _enabled_pub, _lateral_pid_pub, _target_speed_pub
    global _skip_waypoints_pub, _max_lookat_pub, _goal_beta_pub, _command_beta_pub, _speed_beta_pub, _inference_hz_pub
    global _use_speed_controller_pub, _steer_min_speed_pub, _model_pub
    if not rospy.core.is_initialized():
        rospy.init_node("web_interface_node", anonymous=False, disable_signals=True)
    _command_pub = rospy.Publisher("/controller_node/command", Int32, queue_size=1, latch=True)
    _enabled_pub = rospy.Publisher("/controller_node/enabled", Bool, queue_size=1, latch=True)
    _lateral_pid_pub = rospy.Publisher("/controller_node/lateral_pid", Float32MultiArray, queue_size=1, latch=True)
    _target_speed_pub = rospy.Publisher("/controller_node/target_speed_kmh", Float32, queue_size=1, latch=True)
    _skip_waypoints_pub = rospy.Publisher("/controller_node/skip_waypoints", Int32, queue_size=1, latch=True)
    _max_lookat_pub = rospy.Publisher("/controller_node/max_lookat", Int32, queue_size=1, latch=True)
    _goal_beta_pub = rospy.Publisher("/controller_node/goal_beta", Float32, queue_size=1, latch=True)
    _command_beta_pub = rospy.Publisher("/model_node/command_beta", Float32, queue_size=1, latch=True)
    _speed_beta_pub = rospy.Publisher("/controller_node/speed_beta", Float32, queue_size=1, latch=True)
    _use_speed_controller_pub = rospy.Publisher("/controller_node/use_speed_controller", Bool, queue_size=1, latch=True)
    _steer_min_speed_pub = rospy.Publisher("/controller_node/steer_min_speed_kmh", Float32, queue_size=1, latch=True)
    _inference_hz_pub = rospy.Publisher("/model_node/inference_hz", Float32, queue_size=1, latch=True)
    _model_pub = rospy.Publisher("/model_node/model", String, queue_size=1, latch=True)
    _enabled_pub.publish(Bool(data=False))
    _target_speed_pub.publish(Float32(data=latest_target_speed_kmh))
    _skip_waypoints_pub.publish(Int32(data=latest_skip_waypoints))
    _max_lookat_pub.publish(Int32(data=latest_max_lookat))
    _goal_beta_pub.publish(Float32(data=latest_goal_beta))
    _command_beta_pub.publish(Float32(data=latest_command_beta))
    _speed_beta_pub.publish(Float32(data=latest_speed_beta))
    _use_speed_controller_pub.publish(Bool(data=latest_use_speed_controller))
    _steer_min_speed_pub.publish(Float32(data=latest_steer_min_speed_kmh))
    _inference_hz_pub.publish(Float32(data=latest_inference_hz))
    threading.Thread(target=_spin, daemon=True).start()


def publish_command(command):
    global latest_command
    latest_command = command
    _command_pub.publish(Int32(data=command))


def set_enabled(flag):
    global latest_enabled
    latest_enabled = bool(flag)
    _enabled_pub.publish(Bool(data=latest_enabled))


def set_lateral_pid(kp, ki, kd):
    global latest_lateral_pid
    latest_lateral_pid = [float(kp), float(ki), float(kd)]
    _lateral_pid_pub.publish(Float32MultiArray(data=latest_lateral_pid))


def set_target_speed(speed_kmh):
    global latest_target_speed_kmh
    latest_target_speed_kmh = float(speed_kmh)
    _target_speed_pub.publish(Float32(data=latest_target_speed_kmh))


def set_skip_waypoints(index):
    global latest_skip_waypoints
    latest_skip_waypoints = int(index)
    _skip_waypoints_pub.publish(Int32(data=latest_skip_waypoints))


def set_max_lookat(index):
    global latest_max_lookat
    latest_max_lookat = int(index)
    _max_lookat_pub.publish(Int32(data=latest_max_lookat))


def set_goal_beta(beta):
    global latest_goal_beta
    latest_goal_beta = float(beta)
    _goal_beta_pub.publish(Float32(data=latest_goal_beta))


def set_command_beta(beta):
    global latest_command_beta
    latest_command_beta = float(beta)
    _command_beta_pub.publish(Float32(data=latest_command_beta))


def set_speed_beta(beta):
    global latest_speed_beta
    latest_speed_beta = float(beta)
    _speed_beta_pub.publish(Float32(data=latest_speed_beta))


def set_use_speed_controller(flag):
    global latest_use_speed_controller
    latest_use_speed_controller = bool(flag)
    _use_speed_controller_pub.publish(Bool(data=latest_use_speed_controller))


def set_steer_min_speed(speed_kmh):
    global latest_steer_min_speed_kmh
    latest_steer_min_speed_kmh = float(speed_kmh)
    _steer_min_speed_pub.publish(Float32(data=latest_steer_min_speed_kmh))


def set_model(name):
    _model_pub.publish(String(data=str(name)))


def set_inference_hz(hz):
    global latest_inference_hz
    latest_inference_hz = float(hz)
    _inference_hz_pub.publish(Float32(data=latest_inference_hz))
