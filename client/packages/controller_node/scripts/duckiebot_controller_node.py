#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

import rospy
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, Int32MultiArray

VEHICLE_NAME = os.environ.get("VEHICLE_NAME", "bot")


# Configuration

@dataclass
class PIDGains:
    k_p: float
    k_i: float
    k_d: float


LATERAL_GAINS = PIDGains(k_p=0.55, k_i=0.0, k_d=0.1)

SKIP_WAYPOINTS = 1
MAX_LOOKAT = 6
GOAL_ANGLE_BETA = 0.95
STRAIGHT_EPS = 0.005

CONTROL_HZ = 20.0

DEFAULT_TARGET_SPEED_KMH = 0.22 * 3.6  # 0.22 m/s
MAX_SPEED_KMH = 15.0
DEFAULT_SPEED_BETA = 0.6
DEFAULT_TEMPORAL_DT = 0.25
V_DEADBAND_MPS = 0.03
DEFAULT_STEER_MIN_SPEED_KMH = 0.0

MAX_STEER = 0.8
MAX_STEER_DELTA = 0.1

CARLA_SCALE = 20.0
MAX_OMEGA = 4.0

OMEGA_SIGN = 1.0
V_MAX_MPS = MAX_SPEED_KMH / 3.6


# Controller math

def select_goal_point(path: np.ndarray, skip: int, max_lookat: int,
                      eps: float = STRAIGHT_EPS) -> tuple[int, int, int]:
    n = len(path)
    start = int(np.clip(skip, 0, n - 1))
    end = int(np.clip(max_lookat, start, n - 1))
    if end - start < 2:
        return start, end, end

    p0 = path[start]
    chord = path[end] - p0
    length = float(np.hypot(chord[0], chord[1]))
    if length < 1e-6:
        return start, end, end

    rel = path[start + 1:end] - p0
    dist = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / length
    i = int(np.argmax(dist))
    apex = end if dist[i] < eps else start + 1 + i
    return start, apex, end


class LateralPID:
    def __init__(self, gains: PIDGains, dt: float, buffer_len: int = 10):
        self._gains = gains
        self._dt = dt
        self._errors: deque[float] = deque(maxlen=buffer_len)

    def set_gains(self, gains: PIDGains):
        self._gains = gains

    def step(self, error: float) -> float:
        self._errors.append(error)
        derivative = (self._errors[-1] - self._errors[-2]) / self._dt if len(self._errors) >= 2 else 0.0
        integral = sum(self._errors) * self._dt
        steer = self._gains.k_p * error + self._gains.k_d * derivative + self._gains.k_i * integral
        return float(np.clip(steer, -1.0, 1.0))


# ROS node

class DuckiebotTrajectoryController:
    def __init__(self):
        veh = rospy.get_param("~veh", VEHICLE_NAME)

        self._lateral = LateralPID(LATERAL_GAINS, 1.0 / CONTROL_HZ)
        self._past_steer = 0.0

        self._skip_waypoints = SKIP_WAYPOINTS
        self._max_lookat = MAX_LOOKAT
        self._goal_beta = GOAL_ANGLE_BETA
        self._goal_angle: float | None = None
        self._goal_indices = (0, 0, 0)

        self._path: np.ndarray | None = None
        self._temporal: np.ndarray | None = None
        self._pending_path: np.ndarray | None = None
        self._pending_temporal: np.ndarray | None = None
        self._pending_action: int | None = None
        self._active_action = 0

        self._target_speed_kmh = DEFAULT_TARGET_SPEED_KMH
        self._model_busy = False

        self._speed_beta = DEFAULT_SPEED_BETA
        self._temporal_dt = DEFAULT_TEMPORAL_DT
        self._speed_bins: np.ndarray | None = None
        self._bins_time = time.monotonic()
        self._new_speed_profile: np.ndarray | None = None
        self._use_speed_controller = True
        self._steer_min_speed_kmh = DEFAULT_STEER_MIN_SPEED_KMH

        self._enabled = False

        self._cmd_pub = rospy.Publisher(
            "/{}/car_cmd_switch_node/cmd".format(veh), Twist2DStamped, queue_size=1)

        self._model_cmd_pub = rospy.Publisher("/model_node/command", Int32, queue_size=1, latch=True)
        rospy.Subscriber("~command", Int32, self._on_command_input, queue_size=1)
        rospy.Subscriber("~enabled", Bool, self._on_enabled, queue_size=1)
        rospy.Subscriber("~lateral_pid", Float32MultiArray, self._on_lateral_pid, queue_size=1)
        rospy.Subscriber("~target_speed_kmh", Float32, self._on_target_speed, queue_size=1)
        rospy.Subscriber("~skip_waypoints", Int32, self._on_skip_waypoints, queue_size=1)
        rospy.Subscriber("~max_lookat", Int32, self._on_max_lookat, queue_size=1)
        rospy.Subscriber("~goal_beta", Float32, self._on_goal_beta, queue_size=1)
        rospy.Subscriber("~speed_beta", Float32, self._on_speed_beta, queue_size=1)
        rospy.Subscriber("~use_speed_controller", Bool, self._on_use_speed_controller, queue_size=1)
        rospy.Subscriber("~steer_min_speed_kmh", Float32, self._on_steer_min_speed, queue_size=1)
        rospy.Subscriber("/model_node/temporal_dt", Float32, self._on_temporal_dt, queue_size=1)
        rospy.Subscriber("/model_node/busy", Bool, self._on_model_busy, queue_size=1)

        rospy.Subscriber("/model_node/spatial", Float32MultiArray, self._on_spatial, queue_size=1)
        rospy.Subscriber("/model_node/temporal", Float32MultiArray, self._on_temporal, queue_size=1)
        rospy.Subscriber("/model_node/action", Int32, self._on_model_action, queue_size=1)

        self._pub_active_spatial = rospy.Publisher("~active_spatial", Float32MultiArray, queue_size=1)
        self._pub_active_temporal = rospy.Publisher("~active_temporal", Float32MultiArray, queue_size=1)
        self._pub_active_action = rospy.Publisher("~active_action", Int32, queue_size=1, latch=True)
        self._pub_goal_indices = rospy.Publisher("~goal_indices", Int32MultiArray, queue_size=1)
        self._pub_cmd_speed = rospy.Publisher("~commanded_speed_kmh", Float32, queue_size=1)
        self._pub_speed_bins = rospy.Publisher("~speed_bins_kmh", Float32MultiArray, queue_size=1)

        self._timer = rospy.Timer(rospy.Duration(1.0 / CONTROL_HZ), self._control_tick)
        rospy.loginfo("DuckiebotTrajectoryController up @ %.0f Hz (veh=%s)", CONTROL_HZ, veh)

    # Message parsing
    @staticmethod
    def _parse_trajectory(msg) -> np.ndarray:
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1, 2)
        arr[:, 1] *= -1.0
        arr /= CARLA_SCALE
        return arr

    def _on_command_input(self, msg):
        self._model_cmd_pub.publish(Int32(data=msg.data))

    def _on_enabled(self, msg):
        self._enabled = bool(msg.data)
        if not self._enabled:
            self._cmd_pub.publish(Twist2DStamped())

    def _on_lateral_pid(self, msg):
        if len(msg.data) < 3:
            return
        kp, ki, kd = float(msg.data[0]), float(msg.data[1]), float(msg.data[2])
        self._lateral.set_gains(PIDGains(k_p=kp, k_i=ki, k_d=kd))
        rospy.loginfo("lateral PID updated: kp=%.3f ki=%.3f kd=%.3f", kp, ki, kd)

    def _on_target_speed(self, msg):
        self._target_speed_kmh = float(np.clip(float(msg.data), 0.0, MAX_SPEED_KMH))
        rospy.loginfo("target speed updated: %.2f km/h", self._target_speed_kmh)

    def _on_skip_waypoints(self, msg):
        self._skip_waypoints = max(0, int(msg.data))
        rospy.loginfo("skip waypoints updated: %d", self._skip_waypoints)

    def _on_max_lookat(self, msg):
        self._max_lookat = max(0, int(msg.data))
        rospy.loginfo("max look-at updated: %d", self._max_lookat)

    def _on_goal_beta(self, msg):
        self._goal_beta = float(np.clip(float(msg.data), 0.0, 1.0))
        rospy.loginfo("goal angle EMA beta updated: %.2f", self._goal_beta)

    def _on_speed_beta(self, msg):
        self._speed_beta = float(np.clip(float(msg.data), 0.05, 1.0))
        rospy.loginfo("speed bins EMA beta updated: %.2f", self._speed_beta)

    def _on_model_busy(self, msg):
        # While weights reload there are no new predictions, and _tick would
        # happily keep driving the last accepted path. Stop instead.
        self._model_busy = bool(msg.data)
        if self._model_busy:
            # Drop the stale path too: the new weights may disagree with it, so
            # wait for a fresh prediction rather than resuming on the old one.
            self._path = None
            self._cmd_pub.publish(Twist2DStamped())

    def _on_temporal_dt(self, msg):
        self._temporal_dt = max(float(msg.data), 1e-3)
        rospy.loginfo("temporal dt from model metadata: %.3fs", self._temporal_dt)

    def _on_use_speed_controller(self, msg):
        self._use_speed_controller = bool(msg.data)
        rospy.loginfo("speed controller %s", "ON (bins)" if self._use_speed_controller else "OFF (constant)")

    def _on_steer_min_speed(self, msg):
        self._steer_min_speed_kmh = max(0.0, float(msg.data))
        rospy.loginfo("steer min speed updated: %.2f km/h", self._steer_min_speed_kmh)

    def _on_spatial(self, msg):
        self._pending_path = self._parse_trajectory(msg)
        self._maybe_accept_pending()

    def _on_temporal(self, msg):
        self._pending_temporal = self._parse_trajectory(msg)
        self._maybe_accept_pending()

    def _on_model_action(self, msg):
        self._pending_action = int(msg.data)
        self._maybe_accept_pending()

    # Prediction acceptance
    def _maybe_accept_pending(self):
        if (self._pending_path is None or self._pending_temporal is None
                or self._pending_action is None):
            return

        path = self._pending_path
        temporal = self._pending_temporal
        action = self._pending_action
        self._pending_path = None
        self._pending_temporal = None
        self._pending_action = None

        self._temporal = temporal
        self._active_action = action
        # Goal angle must be up to date before the path goes live: _control_tick runs on the
        # rospy.Timer thread and would otherwise see a fresh path with a stale/None goal angle.
        self._update_goal_angle(path)
        self._path = path
        self._pub_active_action.publish(Int32(data=action))

        pts = np.vstack([[0.0, 0.0], temporal])
        self._new_speed_profile = np.linalg.norm(np.diff(pts, axis=0), axis=1) / self._temporal_dt

    # Goal angle
    def _update_goal_angle(self, path):
        self._goal_indices = select_goal_point(
            path, self._skip_waypoints, self._max_lookat)
        _, apex, _ = self._goal_indices

        gx, gy = path[apex]
        angle = math.atan2(gy, gx)
        self._goal_angle = angle if self._goal_angle is None else (
            self._goal_beta * angle + (1.0 - self._goal_beta) * self._goal_angle)

    # Speed bins
    def _update_speed_bins(self) -> float:
        now = time.monotonic()
        dt = self._temporal_dt

        if self._speed_bins is not None:
            steps = int((now - self._bins_time) // dt)
            if steps > 0:
                shifted = np.zeros_like(self._speed_bins)
                if steps < len(self._speed_bins):
                    shifted[:len(self._speed_bins) - steps] = self._speed_bins[steps:]
                self._speed_bins = shifted
                self._bins_time += steps * dt

        profile = self._new_speed_profile
        if profile is not None:
            self._new_speed_profile = None
            if self._speed_bins is None or len(self._speed_bins) != len(profile):
                self._speed_bins = profile.astype(float)
                self._bins_time = now
            else:
                self._speed_bins = (self._speed_beta * profile
                                    + (1.0 - self._speed_beta) * self._speed_bins)

        return 0.0 if self._speed_bins is None else float(self._speed_bins[0])

    # Control loop
    def _control_tick(self, _evt):
        # rospy.Timer does not catch exceptions: one throwing tick kills the timer thread for
        # good and the robot silently stops steering. A bad tick must cost one tick, not the loop.
        try:
            self._tick()
        except Exception:
            rospy.logerr("control tick failed", exc_info=True)

    def _tick(self):
        path = self._path
        if path is None or self._goal_angle is None:
            return

        self._pub_active_spatial.publish(
            Float32MultiArray(data=path.astype(np.float32).flatten().tolist()))
        if self._temporal is not None:
            self._pub_active_temporal.publish(
                Float32MultiArray(data=self._temporal.astype(np.float32).flatten().tolist()))

        self._pub_goal_indices.publish(Int32MultiArray(data=list(self._goal_indices)))

        v_now = self._update_speed_bins()
        if self._use_speed_controller:
            v = min(v_now, self._target_speed_kmh / 3.6)
        else:
            v = self._target_speed_kmh / 3.6
        v = float(np.clip(v, 0.0, V_MAX_MPS))
        if v < V_DEADBAND_MPS:
            v = 0.0
        self._pub_cmd_speed.publish(Float32(data=v * 3.6))
        bins = [] if self._speed_bins is None else (self._speed_bins * 3.6).tolist()
        self._pub_speed_bins.publish(Float32MultiArray(data=bins))

        if v * 3.6 >= self._steer_min_speed_kmh:
            steer = self._lateral.step(self._goal_angle)
            steer = max(self._past_steer - MAX_STEER_DELTA,
                        min(self._past_steer + MAX_STEER_DELTA, steer))
            steer = float(np.clip(steer, -MAX_STEER, MAX_STEER))
            self._past_steer = steer
            omega = OMEGA_SIGN * MAX_OMEGA * steer
        else:
            self._past_steer = 0.0
            omega = 0.0

        if not self._enabled or self._model_busy:
            return

        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = omega
        self._cmd_pub.publish(cmd)

    def on_shutdown(self):
        stop = Twist2DStamped()
        stop.header.stamp = rospy.Time.now()
        stop.v = 0.0
        stop.omega = 0.0
        self._cmd_pub.publish(stop)


def main():
    rospy.init_node("controller_node", anonymous=False)
    node = DuckiebotTrajectoryController()
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()


if __name__ == "__main__":
    main()
