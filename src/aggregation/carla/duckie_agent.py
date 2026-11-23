import math
import random

from shapely.errors import GEOSException

from ...carla_agents.navigation.behavior_agent import BehaviorAgent
from ...carla_agents.navigation.local_planner import RoadOption
from ...carla_agents.tools.misc import (
    get_speed,
    get_trafficlight_trigger_location,
    is_within_distance,
)
from ..command import Command
from .junction import available_junction_actions

_COMMAND_TO_ROADOPTION = {
    Command.LEFT: RoadOption.LEFT,
    Command.STRAIGHT: RoadOption.STRAIGHT,
    Command.RIGHT: RoadOption.RIGHT,
}


class DuckieBehaviorAgent(BehaviorAgent):
    DUCKIE_DETECT_DIST = 15.0
    DUCKIE_STOP_GAP = 3.0

    DUCKIE_WAIT_TIMEOUT = 4.0

    STOP_SIGN_DETECT_DIST = 12.0
    STOP_SIGN_BRAKE_GAP = 2.5

    YIELD_RIGHT_BEFORE_LEFT = True
    JUNCTION_SCAN_DIST = 18.0

    NORMAL_SPEED = 20.0
    CREEP_SPEED = 7.0

    JUNCTION_CHOICE_LOOKAHEAD = 15.0

    def __init__(
        self,
        vehicle,
        behavior="normal",
        opt_dict=None,
        map_inst=None,
        grp_inst=None,
        command_counts=None,
        rng=None,
        intentions=None,
        reservations=None,
        balance_turns=False,
    ):
        if opt_dict is None:
            opt_dict = {}
        opt_dict = dict(opt_dict)
        opt_dict.setdefault("sampling_radius", 1.0)
        opt_dict.setdefault("base_min_distance", 1.0)

        opt_dict.setdefault("use_bbs_detection", True)
        opt_dict.setdefault(
            "lateral_control_dict",
            {
                "K_P": 2.7,
                "K_I": 0.05,
                "K_D": 0.3,
                "dt": 1.0 / 20.0,
            },
        )
        super().__init__(
            vehicle,
            behavior=behavior,
            opt_dict=opt_dict,
            map_inst=map_inst,
            grp_inst=grp_inst,
        )

        self._behavior.max_speed = min(self._behavior.max_speed, self.NORMAL_SPEED)

        self._balance_turns = balance_turns
        self._command_counts = command_counts if command_counts is not None else {}
        self._rng = rng or random.Random()
        self._committed_junction = False
        self._entered_committed = False

        self._forced_command = None
        self._forced_entrance = None
        self._forced_path = None
        self._passed_entrance = None

        self._duckie_wait_start = None
        self._duckie_wait_id = None

        self._stop_signs_list = self._world.get_actors().filter("traffic.stop")
        self._stop_sign_wps = {}
        self._served_stop_signs = {}
        self._stop_wait_start = None

        self._yield_start = None
        self._junction_go_id = None
        self._junction_stops = {}

        self._intentions = intentions if intentions is not None else {}

        self._reservations = reservations if reservations is not None else {}

    def _duckie_ahead(self, ego_wp):
        ego_transform = self._vehicle.get_transform()
        ego_loc = ego_transform.location
        next_wp = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
        nearest, nearest_dist = None, float("inf")
        for d in self._world.get_actors().filter("static.prop.rubberduckie"):
            if d.get_location().distance(ego_loc) >= self.DUCKIE_DETECT_DIST:
                continue
            d_wp = self._map.get_waypoint(d.get_location(), project_to_road=True)
            same_lane = (
                d_wp.road_id == ego_wp.road_id and d_wp.lane_id == ego_wp.lane_id
            )
            if not same_lane and next_wp is not None:
                same_lane = (
                    d_wp.road_id == next_wp.road_id and d_wp.lane_id == next_wp.lane_id
                )
            if not same_lane:
                continue

            if d.get_location().distance(d_wp.transform.location) > 1.0:
                continue
            if not is_within_distance(
                d.get_transform(), ego_transform, self.DUCKIE_DETECT_DIST, [0, 60]
            ):
                continue
            dist = d.get_location().distance(ego_loc)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = d
        return (True, nearest, nearest_dist) if nearest else (False, None, -1)

    def duckie_manager(self, ego_wp, now):
        detected, duckie, dist = self._duckie_ahead(ego_wp)
        if not detected:
            self._duckie_wait_start = None
            self._duckie_wait_id = None
            return False, None
        gap = dist - self._vehicle.bounding_box.extent.x
        if gap <= self.DUCKIE_STOP_GAP:
            if self._duckie_wait_id != duckie.id:
                self._duckie_wait_id = duckie.id
                self._duckie_wait_start = now
            elif (
                self.DUCKIE_WAIT_TIMEOUT > 0
                and now - self._duckie_wait_start >= self.DUCKIE_WAIT_TIMEOUT
            ):
                try:
                    duckie.destroy()
                except RuntimeError:
                    pass
                self._duckie_wait_start = None
                self._duckie_wait_id = None
                return False, None
            return True, None

        self._duckie_wait_start = None
        self._duckie_wait_id = None
        return False, self._ramp_speed(gap, self.DUCKIE_STOP_GAP, 10.0)

    def _stop_sign_ahead(self, ego_wp):
        ego_transform = self._vehicle.get_transform()
        ego_loc = ego_transform.location
        best_sign, best_wp, best_dist = None, None, float("inf")

        plan_locs = [
            wp.transform.location for wp, _ in list(self._local_planner.get_plan())[:20]
        ]

        for sign in self._stop_signs_list:
            if sign.id in self._served_stop_signs:
                continue

            trigger_wp = self._stop_sign_wps.get(sign.id)
            if trigger_wp is None:
                trigger_wp = self._map.get_waypoint(
                    get_trafficlight_trigger_location(sign)
                )
                self._stop_sign_wps[sign.id] = trigger_wp

            dist = trigger_wp.transform.location.distance(ego_loc)
            if dist > self.STOP_SIGN_DETECT_DIST:
                continue

            sign_loc = trigger_wp.transform.location
            is_on_path = False
            for plan_loc in plan_locs:
                if plan_loc.distance(sign_loc) < 3.5:
                    is_on_path = True
                    break

            if not is_on_path:
                continue

            if not is_within_distance(
                trigger_wp.transform, ego_transform, self.STOP_SIGN_DETECT_DIST, [0, 90]
            ):
                continue

            if dist < best_dist:
                best_sign, best_wp, best_dist = sign, trigger_wp, dist

        if best_sign is None:
            return None, None, -1
        return best_sign, best_wp, best_dist

    def stop_sign_manager(self, ego_wp, now):
        ego_loc = self._vehicle.get_location()
        for sid in list(self._served_stop_signs):
            if self._served_stop_signs[sid].transform.location.distance(ego_loc) > 20.0:
                del self._served_stop_signs[sid]

        if ego_wp.is_junction:
            self._stop_wait_start = None
            return False, None

        sign, trigger_wp, dist = self._stop_sign_ahead(ego_wp)
        if sign is None:
            self._stop_wait_start = None
            return False, None

        if self._stop_wait_start is not None:
            if now - self._stop_wait_start >= 1.0:
                if self._stop_yield_blocked(now):
                    return True, None
                self._served_stop_signs[sign.id] = trigger_wp
                self._stop_wait_start = None
                return False, None
            return True, None

        if dist <= self.STOP_SIGN_BRAKE_GAP:
            if get_speed(self._vehicle) < 1.0:
                self._stop_wait_start = now
            return True, None

        return False, self._ramp_speed(
            dist, self.STOP_SIGN_BRAKE_GAP, self.STOP_SIGN_DETECT_DIST
        )

    def _stop_yield_blocked(self, now):
        if not self.YIELD_RIGHT_BEFORE_LEFT:
            return False
        junction, _, _ = self._upcoming_junction()
        if junction is None:
            return False
        return self._yield_decision(junction, self._forced_roadoption(), now)

    def _apply_junction_choice(self, ego_wp):
        if self._committed_junction:
            if ego_wp.is_junction:
                self._entered_committed = True
                return
            if not self._entered_committed:
                return

            self._committed_junction = False
            self._entered_committed = False
            self._passed_entrance = self._forced_entrance
            self._forced_command = None
            self._forced_entrance = None
            self._forced_path = None

        if ego_wp.is_junction:
            return
        ego_loc = self._vehicle.get_location()

        if self._passed_entrance is not None:
            if ego_loc.distance(self._passed_entrance) < self.JUNCTION_CHOICE_LOOKAHEAD:
                return
            self._passed_entrance = None

        actions, entrance_index = available_junction_actions(
            ego_wp, self.JUNCTION_CHOICE_LOOKAHEAD
        )
        if not actions:
            return

        if self._balance_turns:
            fewest = min(self._command_counts.get(cmd, 0) for cmd in actions)
            cmd = self._rng.choice(
                [c for c in actions if self._command_counts.get(c, 0) == fewest]
            )
            self._command_counts[cmd] = self._command_counts.get(cmd, 0) + 1
        else:
            cmd = self._rng.choice(list(actions))

        path = actions[cmd]
        plan = [
            (
                wp,
                _COMMAND_TO_ROADOPTION[cmd]
                if i == entrance_index
                else RoadOption.LANEFOLLOW,
            )
            for i, wp in enumerate(path)
        ]
        self._local_planner.set_global_plan(
            plan, stop_waypoint_creation=False, clean_queue=True
        )
        self._committed_junction = True
        self._entered_committed = False
        self._forced_command = cmd
        self._forced_entrance = path[entrance_index].transform.location
        self._forced_path = path

    def _upcoming_junction(self):
        prev = self._vehicle.get_location()
        dist = 0.0
        for wp, option in self._local_planner.get_plan():
            loc = wp.transform.location
            dist += loc.distance(prev)
            prev = loc
            if dist > 8.0:
                break
            if wp.is_junction:
                return wp.get_junction(), dist, option
        return None, -1, None

    def _junction_has_stop_sign(self, junction):
        if junction.id not in self._junction_stops:
            center = junction.bounding_box.location
            has_stop = any(
                get_trafficlight_trigger_location(s).distance(center)
                < self.JUNCTION_SCAN_DIST
                for s in self._stop_signs_list
            )
            self._junction_stops[junction.id] = has_stop
        return self._junction_stops[junction.id]

    def _junction_conflicts(self, junction, turning_left):
        center = junction.bounding_box.location
        ego_loc = self._vehicle.get_location()
        ego_fwd = self._vehicle.get_transform().get_forward_vector()
        conflicts = []
        for v in self._world.get_actors().filter("*vehicle*"):
            if v.id == self._vehicle.id:
                continue
            v_loc = v.get_location()
            if v_loc.distance(center) > self.JUNCTION_SCAN_DIST:
                continue
            v_wp = self._map.get_waypoint(v_loc)
            if v_wp.is_junction and v_wp.get_junction().id == junction.id:
                conflicts.append(v)
                continue
            v_fwd = v.get_transform().get_forward_vector()
            to_c = center - v_loc
            if v_fwd.x * to_c.x + v_fwd.y * to_c.y <= 0.0:
                continue
            to_v = v_loc - ego_loc
            angle = math.degrees(
                math.atan2(
                    ego_fwd.x * to_v.y - ego_fwd.y * to_v.x,
                    ego_fwd.x * to_v.x + ego_fwd.y * to_v.y,
                )
            )
            from_right = 15.0 < angle < 165.0
            oncoming = (
                turning_left
                and -60.0 < angle < 60.0
                and v_fwd.x * ego_fwd.x + v_fwd.y * ego_fwd.y < -0.5
            )
            if from_right:
                conflicts.append(v)
            elif oncoming and self._intentions.get(v.id) != RoadOption.LEFT:
                conflicts.append(v)
        return conflicts

    def _yield_decision(self, junction, option, now):
        if self._junction_go_id == junction.id:
            self._reservations[junction.id] = (self._vehicle.id, now)
            return False
        conflicts = self._junction_conflicts(junction, option == RoadOption.LEFT)
        if conflicts:
            if self._yield_start is None:
                self._yield_start = now
            deadlock_broken = (
                now - self._yield_start > 4.0
                and all(get_speed(v) < 1.0 for v in conflicts)
                and self._vehicle.id < min(v.id for v in conflicts)
            )
            if not deadlock_broken:
                return True
        self._yield_start = None

        if self._claim_junction(junction.id, now):
            self._junction_go_id = junction.id
            return False
        return True

    def _claim_junction(self, junction_id, now):
        claim = self._reservations.get(junction_id)
        if claim is not None:
            holder_id, ts = claim
            if holder_id != self._vehicle.id and (now - ts) < 6.0:
                return False
        self._reservations[junction_id] = (self._vehicle.id, now)
        return True

    def junction_priority_manager(self, ego_wp, now):
        if not self.YIELD_RIGHT_BEFORE_LEFT:
            return False
        if ego_wp.is_junction:
            self._yield_start = None
            return False

        junction, _, _ = self._upcoming_junction()
        if junction is None:
            self._yield_start = None
            if self._junction_go_id is not None:
                if (
                    self._reservations.get(self._junction_go_id, (None, None))[0]
                    == self._vehicle.id
                ):
                    self._reservations.pop(self._junction_go_id, None)
                self._junction_go_id = None
            return False
        if self._junction_has_stop_sign(junction):
            return False
        return self._yield_decision(junction, self._forced_roadoption(), now)

    def _ramp_speed(self, gap, stop_gap, slow_gap):
        frac = (gap - stop_gap) / (slow_gap - stop_gap)
        frac = max(0.0, min(1.0, frac))
        return self.CREEP_SPEED + frac * (self.NORMAL_SPEED - self.CREEP_SPEED)

    def anchor_path(self):
        return self._forced_path

    def command_label(self, report_distance_m=10.0):
        if self._forced_command is None:
            return Command.DEFAULT
        if (
            self._vehicle.get_location().distance(self._forced_entrance)
            <= report_distance_m
        ):
            return self._forced_command
        return Command.DEFAULT

    def _forced_roadoption(self):
        if self._forced_command is None:
            return RoadOption.LANEFOLLOW
        return _COMMAND_TO_ROADOPTION[self._forced_command]

    def run_step(self, debug=False):
        self._update_information()
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_wp = self._map.get_waypoint(self._vehicle.get_location())
        now = self._world.get_snapshot().timestamp.elapsed_seconds

        self._apply_junction_choice(ego_wp)

        self._intentions[self._vehicle.id] = (
            _COMMAND_TO_ROADOPTION[self._forced_command]
            if self._forced_command is not None
            else None
        )

        speed_cap = self.NORMAL_SPEED

        hold, cap = self.stop_sign_manager(ego_wp, now)
        if hold:
            return self.emergency_stop()
        if cap is not None:
            speed_cap = min(speed_cap, cap)

        hold, cap = self.duckie_manager(ego_wp, now)
        if hold:
            return self.emergency_stop()
        if cap is not None:
            speed_cap = min(speed_cap, cap)

        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_wp)
        if walker_state:
            distance = (
                w_distance
                - max(walker.bounding_box.extent.y, walker.bounding_box.extent.x)
                - max(
                    self._vehicle.bounding_box.extent.y,
                    self._vehicle.bounding_box.extent.x,
                )
            )
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()

        try:
            vehicle_state, vehicle, v_gap = self.collision_and_car_avoid_manager(ego_wp)
        except GEOSException:
            return self.emergency_stop()
        if vehicle_state:
            v_gap = (
                v_gap
                - max(vehicle.bounding_box.extent.y, vehicle.bounding_box.extent.x)
                - max(
                    self._vehicle.bounding_box.extent.y,
                    self._vehicle.bounding_box.extent.x,
                )
            )
            if v_gap < self._behavior.braking_distance:
                return self.emergency_stop()

        if self.junction_priority_manager(ego_wp, now):
            return self.emergency_stop()

        if vehicle_state:
            return self.car_following_manager(vehicle, v_gap)

        self._local_planner.set_speed(
            min(
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist,
                speed_cap,
            )
        )
        return self._local_planner.run_step(debug=False)


def build_global_route_planner(carla_map, sampling_resolution=2.0):
    from ...carla_agents.navigation.global_route_planner import GlobalRoutePlanner

    return GlobalRoutePlanner(carla_map, sampling_resolution)


class AgentDriver:
    def __init__(
        self,
        carla_map,
        command_counts,
        grp=None,
        report_distance_m=5.0,
        rng=None,
        duckie_wait_timeout=None,
    ):
        self._map = carla_map
        self._command_counts = command_counts
        self._grp = grp
        self._report_distance_m = report_distance_m
        self._rng = rng or random.Random()

        self._duckie_wait_timeout = duckie_wait_timeout
        self.ego = None
        self._ego_vehicle = None
        self._traffic = []
        self._intentions = {}
        self._junction_reservations = {}

    def set_ego(self, vehicle, target_speed_kmh=None):
        self._ego_vehicle = vehicle
        self.ego = DuckieBehaviorAgent(
            vehicle,
            map_inst=self._map,
            grp_inst=self._grp,
            command_counts=self._command_counts,
            rng=self._rng,
            intentions=self._intentions,
            reservations=self._junction_reservations,
            balance_turns=True,
        )
        if self._duckie_wait_timeout is not None:
            self.ego.DUCKIE_WAIT_TIMEOUT = self._duckie_wait_timeout
        if target_speed_kmh is not None:
            self.ego.NORMAL_SPEED = target_speed_kmh
            self.ego._behavior.max_speed = min(
                self.ego._behavior.max_speed, target_speed_kmh
            )

    def reset_traffic(self):
        self._traffic = []

    def add_traffic(self, vehicle, target_speed_kmh):
        agent = DuckieBehaviorAgent(
            vehicle,
            map_inst=self._map,
            grp_inst=self._grp,
            rng=self._rng,
            intentions=self._intentions,
            reservations=self._junction_reservations,
        )
        if self._duckie_wait_timeout is not None:
            agent.DUCKIE_WAIT_TIMEOUT = self._duckie_wait_timeout
        agent.NORMAL_SPEED = target_speed_kmh
        agent._behavior.max_speed = min(agent._behavior.max_speed, target_speed_kmh)
        self._traffic.append((agent, vehicle))

    def control_tick(self):
        if self.ego is not None:
            try:
                self._ego_vehicle.apply_control(self.ego.run_step())
            except (RuntimeError, IndexError):
                if self._ego_vehicle.is_alive:
                    self._ego_vehicle.destroy()
                raise RuntimeError("ego route exhausted / actor lost at a dead end")
        alive = []
        for agent, vehicle in self._traffic:
            try:
                vehicle.apply_control(agent.run_step())
                alive.append((agent, vehicle))
            except (RuntimeError, IndexError):
                pass
        self._traffic = alive

    def label(self, vehicle):
        return int(
            self.ego.command_label(self._report_distance_m)
        ), self.ego.anchor_path()
