from __future__ import annotations

import queue

import carla

from ..config import BEVConfig, CameraRigConfig


def to_carla_transform(cfg, yaw_offset: float = 0.0) -> carla.Transform:
    return carla.Transform(
        carla.Location(x=cfg.x, y=cfg.y, z=cfg.z),
        carla.Rotation(pitch=cfg.pitch, yaw=cfg.yaw + yaw_offset, roll=cfg.roll),
    )


def rig_ground_transform(
    vehicle_transform: carla.Transform, rig: CameraRigConfig, yaw_offset: float = 0.0
) -> carla.Transform:
    shifted_location = vehicle_transform.transform(
        carla.Location(0.0, rig.transform.y, 0.0)
    )
    rot = vehicle_transform.rotation
    return carla.Transform(
        shifted_location,
        carla.Rotation(pitch=rot.pitch, yaw=rot.yaw + yaw_offset, roll=rot.roll),
    )


def spawn_main_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
    rig: CameraRigConfig,
    yaw_offset: float = 0.0,
) -> carla.Sensor:
    bp = world.get_blueprint_library().find("sensor.camera.rgb.wide_angle_lens")
    bp.set_attribute("image_size_x", str(rig.image_size_x))
    bp.set_attribute("image_size_y", str(rig.image_size_y))
    bp.set_attribute("camera_model", rig.camera_model)
    bp.set_attribute("fov", str(rig.fov_deg))
    bp.set_attribute("role_name", rig.role_name)
    return world.spawn_actor(
        bp, to_carla_transform(rig.transform, yaw_offset), attach_to=vehicle
    )


def spawn_main_segmentation_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
    rig: CameraRigConfig,
    yaw_offset: float = 0.0,
) -> carla.Sensor:
    bp = world.get_blueprint_library().find(
        "sensor.camera.semantic_segmentation.wide_angle_lens"
    )
    bp.set_attribute("image_size_x", str(rig.image_size_x))
    bp.set_attribute("image_size_y", str(rig.image_size_y))
    bp.set_attribute("camera_model", rig.camera_model)
    bp.set_attribute("fov", str(rig.fov_deg))
    return world.spawn_actor(
        bp, to_carla_transform(rig.transform, yaw_offset), attach_to=vehicle
    )


def spawn_bev_camera(
    world: carla.World, vehicle: carla.Vehicle, cfg: BEVConfig, yaw_offset: float = 0.0
) -> carla.Sensor:
    bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
    bp.set_attribute("image_size_x", str(cfg.image_size))
    bp.set_attribute("image_size_y", str(cfg.image_size))
    bp.set_attribute("fov", str(cfg.fov_deg))
    return world.spawn_actor(
        bp, to_carla_transform(cfg.transform, yaw_offset), attach_to=vehicle
    )


class SensorSync:
    def __init__(self, world: carla.World, sensors: dict[str, carla.Sensor]):
        self.world = world
        self.sensors = sensors
        self._queues: dict[str, queue.Queue] = {}
        for name, sensor in sensors.items():
            q = queue.Queue()
            sensor.listen(q.put)
            self._queues[name] = q

    def tick(self, timeout: float = 2.0) -> dict:
        frame = self.world.tick()
        data = {
            name: self._retrieve(q, frame, timeout) for name, q in self._queues.items()
        }
        return frame, data

    @staticmethod
    def _retrieve(q: queue.Queue, frame: int, timeout: float):
        while True:
            data = q.get(timeout=timeout)
            if data.frame == frame:
                return data

    def destroy(self) -> None:
        for sensor in self.sensors.values():
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
