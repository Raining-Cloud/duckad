from __future__ import annotations

import cv2
import numpy as np

from .config import TargetImageConfig

DUCKIETOWN_LABELS = {
    "center_lane": 29,
    "side_lane": 30,
    "asphalt": 31,
    "stop_lane": 32,
    "sign": 33,
    "bot": 34,
    "duck": 35,
}

BEV_BACKGROUND_LABEL = "background"
BEV_BACKGROUND_INDEX = 0
BEV_CLASS_NAMES = [BEV_BACKGROUND_LABEL, *DUCKIETOWN_LABELS.keys()]
BEV_NUM_CLASSES = len(BEV_CLASS_NAMES)
_TAG_TO_CLASS_INDEX = {
    tag_id: index for index, tag_id in enumerate(DUCKIETOWN_LABELS.values(), start=1)
}

FOREGROUND_TAG_IDS = tuple(sorted(DUCKIETOWN_LABELS.values()))


def letterbox(
    image: np.ndarray, target: TargetImageConfig, interpolation: int = cv2.INTER_AREA
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = target.size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    channels = image.shape[2] if image.ndim == 3 else 1
    canvas_shape = (
        (target.size, target.size, channels)
        if image.ndim == 3
        else (target.size, target.size)
    )
    canvas = np.full(canvas_shape, target.pad_color, dtype=image.dtype)
    y0, x0 = (target.size - new_h) // 2, (target.size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def bev_semantic_labels(bev_bgr: np.ndarray) -> np.ndarray:
    return bev_bgr[:, :, 2:3].transpose(2, 0, 1)


def bev_class_index(bev_seg: np.ndarray) -> np.ndarray:
    ids = bev_seg[..., 0, :, :]
    class_index = np.full(ids.shape, BEV_BACKGROUND_INDEX, dtype=np.int64)
    for tag_id, index in _TAG_TO_CLASS_INDEX.items():
        class_index[ids == tag_id] = index
    return class_index


def background_mask(seg_bgr: np.ndarray) -> np.ndarray:
    ids = seg_bgr[:, :, 2]
    return np.where(np.isin(ids, FOREGROUND_TAG_IDS), 0, 255).astype(np.uint8)


def carla_image_to_bgr(carla_image) -> np.ndarray:
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    return array[:, :, :3]
