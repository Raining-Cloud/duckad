from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from ..config import CACHE_SIZE, IMAGENET_DIR


class ImageNetBackgroundCache:
    def __init__(self, size: int = 224, seed: int | None = None):
        self.root = Path(IMAGENET_DIR)
        self.size = size
        self.cache_size = CACHE_SIZE
        self._seed = seed
        self._images: np.ndarray | None = None

    def _build(self) -> np.ndarray:
        paths = sorted(self.root.rglob("*.JPEG"))

        if not paths:
            raise FileNotFoundError(f"No *.JPEG images found under {self.root}")

        rng = random.Random(self._seed)
        chosen = [rng.choice(paths) for _ in range(self.cache_size)]

        images = np.empty((self.cache_size, self.size, self.size, 3), dtype=np.uint8)
        filled = 0
        for path in chosen:
            img = cv2.imread(
                str(path), cv2.IMREAD_COLOR
            )  # BGR, HxWx3; None on decode failure
            if img is None:
                continue
            images[filled] = self._cover_crop(img)
            filled += 1
        if filled == 0:
            raise RuntimeError(
                f"Could not decode any ImageNet images under {self.root}"
            )

        return images[:filled]

    def _cover_crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.size / min(h, w)
        new_w, new_h = (
            max(self.size, round(w * scale)),
            max(self.size, round(h * scale)),
        )
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y0, x0 = (new_h - self.size) // 2, (new_w - self.size) // 2

        return resized[y0 : y0 + self.size, x0 : x0 + self.size]

    def sample(self) -> np.ndarray:
        if self._images is None:
            self._images = self._build()

        return self._images[random.randrange(len(self._images))]
