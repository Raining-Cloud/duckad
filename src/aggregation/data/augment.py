from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np

from .background import ImageNetBackgroundCache


class RandomTerrainBackground(A.ImageOnlyTransform):
    def __init__(self, cache: ImageNetBackgroundCache, p: float = 0.5):
        super().__init__(p=p)
        self.cache = cache

    def get_params_dependent_on_data(
        self, params: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, Any]:
        mask = data.get("mask")
        if mask is None:
            return {"background": None, "mask": None}
        bg = self.cache.sample()
        h, w = params["shape"][:2]
        if bg.shape[:2] != (h, w):
            bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)
        return {"background": bg, "mask": mask}

    def apply(
        self,
        img: np.ndarray,
        background: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        **params: Any,
    ) -> np.ndarray:
        if background is None or mask is None:
            return img
        return np.where(mask[..., None] == 255, background, img).astype(img.dtype)


def build_train_transform(
    background_cache: ImageNetBackgroundCache | None = None,
) -> A.Compose:
    """Photometric-only augmentation for the front-camera INPUT image - never the BEV mask or
    the trajectory. Every transform here perturbs pixel colors/exposure or paints occlusions,
    but none changes geometry, so it's safe to apply to the image alone: the recorded BEV
    label map and future-trajectory targets stay valid untouched. If a spatial transform
    (flip/affine/crop) is ever added, it would also have to warp the mask (and trajectory),
    which this image-only pipeline deliberately does not do.

    No ToTensorV2 / normalization here on purpose: BatchedTrajectoryDataset does the uint8 ->
    float [0, 1] CHW conversion itself, matching eval.inference's preprocessing so train and
    eval see identically-scaled inputs. Operates on and returns an HWC uint8 image.

    (Images are stored BGR; the hue/color transforms interpret them as RGB, so the exact hue
    rotation differs from a true-RGB image - harmless for augmentation, and it keeps train/eval
    channel order consistent since eval feeds BGR too.)

    If `background_cache` is given, a RandomTerrainBackground (p=0.5) runs FIRST, replacing the
    terrain region with a random ImageNet image before the photometric transforms - so the
    inserted background picks up the same color/exposure jitter and blends in. This needs the
    per-sample terrain mask passed as `mask=` to the Compose call (see dataset.py).
    """
    background = (
        [RandomTerrainBackground(background_cache, p=0.5)]
        if background_cache is not None
        else []
    )
    return A.Compose(
        [
            *background,
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.01, 0.05)),
                    A.ChromaticAberration(),
                    A.ImageCompression(
                        quality_range=(40, 100), compression_type="jpeg"
                    ),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5, p=1),
                    A.Blur(blur_limit=3, p=1),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=1
                    ),
                    A.CLAHE(clip_limit=2.0, p=1),
                    A.RandomGamma(gamma_limit=(80, 120), p=1),
                ],
                p=0.8,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5
            ),
            A.OneOf(
                [
                    A.CoarseDropout(
                        num_holes_range=(1, 4),
                        hole_height_range=(1, 32),
                        hole_width_range=(1, 32),
                        fill=0,
                    ),
                    A.CoarseDropout(
                        num_holes_range=(3, 10),
                        hole_height_range=(1, 2),
                        hole_width_range=(200, 500),
                        fill="random_uniform",
                    ),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        src_radius=50,
                        src_color=(255, 255, 255),
                        p=1,
                    ),
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_limit=(1, 3),
                        shadow_dimension=4,
                    ),
                ],
                p=0.3,
            ),
        ]
    )
