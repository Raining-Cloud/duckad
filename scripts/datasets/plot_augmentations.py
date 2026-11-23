"""Render the augmentation showcase figure for the presentation.

Every panel applies ONE transform from data.augment.build_train_transform to the SAME base
frame, with parameters pinned to representative values inside the training ranges, so
the effect of each transform is isolated and directly comparable. A bottom strip then
draws the FULL pipeline several times on that same frame - the epoch-to-epoch variety
the network actually trains on. Run from the repo root (IMAGENET_DIR is relative).

Usage: uv run python scripts/datasets/plot_augmentations.py [--episode DIR --frame ID --rig N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aggregation.data.augment import RandomTerrainBackground, build_train_transform
from src.aggregation.data.background import ImageNetBackgroundCache

# Base frame: head-on junction approach on duckietown_05 - red stop line, center dashes,
# signs on both sides, and a large visible terrain region (so the background swap reads
# clearly on the slide).
EPISODE = "data/final/ducks_obstacle/duckietown_05_lab_01_20260711-141138"
FRAME = 85903
RIG = 1  # row within the tick: 0 front, 1 left, 2 right
OUT = Path("figures/augmentations.png")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GROUP_COLORS = {
    "input": "#52514e",
    "background": "#2a78d6",
    "noise": "#eb6834",
    "blur": "#1baf7a",
    "tone": "#eda100",
    "color": "#e87ba4",
    "occlusion": "#4a3aa7",
    "lighting": "#e34948",
}

cache = ImageNetBackgroundCache(seed=7)


def panels() -> list[tuple[str, str, str, A.BasicTransform | None, int]]:
    """(title, subtitle, group, transform, seed) per grid cell, reading order."""
    return [
        ("Original", "224×224 front camera", "input", None, 0),
        ("Background swap", "ImageNet terrain · p=0.5", "background",
         RandomTerrainBackground(cache, p=1), 12),
        ("Gaussian noise", "σ=0.05 · group p=0.3", "noise",
         A.GaussNoise(std_range=(0.05, 0.05), p=1), 3),
        ("Chromatic aberration", "group p=0.3", "noise",
         A.ChromaticAberration(p=1), 4),
        ("JPEG compression", "quality 40 · group p=0.3", "noise",
         A.ImageCompression(quality_range=(40, 40), compression_type="jpeg", p=1), 5),
        ("Motion blur", "kernel 5 · group p=0.4", "blur",
         A.MotionBlur(blur_limit=(5, 5), p=1), 6),
        ("Box blur", "kernel 3 · group p=0.4", "blur",
         A.Blur(blur_limit=(3, 3), p=1), 7),
        ("Brightness / contrast", "±0.2 · group p=0.8", "tone",
         A.RandomBrightnessContrast(brightness_limit=(-0.2, -0.2),
                                    contrast_limit=(0.2, 0.2), p=1), 8),
        ("CLAHE", "clip 2.0 · group p=0.8", "tone",
         A.CLAHE(clip_limit=(2.0, 2.0), p=1), 9),
        ("Gamma", "γ=0.8 · group p=0.8", "tone",
         A.RandomGamma(gamma_limit=(80, 80), p=1), 10),
        ("Hue / saturation", "h±10 s±20 v±10 · p=0.5", "color",
         A.HueSaturationValue(hue_shift_limit=(10, 10), sat_shift_limit=(20, 20),
                              val_shift_limit=(10, 10), p=1), 11),
        ("Dropout blocks", "≤4 holes ≤32px · p=0.3", "occlusion",
         A.CoarseDropout(num_holes_range=(4, 4), hole_height_range=(12, 32),
                         hole_width_range=(12, 32), fill=0, p=1), 13),
        ("Dropout streaks", "3–10 lines · p=0.3", "occlusion",
         A.CoarseDropout(num_holes_range=(6, 6), hole_height_range=(1, 2),
                         hole_width_range=(200, 500), fill="random_uniform", p=1), 14),
        ("Sun flare", "upper half · group p=0.3", "lighting",
         A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=50,
                          src_color=(255, 255, 255), p=1), 161),
        ("Shadow", "lower half · group p=0.3", "lighting",
         A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(2, 3),
                        shadow_dimension=4, p=1), 16),
    ]


def apply_one(img: np.ndarray, seg: np.ndarray, t: A.BasicTransform | None, seed: int) -> np.ndarray:
    if t is None:
        return img
    compose = A.Compose([t], seed=seed)
    return compose(image=img, mask=seg)["image"]


def load_tick(episode: str, frame: int) -> dict[str, np.ndarray]:
    """The rows recorded at one frame id (one per rig, front/left/right). Located by episode
    and frame id, not shard name, so it works on both raw recordings and the Hub rebuild."""
    for shard in sorted(Path(episode).glob("batch_*.npz")):
        with np.load(shard) as d:
            hit = np.flatnonzero(d["frame"] == frame)
            if hit.size:
                return {k: d[k][hit] for k in d.files if d[k].ndim > 0}
    raise FileNotFoundError(f"frame {frame} not found under {episode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episode", default=EPISODE, help="episode directory of batch_*.npz")
    parser.add_argument("--frame", type=int, default=FRAME, help="frame id of the tick to draw")
    parser.add_argument("--rig", type=int, default=RIG, help="0 front, 1 left, 2 right")
    args = parser.parse_args()
    data = load_tick(args.episode, args.frame)
    img = data["main_img"][args.rig]  # HWC uint8 BGR
    seg = data["main_seg"][args.rig]

    n_strip = 5
    n_cols = 5
    fig = plt.figure(figsize=(n_cols * 2.35, 4 * 2.5 + 0.85), facecolor=SURFACE)
    gs = fig.add_gridspec(5, n_cols,
                          height_ratios=[1, 1, 1, 0.14, 1],
                          left=0.012, right=0.988, top=0.925, bottom=0.012,
                          hspace=0.22, wspace=0.10)

    fig.suptitle("Augmentations", x=0.015, y=0.975,
                 ha="left", fontsize=20, fontweight="bold", color=INK)

    for i, (title, subtitle, group, t, seed) in enumerate(panels()):
        ax = fig.add_subplot(gs[i // n_cols, i % n_cols])
        out = apply_one(img, seg, t, seed)
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        color = GROUP_COLORS[group]
        # Accent bar + large title above each panel; text stays in ink, the bar alone
        # carries the group. Subtitles/parameters are omitted for slide readability.
        ax.plot([0, 40], [-8, -8], color=color, lw=3.5, solid_capstyle="butt",
                clip_on=False, transform=ax.transData)
        ax.text(0, -16, title, fontsize=13, fontweight="bold", color=INK,
                va="bottom", clip_on=False)

    strip_ax = fig.add_subplot(gs[3, :])
    strip_ax.set_axis_off()
    strip_ax.text(0, 0.1, "Full Pipeline",
                  fontsize=15, fontweight="bold", color=INK, va="bottom")

    # The background cache draws from process-global RNG state, so a fixed Compose seed
    # alone doesn't pin the strip's look. Instead: render a pool of full-pipeline draws,
    # rank by pixel deviation from the original, and keep a mild-to-strong quantile
    # spread - representative of training variety and deterministic for this script.
    transform = build_train_transform(cache)
    pool = []
    for seed in range(40):
        out = np.asarray(A.Compose(transform.transforms, seed=seed)(image=img, mask=seg)["image"])
        pool.append((np.abs(out.astype(int) - img.astype(int)).mean(), out))
    pool.sort(key=lambda pair: pair[0])
    picks = [pool[round(q * (len(pool) - 1))][1]
             for q in np.linspace(0.1, 1.0, n_strip)]
    for j, out in enumerate(picks):
        ax = fig.add_subplot(gs[4, j])
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
