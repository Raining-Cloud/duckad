from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..imaging import BEV_NUM_CLASSES, bev_class_index


class BatchedTrajectoryDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        normalization_constants: dict[str, float] | None = None,
        cache_dir: str | Path | None = None,
        transform=None,
    ):
        self.transform = transform
        self.root = Path(root)
        self.batch_paths = sorted(self.root.rglob("batch_*.npz"))

        if not self.batch_paths:
            raise FileNotFoundError(f"No batch_*.npz files found under {self.root}")

        with np.load(self.batch_paths[0]) as first:
            self.num_waypoints = int(first["trajectory"].shape[1])
            self.bev_classes, self.bev_size = (
                BEV_NUM_CLASSES,
                int(first["bev_seg"].shape[2]),
            )
            self.image_size = int(first["main_img"].shape[1])
            self.num_temporal_points = int(first["temporal_trajectory"].shape[1])

        batch_sizes = []
        for path in self.batch_paths:
            with np.load(path) as data:
                batch_sizes.append(int(data["speed"].shape[0]))
        self._cum_sizes = np.cumsum(batch_sizes).tolist()
        self._length = int(self._cum_sizes[-1])

        constants = (
            normalization_constants
            if normalization_constants is not None
            else json.loads((self.root / "trajectory_norm.json").read_text())
        )
        self._traj_min = torch.tensor(
            [constants["x_min"], constants["y_min"]], dtype=torch.float32
        )
        self._traj_scale = torch.tensor(
            [
                2.0 / (constants["x_max"] - constants["x_min"]),
                2.0 / (constants["y_max"] - constants["y_min"]),
            ],
            dtype=torch.float32,
        )

        self._temporal_min = torch.tensor(
            [constants["tx_min"], constants["ty_min"]], dtype=torch.float32
        )
        self._temporal_scale = torch.tensor(
            [
                2.0 / (constants["tx_max"] - constants["tx_min"]),
                2.0 / (constants["ty_max"] - constants["ty_min"]),
            ],
            dtype=torch.float32,
        )

        cache_dir = (
            Path(cache_dir) if cache_dir is not None else self.root / ".mmap_cache"
        )
        self._mmaps = self._open_or_build_cache(cache_dir, batch_sizes)
        self.temporal_dt = float(self._mmaps["temporal_dt"][0])

    def _open_or_build_cache(
        self, cache_dir: Path, batch_sizes: list[int]
    ) -> dict[str, np.ndarray]:
        manifest_path = cache_dir / "manifest.json"
        keys = self._cache_keys()

        expected = {
            "length": self._length,
            "shards": [[p.name, n] for p, n in zip(self.batch_paths, batch_sizes)],
        }
        if manifest_path.exists():
            try:
                if json.loads(manifest_path.read_text()) == expected:
                    return {k: self._open_memmap(cache_dir / f"{k}.npy") for k in keys}
            except (OSError, ValueError):
                pass

        self._build_cache(cache_dir, expected, keys, batch_sizes)
        return {k: self._open_memmap(cache_dir / f"{k}.npy") for k in keys}

    def _cache_keys(self) -> tuple[str, ...]:
        with np.load(self.batch_paths[0]) as first:
            return tuple(first.files)

    @staticmethod
    def _open_memmap(path: Path) -> np.ndarray:
        return np.lib.format.open_memmap(path, mode="r")

    def _build_cache(
        self,
        cache_dir: Path,
        manifest: dict,
        keys: tuple[str, ...],
        batch_sizes: list[int],
    ) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"building memmap cache for {len(self.batch_paths)} shards under {cache_dir} ..."
        )

        with np.load(self.batch_paths[0]) as first:
            writers = {
                k: np.lib.format.open_memmap(
                    cache_dir / f"{k}.npy",
                    mode="w+",
                    dtype=first[k].dtype,
                    shape=(self._length, *first[k].shape[1:]),
                )
                for k in keys
            }

        offset = 0
        for i, (path, n) in enumerate(zip(self.batch_paths, batch_sizes)):
            with np.load(path) as data:
                for k in keys:
                    writers[k][offset : offset + n] = data[k]
            offset += n
            print(
                f"  cached shard {i + 1}/{len(self.batch_paths)}", end="\r", flush=True
            )

        for w in writers.values():
            w.flush()
        del writers
        (cache_dir / "manifest.json").write_text(json.dumps(manifest))
        print(f"\nmemmap cache ready ({offset} samples)")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self._length:
            raise IndexError(index)

        img = self._mmaps["main_img"][index]
        seg = self._mmaps["main_seg"][index]

        if self.transform is not None:
            out = self.transform(image=img, mask=seg)
            img, seg = out["image"], out["mask"]
        else:
            img = img.copy()

        main_img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        main_seg = torch.from_numpy(np.array(seg)).unsqueeze(0).float() / 255.0
        bev_seg = torch.from_numpy(
            bev_class_index(self._mmaps["bev_seg"][index])
        ).long()
        trajectory = torch.from_numpy(self._mmaps["trajectory"][index].copy()).float()
        trajectory = (trajectory - self._traj_min) * self._traj_scale - 1.0
        command = torch.tensor(int(self._mmaps["command"][index]), dtype=torch.long)

        temporal = torch.from_numpy(
            self._mmaps["temporal_trajectory"][index].copy()
        ).float()
        temporal = (temporal - self._temporal_min) * self._temporal_scale - 1.0

        return {
            "main_img": main_img,
            "main_seg": main_seg,
            "bev_seg": bev_seg,
            "trajectory": trajectory,
            "command": command,
            "temporal_trajectory": temporal,
        }
