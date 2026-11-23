from __future__ import annotations

import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np

from ..config import StorageConfig


class SegmentBuffer:
    def __init__(
        self,
        cfg: StorageConfig,
        temporal_num_points: int,
        temporal_stride_ticks: int,
        temporal_dt: float,
        file_suffix: str,
    ):
        self._out_dir = Path(cfg.out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.temporal_num_points = temporal_num_points
        self.temporal_stride_ticks = temporal_stride_ticks
        self.temporal_dt = temporal_dt
        self._file_suffix = file_suffix
        self._horizon = temporal_num_points * temporal_stride_ticks
        rigs_per_tick = 3
        self._shard = max(
            rigs_per_tick, (cfg.batch_size // rigs_per_tick) * rigs_per_tick
        )
        self._file_index = 0
        self._ego: list[tuple[float, float, float]] = []
        self._pending: deque[dict] = deque()
        self._ready: list[dict] = []
        self._dropped = 0

        self._write_queue: queue.Queue = queue.Queue(maxsize=4)
        self._write_error: BaseException | None = None
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

    def add_ego(self, x: float, y: float, z: float) -> None:
        self._ego.append((x, y, z))
        finalizable_tick = len(self._ego) - 1 - self._horizon
        while self._pending and self._pending[0]["tick"] <= finalizable_tick:
            row = self._pending.popleft()
            row["temporal_trajectory"] = self._temporal_for_row(row)
            self._ready.append(row)
        while len(self._ready) >= self._shard:
            self._write(self._ready[: self._shard])
            del self._ready[: self._shard]

    def add(
        self,
        *,
        tick: int,
        frame: int,
        main_img: np.ndarray,
        main_seg: np.ndarray,
        bev_seg: np.ndarray,
        trajectory: np.ndarray,
        speed: float,
        command: int,
        ref_inv_matrix: np.ndarray,
    ) -> None:
        self._pending.append(
            {
                "tick": tick,
                "frame": frame,
                "main_img": np.ascontiguousarray(main_img),
                "main_seg": np.ascontiguousarray(main_seg),
                "bev_seg": np.ascontiguousarray(bev_seg),
                "trajectory": np.ascontiguousarray(trajectory),
                "speed": speed,
                "command": command,
                "ref_inv_matrix": np.asarray(
                    ref_inv_matrix, dtype=np.float64
                ),  # 4x4 world->local
            }
        )

    def _temporal_for_row(self, row: dict) -> np.ndarray:
        base = row["tick"]
        idx = [
            base + k * self.temporal_stride_ticks
            for k in range(1, self.temporal_num_points + 1)
        ]
        future = np.array([self._ego[i] for i in idx], dtype=np.float64)
        future_h = np.concatenate([future, np.ones((len(future), 1))], axis=1)
        local = future_h @ row["ref_inv_matrix"].T
        return local[:, :2].astype(np.float32)

    def end_segment(self) -> None:
        self._dropped += len(self._pending)
        self._pending.clear()
        self._ego.clear()

    def close(self) -> int:
        while len(self._ready) >= self._shard:
            self._write(self._ready[: self._shard])
            del self._ready[: self._shard]
        self._dropped += len(self._ready) + len(self._pending)
        self._ready.clear()
        self._pending.clear()
        self._write_queue.put(None)
        self._writer.join()
        if self._write_error is not None:
            raise self._write_error
        return self._dropped

    def _write(self, rows: list[dict]) -> None:
        self._write_queue.put((self._file_index, list(rows)))
        self._file_index += 1

    def _writer_loop(self) -> None:
        while (job := self._write_queue.get()) is not None:
            index, rows = job
            try:
                self._save(index, rows)
            except BaseException as exc:
                self._write_error = exc

    def _save(self, index: int, rows: list[dict]) -> None:
        path = self._out_dir / f"batch_{index:06d}_{self._file_suffix}.npz"
        np.savez_compressed(
            path,
            frame=np.array([r["frame"] for r in rows]),
            main_img=np.stack([r["main_img"] for r in rows]),
            main_seg=np.stack([r["main_seg"] for r in rows]),
            bev_seg=np.stack([r["bev_seg"] for r in rows]),
            trajectory=np.stack([r["trajectory"] for r in rows]),
            temporal_trajectory=np.stack([r["temporal_trajectory"] for r in rows]),
            speed=np.array([r["speed"] for r in rows], dtype=np.float32),
            command=np.array([r["command"] for r in rows], dtype=np.int64),
            temporal_dt=np.float32(self.temporal_dt),
        )
