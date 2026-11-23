"""Download the DuckAD dataset from the Hugging Face Hub and rebuild the npz shards.

Inverse of export_hf.py: turns the parquet splits back into the batch_*.npz layout the
training code reads, so the released dataset can be trained on directly.

    train -> data/final/<scenario>/<episode>/batch_*.npz
    test  -> data/test/<scenario>/<episode>/batch_*.npz

trajectory_norm.json is not part of the Hub dataset; fetch_data.sh regenerates it, or run
    python -m src.aggregation.data.normalize data/final --clip-percentile 1

Usage:
    uv run --with huggingface_hub --with pyarrow --with pillow scripts/datasets/import_hf.py
    uv run --with huggingface_hub --with pyarrow --with pillow scripts/datasets/import_hf.py --repo <user>/<dataset>
"""

import argparse
import io
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

# Not stored on the Hub: synchronous CARLA at fixed_delta_seconds=0.05 with temporal_hz=3.33
# gives a 6-tick stride -> 0.3 s between temporal points (see TrajectoryConfig.temporal_dt).
TEMPORAL_DT = np.float32(0.3)
SHARD_SIZE = 255  # rows per shard, as written by the recorder (multiple of the 3 rigs per tick)


def decode(cell: dict) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(cell["bytes"])))


def save_shard(out_dir: Path, name: str, rows: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / name,
        frame=np.array([r["frame"] for r in rows], dtype=np.int64),
        main_img=np.stack([decode(r["image"])[..., ::-1] for r in rows]),  # RGB -> stored BGR
        main_seg=np.stack([decode(r["seg"]) for r in rows]),
        bev_seg=np.stack([decode(r["bev"])[None] for r in rows]),  # (1, H, W) as recorded
        trajectory=np.array([r["trajectory"] for r in rows], dtype=np.float32),
        temporal_trajectory=np.array([r["temporal_trajectory"] for r in rows], dtype=np.float32),
        speed=np.array([r["speed"] for r in rows], dtype=np.float32),
        command=np.array([r["command"] for r in rows], dtype=np.int64),
        temporal_dt=TEMPORAL_DT,
    )


def convert_file(parquet_path: str, out_root: str, file_index: int) -> int:
    """Convert one parquet file. Shard names carry the file index, so files convert in parallel."""
    out_root = Path(out_root)
    pending: list[dict] = []
    key = None
    shard_index = 0

    def flush():
        nonlocal pending, shard_index
        if pending:
            scenario, episode = key
            save_shard(out_root / scenario / episode,
                       f"batch_{file_index:03d}{shard_index:05d}_hf.npz", pending)
            shard_index += 1
            pending = []

    rows_total = 0
    pf = pq.ParquetFile(parquet_path)
    for g in range(pf.num_row_groups):
        for row in pf.read_row_group(g).to_pylist():
            row_key = (row["scenario"], row["episode"])
            if row_key != key or len(pending) >= SHARD_SIZE:
                flush()
                key = row_key
            pending.append(row)
            rows_total += 1
    flush()
    return rows_total


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="pamasan/duckad-data")
    parser.add_argument("--train-dir", default="data/final")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--cache-dir", default=None, help="where the parquet files are downloaded")
    parser.add_argument("--num-proc", type=int, default=os.cpu_count())
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    local = snapshot_download(args.repo, repo_type="dataset", allow_patterns="data/*.parquet",
                              cache_dir=args.cache_dir)
    files = sorted(Path(local, "data").glob("*.parquet"))
    jobs = [(str(f), args.train_dir if f.name.startswith("train-") else args.test_dir, i)
            for i, f in enumerate(files)]

    print(f">>> converting {len(files)} parquet files with {args.num_proc} processes ...")
    totals = {args.train_dir: 0, args.test_dir: 0}
    with ProcessPoolExecutor(args.num_proc) as pool:
        for (path, out, _), n in zip(jobs, pool.map(convert_file, *zip(*jobs))):
            totals[out] += n
            print(f"    {Path(path).name}: {n} frames -> {out}", flush=True)

    for out, n in totals.items():
        print(f">>> {out}: {n} frames")


if __name__ == "__main__":
    main()
