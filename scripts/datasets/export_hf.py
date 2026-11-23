"""Export the DuckAD npz shards as a Hugging Face dataset with a working dataset viewer.

Converts data/final (train) and data/test (test) into parquet shards with
image columns, so every frame is browsable on the Hub: main camera, segmentation mask,
BEV, command, speed and both trajectories.

Usage (needs `huggingface-cli login` first):
    uv run --with datasets --with pillow scripts/datasets/export_hf.py <user>/duckad-data
    uv run --with datasets --with pillow scripts/datasets/export_hf.py <user>/duckad-data --private
"""

import argparse
import glob
import os

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value

COMMANDS = ["DEFAULT", "LEFT", "STRAIGHT", "RIGHT"]  # src.aggregation.command.Command order

FEATURES = Features(
    {
        "image": Image(),
        "seg": Image(),  # main-camera semantic mask (class ids as grayscale)
        "bev": Image(),  # 64x64 ground-truth BEV segmentation (class ids as grayscale)
        "command": ClassLabel(names=COMMANDS),
        "speed": Value("float32"),
        "trajectory": Sequence(Sequence(Value("float32"), length=2), length=10),
        "temporal_trajectory": Sequence(Sequence(Value("float32"), length=2), length=10),
        "scenario": Value("string"),
        "map": Value("string"),
        "episode": Value("string"),
        "frame": Value("int64"),
    }
)


def iter_shards(shards: list[str], root: str):
    for shard in shards:
        episode = os.path.basename(os.path.dirname(shard))
        rel = os.path.relpath(os.path.dirname(shard), root)
        scenario = rel.split(os.sep)[0] if os.sep in rel else os.path.basename(root)
        with np.load(shard) as d:
            n = d["frame"].shape[0]
            for i in range(n):
                yield {
                    "image": d["main_img"][i][..., ::-1],  # stored BGR -> RGB for display
                    "seg": d["main_seg"][i],
                    "bev": d["bev_seg"][i, 0],
                    "command": int(d["command"][i]),
                    "speed": float(d["speed"][i]),
                    "trajectory": d["trajectory"][i].tolist(),
                    "temporal_trajectory": d["temporal_trajectory"][i].tolist(),
                    "scenario": scenario,
                    "map": "_".join(episode.split("_")[:2]),
                    "episode": episode,
                    "frame": int(d["frame"][i]),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hub repo id, e.g. <user>/duckad-data")
    parser.add_argument("--train-dir", default="data/final")
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-proc", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None,
                        help="only export this many frames per split (smoke-testing the upload)")
    args = parser.parse_args()

    def build(root: str) -> Dataset:
        shards = sorted(glob.glob(os.path.join(root, "**", "batch_*.npz"), recursive=True))
        if args.limit is not None:
            shards = shards[: max(1, -(-args.limit // 255))]  # 255 frames per shard
        ds = Dataset.from_generator(
            iter_shards,
            gen_kwargs={"shards": shards, "root": root},  # the shard list is split across workers
            features=FEATURES,
            num_proc=min(args.num_proc, len(shards)),
        )
        return ds.select(range(args.limit)) if args.limit is not None else ds

    splits = DatasetDict({"train": build(args.train_dir), "test": build(args.test_dir)})
    for name, ds in splits.items():
        print(f"{name}: {len(ds)} frames")
    splits.push_to_hub(args.repo, private=args.private)


if __name__ == "__main__":
    main()
