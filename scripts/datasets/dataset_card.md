---
license: cc-by-4.0
pretty_name: DuckAD Driving Dataset
task_categories:
- robotics
- image-segmentation
tags:
- autonomous-driving
- imitation-learning
- end-to-end-driving
- carla
- duckietown
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: image
    dtype: image
  - name: seg
    dtype: image
  - name: bev
    dtype: image
  - name: command
    dtype:
      class_label:
        names:
          '0': DEFAULT
          '1': LEFT
          '2': STRAIGHT
          '3': RIGHT
  - name: speed
    dtype: float32
  - name: trajectory
    list:
      list: float32
      length: 2
    length: 10
  - name: temporal_trajectory
    list:
      list: float32
      length: 2
    length: 10
  - name: scenario
    dtype: string
  - name: map
    dtype: string
  - name: episode
    dtype: string
  - name: frame
    dtype: int64
  splits:
  - name: train
    num_bytes: 11667691286
    num_examples: 214200
  - name: test
    num_bytes: 659226662
    num_examples: 10200
  download_size: 12310572997
  dataset_size: 12326917948
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---

# DuckAD Driving Dataset

Expert driving demonstrations for **DuckAD**, an end-to-end vision-based driving model,
collected in [CARLA](https://carla.org) on custom Duckietown-style maps. A rule-aware expert
driver was rolled out under six traffic/obstacle scenarios; every frame pairs a front camera
image with the expert's future trajectory, a high-level navigation command, and ground-truth
bird's-eye-view (BEV) semantics.

- **214,200 training frames** from two maps (`duckietown_04`, `duckietown_05`), six scenarios
  × 35,700 frames each.
- **10,200 test frames** from an *unseen* map (`duckietown_06`) rendered under *unseen* HDRI
  lighting, for cross-map generalization evaluation.
- The simulator runs synchronously at 20 Hz and each timestep is captured from **three camera
  rigs** (center, left, right — side rigs give recovery-style viewpoint diversity with
  correspondingly transformed trajectory labels), so the training set corresponds to roughly
  **one hour of expert driving**.

## Scenarios

| `scenario` | Traffic | Duckies |
|---|---|---|
| `baseline` | – | – |
| `traffic_only` | ✓ | – |
| `ducks_roadside` | – | roadside |
| `ducks_roadside_traffic` | ✓ | roadside |
| `ducks_obstacle` | – | on-road obstacles |
| `ducks_obstacle_traffic` | ✓ | on-road obstacles |

## Fields

| Field | Type | Description |
|---|---|---|
| `image` | 224×224 RGB image | front (fisheye) camera frame |
| `seg` | 224×224 grayscale image | binary foreground mask: 255 = Duckietown foreground (road surface, lane markings, signs, bots, duckies), 0 = replaceable background. Used for background-swap augmentation. |
| `bev` | 64×64 grayscale image | ground-truth BEV semantics as raw CARLA semantic tag ids (see below) |
| `command` | class label | navigation command: `DEFAULT` (lane follow), `LEFT`, `STRAIGHT`, `RIGHT` (junction maneuvers) |
| `speed` | float32 | ego speed in m/s |
| `trajectory` | 10×2 float32 | future **spatial** waypoints at 1 m arc-length spacing, meters in the ego frame (x forward, y left) |
| `temporal_trajectory` | 10×2 float32 | future ego positions sampled every **0.3 s** (3.33 Hz), same ego frame — encodes the speed profile |
| `scenario` / `map` / `episode` / `frame` | strings / int | provenance metadata |

**BEV tag ids:** 29 `center_lane`, 30 `side_lane`, 31 `asphalt`, 32 `stop_lane`, 33 `sign`,
34 `bot`, 35 `duck`; any other id (e.g. 11 = terrain) is background. For training we remap
these to 8 contiguous classes (`background`=0 + the 7 above).

## Usage

```python
from datasets import load_dataset

ds = load_dataset("pamasan/duckad-data", split="train")
sample = ds[0]
sample["image"]                # PIL.Image, 224x224 RGB front camera
sample["seg"]                  # PIL.Image, 224x224 foreground mask (255 = Duckietown foreground)
sample["bev"]                  # PIL.Image, 64x64 BEV semantics (CARLA tag ids, see table above)
sample["command"]              # int class label: 0 DEFAULT, 1 LEFT, 2 STRAIGHT, 3 RIGHT
sample["speed"]                # ego speed, m/s
sample["trajectory"]           # 10 spatial waypoints [x, y], 1 m spacing, meters in ego frame
sample["temporal_trajectory"]  # 10 future ego positions [x, y] sampled every 0.3 s -> speed profile

import numpy as np
bev_ids = np.array(sample["bev"])                     # 64x64 tag ids
fg_mask = np.array(sample["seg"]) > 0                 # boolean foreground mask
```

## Collection

Data was collected with a rule-aware expert (lane following, junction turns, stopping for
duckies and traffic) driving in synchronous CARLA at 20 Hz (`fixed_delta_seconds=0.05`).
Each recording segment respawns the ego at a new random location; traffic vehicles are
driven by the same expert. Junction approaches on the training maps are labeled
with the expert's chosen `LEFT`/`STRAIGHT`/`RIGHT` command; everywhere else the command is
`DEFAULT`.

## Notes

- Frames are stored as captured — **no augmentation is baked in**.
- The `test` split is for cross-map/lighting generalization; it uses a map and HDRI
  environment that never appear in `train`.

## License

Released under **CC-BY 4.0**. All visible content (maps, duckiebot vehicles, props) is
custom-built for this dataset.
