#!/usr/bin/env python3
import json
import time

import cv2
import numpy as np
import onnxruntime as ort

# Opset: the Jetson wheel is onnxruntime 1.11, whose official ai.onnx support stops at 15, and it
# refuses anything newer outright rather than trying:
#   FAIL : Load model from .../noaug.onnx failed: ... Opset 17 is under development and support
#   for this is limited. ... Current official support for domain ai.onnx is till opset 15.
# PyTorch 2.12 defaults its exporter to 17, so an export that does not pin opset_version lands
# here unloadable. All shipped models are opset 15; if that error reappears after a re-export,
# run scripts/convert_opset.py rather than setting ORT_ALLOW_RELEASED_ONNX_OPSET_ONLY=0 -- the
# error is worth keeping, because a future model might genuinely need a newer op rather than
# just being stamped for one.

IMG_SIZE = 224
DEFAULT_TEMPORAL_DT = 0.25
COMMAND_NAMES = ["default", "left", "straight", "right"]
CUDA_PROVIDER_OPTIONS = {"cudnn_conv_algo_search": "HEURISTIC"}
INTRA_OP_THREADS = 2


def letterbox(image, size=IMG_SIZE):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - new_h) // 2, (size - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def denorm(points, c, keys):
    lo = np.array([c[keys[0]], c[keys[1]]])
    span = np.array([c[keys[2]] - c[keys[0]], c[keys[3]] - c[keys[1]]])
    return (points + 1.0) * 0.5 * span + lo


def softmax(logits):
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class OnnxModel(object):
    def __init__(self, onnx_path, providers=None):
        available = ort.get_available_providers()
        print("[OnnxModel] onnxruntime {}".format(ort.__version__))
        print("[OnnxModel] available providers: {}".format(available))

        if providers is None:
            preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in available]
        providers = [(p, CUDA_PROVIDER_OPTIONS) if p == "CUDAExecutionProvider" else p
                     for p in providers]
        print("[OnnxModel] requesting providers: {}".format(providers))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = INTRA_OP_THREADS
        started = time.perf_counter()
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        print("[OnnxModel] session created in {:.1f}s".format(time.perf_counter() - started))
        active = self.session.get_providers()
        print("[OnnxModel] active providers: {}".format(active))
        if active and active[0] == "CPUExecutionProvider":
            print("[OnnxModel] WARNING: running on CPU - GPU execution provider not in use")

        meta = self.session.get_modelmeta().custom_metadata_map
        if "trajectory_norm" not in meta:
            raise SystemExit("ONNX has no embedded norm stats; re-export with eval.export_onnx")
        self.norm = json.loads(meta["trajectory_norm"])
        self.dt = float(meta.get("temporal_dt", DEFAULT_TEMPORAL_DT))

    def warmup(self, runs=2):
        # The CUDA EP defers kernel JIT, cuDNN algo selection and workspace allocation to the
        # first run with a given input shape. Pay that here, before the camera subscription.
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        for i in range(runs):
            started = time.perf_counter()
            self.predict(dummy, 2)
            print("[OnnxModel] warmup {}/{}: {:.0f} ms".format(
                i + 1, runs, (time.perf_counter() - started) * 1000.0))

    def predict(self, bgr, command):
        letterboxed = letterbox(bgr)
        image_input = letterboxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = dict(zip(
            (o.name for o in self.session.get_outputs()),
            self.session.run(None, {"image": image_input, "command": np.array([command], np.int64)})))
        spatial = denorm(out["spatial"][0], self.norm, ("x_min", "y_min", "x_max", "y_max"))
        temporal = denorm(out["temporal"][0], self.norm, ("tx_min", "ty_min", "tx_max", "ty_max"))
        bev_idx = out["bev"][0].argmax(0)
        command_logits = out["command_logits"][0].astype(np.float64)
        speed_kmh = float(np.linalg.norm(temporal[0]) / self.dt * 3.6)
        return letterboxed, spatial, temporal, bev_idx, command_logits, speed_kmh
