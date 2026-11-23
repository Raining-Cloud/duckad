#!/usr/bin/env python3
"""Down-convert exported ONNX models to an opset the robot's onnxruntime accepts.

The Jetson wheel is onnxruntime 1.11, whose official ai.onnx support stops at opset 15; it
refuses a newer model outright rather than trying. PyTorch 2.12 defaults its exporter to opset
17, so any export that does not pin opset_version lands unloadable on the robot.

This rewrites such models in place via onnx.version_converter. For these graphs it is a
formality -- they use no op introduced or respecified after opset 15 -- but going through the
official converter means the adapters run rather than us just restamping the version field.

Two things are preserved explicitly because losing either breaks the robot:

  metadata_props   model_node reads trajectory_norm and temporal_dt from here and refuses to
                   start without them.
  ir_version       onnxruntime 1.11 reads IR version <= 8. A modern onnx package will happily
                   stamp a newer IR into the output, which produces a model that is the right
                   opset and still unloadable.

Nothing is written until the converted graph has been checked and its outputs compared against
the original, so a failed conversion leaves the original file untouched.

  scripts/convert_opset.py                     convert everything that needs it, in place
  scripts/convert_opset.py --dry-run           report what would change
  scripts/convert_opset.py --target-opset 13   aim lower
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO_ROOT, "packages", "model_node", "assets")
DEFAULT_TARGET_OPSET = 15
# onnxruntime 1.11 on the robot reads IR version 8 and no higher.
MAX_IR_VERSION = 8
RTOL, ATOL = 1e-5, 1e-6


def ai_onnx_opset(model):
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    return None


def random_inputs(session, seed=0):
    """Feed the model's declared inputs. 'command' is an index, so keep it in range."""
    rng = np.random.default_rng(seed)
    feeds = {}
    for spec in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in spec.shape]
        if "int64" in spec.type:
            feeds[spec.name] = np.full(shape, 2, dtype=np.int64)
        else:
            feeds[spec.name] = rng.random(shape, dtype=np.float32)
    return feeds


def outputs_match(before_path, after_path):
    """Same inputs through both graphs; report the worst absolute difference."""
    before = ort.InferenceSession(before_path, providers=["CPUExecutionProvider"])
    after = ort.InferenceSession(after_path, providers=["CPUExecutionProvider"])
    feeds = random_inputs(before)
    names = [o.name for o in before.get_outputs()]
    if names != [o.name for o in after.get_outputs()]:
        return False, "output names changed"
    a = before.run(None, feeds)
    b = after.run(None, feeds)
    worst, worst_name = 0.0, ""
    for name, x, y in zip(names, a, b):
        if x.shape != y.shape:
            return False, "{}: shape {} -> {}".format(name, x.shape, y.shape)
        diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        if diff > worst:
            worst, worst_name = diff, name
    ok = all(np.allclose(x, y, rtol=RTOL, atol=ATOL) for x, y in zip(a, b))
    return ok, "max |diff| {:.3e} on '{}'".format(worst, worst_name or names[0])


def convert(path, target, dry_run):
    model = onnx.load(path)
    current = ai_onnx_opset(model)
    name = os.path.basename(path)
    if current is None:
        print("  {:<18} no ai.onnx opset import - skipped".format(name))
        return None
    if current <= target:
        print("  {:<18} opset {} <= {}, ir {} - nothing to do".format(
            name, current, target, model.ir_version))
        return None
    print("  {:<18} opset {} -> {} (ir {})".format(name, current, target, model.ir_version))
    if dry_run:
        return "would convert"

    original_ir = model.ir_version
    metadata = {entry.key: entry.value for entry in model.metadata_props}

    converted = version_converter.convert_version(model, target)

    # The converter is free to restamp these; put them back deliberately.
    converted.ir_version = min(original_ir, MAX_IR_VERSION)
    if metadata:
        surviving = {entry.key for entry in converted.metadata_props}
        for key, value in metadata.items():
            if key not in surviving:
                entry = converted.metadata_props.add()
                entry.key, entry.value = key, value
        print("    metadata_props: {} carried over ({})".format(
            len(metadata), ", ".join(sorted(metadata))))

    onnx.checker.check_model(converted)

    # Stage beside the original so a failure here cannot leave a half-written weight file.
    handle, staged = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".onnx.tmp")
    os.close(handle)
    try:
        onnx.save(converted, staged)
        ok, detail = outputs_match(path, staged)
        print("    numerics: {} - {}".format("MATCH" if ok else "MISMATCH", detail))
        if not ok:
            raise RuntimeError("{}: outputs changed, refusing to overwrite".format(name))
        shutil.move(staged, path)
    finally:
        if os.path.exists(staged):
            os.remove(staged)
    got = ai_onnx_opset(onnx.load(path))
    print("    written: opset {}, ir {}".format(got, min(original_ir, MAX_IR_VERSION)))
    return "converted"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", help="specific .onnx files (default: all in assets/)")
    parser.add_argument("--target-opset", type=int, default=DEFAULT_TARGET_OPSET)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = args.models or sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.onnx")))
    if not paths:
        sys.exit("no .onnx files found")
    print("target opset {} (onnx {}, onnxruntime {} used for verification)".format(
        args.target_opset, onnx.__version__, ort.__version__))

    done = 0
    for path in paths:
        if convert(path, args.target_opset, args.dry_run):
            done += 1
    print("{} model(s) {}".format(done, "would be converted" if args.dry_run else "converted"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
