#!/usr/bin/env python3
"""Compare one deterministic Ultralytics YOLO forward pass across backends.

Run this module once per backend.  MatrixMan selection is process-global, so
the comparison deliberately uses two processes and compares their saved CPU
readbacks afterward::

    MATRIXMAN_BACKEND=opengl python3 -m drivers.matrixman.diagnostics.matrixman_yolo_backend_compare \
        --backend opengl --dump /tmp/yolo-opengl.pt --save-input /tmp/yolo-input.pt
    MATRIXMAN_BACKEND=cuda python3 -m drivers.matrixman.diagnostics.matrixman_yolo_backend_compare \
        --backend cuda --dump /tmp/yolo-cuda.pt --input-tensor /tmp/yolo-input.pt
    python3 -m drivers.matrixman.diagnostics.matrixman_yolo_backend_compare \
        --compare /tmp/yolo-opengl.pt /tmp/yolo-cuda.pt

This is intentionally a diagnostic rather than part of the public demo.
MatrixMan tensors are read back only through the explicit diagnostic helper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "demo/models/VisDrone-arm64-480/weights/best.pt"
DEFAULT_VIDEO = ROOT / "demo/videos/video0.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one fixed YOLO input across MatrixMan backends")
    parser.add_argument("--backend", choices=("cuda", "opengl"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--input-tensor", type=Path)
    parser.add_argument("--save-input", type=Path)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--compare", nargs=2, metavar=("OPENGL_DUMP", "CUDA_DUMP"))
    return parser.parse_args()


def _summary(name: str, tensor: torch.Tensor, values: torch.Tensor) -> dict[str, Any]:
    array = values.detach().numpy()
    flat = array.reshape(-1)
    strides = list(getattr(tensor, "_logical_strides", tensor.stride()))
    offset = int(getattr(tensor, "_storage_offset", tensor.storage_offset()))
    return {
        "name": name,
        "shape": list(tensor.shape),
        "strides": [int(value) for value in strides],
        "storage_offset": offset,
        "dtype": str(tensor.dtype),
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "mean": float(array.mean()) if array.size else 0.0,
        "abs_max": float(np.abs(array).max()) if array.size else 0.0,
        "all_finite": bool(np.isfinite(array).all()),
        "first_values": flat[:8].tolist(),
        "values": values.detach().clone(),
    }


def _capture(name: str, value, records: list[dict[str, Any]], readback_tensor, is_matrixman_tensor) -> None:
    if not isinstance(value, torch.Tensor):
        return
    if value.ndim == 0 and value.numel() == 0:
        return
    cpu = readback_tensor(value) if is_matrixman_tensor(value) else value.detach()
    records.append(_summary(name, value, cpu.contiguous()))


def _install_hooks(model, records: list[dict[str, Any]], readback_tensor, is_matrixman_tensor):
    """Capture model boundaries and the Detect head's internal branch outputs."""
    handles = []

    def add_post(module, name):
        def hook(_module, _inputs, output):
            value = _first_tensor(output)
            if value is not None:
                _capture(name, value, records, readback_tensor, is_matrixman_tensor)

        handles.append(module.register_forward_hook(hook))

    def add_pre(module, name):
        def hook(_module, inputs):
            value = _first_tensor(inputs)
            if value is not None:
                _capture(name, value, records, readback_tensor, is_matrixman_tensor)

        handles.append(module.register_forward_pre_hook(hook))

    modules = getattr(model, "model", None)
    if modules is None:
        raise RuntimeError("loaded Ultralytics model has no model module list")
    for index, module in enumerate(modules):
        add_pre(module, f"model.{index} {type(module).__name__} input")
        add_post(module, f"model.{index} {type(module).__name__} output")

    # Narrow follow-up instrumentation for the first known divergence.  Keep
    # these hooks explicit so the report reflects the actual DWConv order and
    # does not require tracing every ATen operation in the model.
    if len(modules) > 1:
        dwconv = modules[1]
        for child_name in ("conv", "bn", "act"):
            child = getattr(dwconv, child_name, None)
            if child is not None and isinstance(child, torch.nn.Module):
                add_post(child, f"model.1.{child_name} {type(child).__name__} output")

    # The Detect module output includes the final decoded tensor, while these
    # nested hooks expose its regression/classification branch boundaries.
    detect_index = len(modules) - 1
    detect = modules[detect_index]
    for name, module in detect.named_modules():
        if not name or any(module.children()):
            continue
        add_post(module, f"model.{detect_index}.{name} {type(module).__name__} output")

    return handles


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _load_input(args: argparse.Namespace) -> torch.Tensor:
    if args.input_tensor:
        return torch.load(args.input_tensor, map_location="cpu", weights_only=False).detach().contiguous()

    import cv2

    if args.frame_index < 0:
        raise ValueError("--frame-index must be non-negative")
    cap = cv2.VideoCapture(str(args.video.expanduser().resolve()))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {args.video}")
    try:
        for _ in range(args.frame_index):
            ok, _ = cap.read()
            if not ok:
                raise RuntimeError(f"video ended before frame {args.frame_index}")
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {args.frame_index}")

    from demo.yolo_helpers import preprocess_frame

    return preprocess_frame(frame, args.imgsz).contiguous()


def run_backend(args: argparse.Namespace) -> int:
    if not args.backend:
        raise ValueError("--backend is required unless --compare is used")
    if not args.dump:
        raise ValueError("--dump is required for a backend trace")
    os.environ["MATRIXMAN_BACKEND"] = args.backend

    from ultralytics import YOLO

    from drivers import matrixman
    from drivers.matrixman.backend import get_backend
    from drivers.matrixman.diagnostics.backend_helpers import readback_tensor

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    cpu_input = _load_input(args)
    if args.save_input:
        args.save_input.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cpu_input, args.save_input)

    matrixman.init()
    handles = []
    try:
        backend = get_backend()
        print(f"selected backend: {backend.name}")
        print(f"device info: {backend.device_info()}")
        model = YOLO(str(model_path)).model.eval()
        records: list[dict[str, Any]] = []
        gpu_input = matrixman.to_device(cpu_input)
        _capture("model input after MatrixMan upload", gpu_input, records, readback_tensor, matrixman.is_matrixman_tensor)
        handles = _install_hooks(model, records, readback_tensor, matrixman.is_matrixman_tensor)
        with torch.no_grad():
            output = _first_tensor(model(gpu_input))
        if output is None:
            raise RuntimeError("YOLO forward returned no tensor")
        _capture("final prediction", output, records, readback_tensor, matrixman.is_matrixman_tensor)
        metadata = {key: value for key, value in _summary("unused", output, readback_tensor(output)).items() if key != "values"}
        torch.save({"backend": backend.name, "records": records, "prediction": metadata}, args.dump)
        print(f"captured {len(records)} tensors")
        print(f"trace written: {args.dump}")
        return 0
    finally:
        for handle in handles:
            handle.remove()
        matrixman.shutdown()


def compare_dumps(paths: list[str]) -> int:
    left = torch.load(paths[0], map_location="cpu", weights_only=False)
    right = torch.load(paths[1], map_location="cpu", weights_only=False)
    left_records = {record["name"]: record for record in left["records"]}
    right_records = {record["name"]: record for record in right["records"]}
    tolerance = 1e-4
    print(f"compare {left['backend']} vs {right['backend']} (atol={tolerance}, rtol={tolerance})")
    last_good = None
    first_bad = None
    for name, lhs in left_records.items():
        rhs = right_records.get(name)
        if rhs is None:
            print(f"{name}: missing from {right['backend']}")
            if first_bad is None:
                first_bad = name
            continue
        lhs_value, rhs_value = lhs["values"], rhs["values"]
        if tuple(lhs_value.shape) != tuple(rhs_value.shape):
            print(f"{name}: shape {list(lhs_value.shape)} vs {list(rhs_value.shape)} FAIL")
            if first_bad is None:
                first_bad = name
            continue
        difference = (lhs_value - rhs_value).abs()
        max_diff = float(difference.max()) if difference.numel() else 0.0
        mean_diff = float(difference.mean()) if difference.numel() else 0.0
        relative = max_diff / max(float(lhs["abs_max"]), float(rhs["abs_max"]), 1e-12)
        good = torch.allclose(lhs_value, rhs_value, atol=tolerance, rtol=tolerance)
        print(
            f"{name}: shape={list(lhs_value.shape)} max_abs_diff={max_diff:.6g} "
            f"mean_abs_diff={mean_diff:.6g} relative_max={relative:.6g} "
            f"{left['backend']}[min={lhs['min']:.6g}, max={lhs['max']:.6g}, "
            f"mean={lhs['mean']:.6g}, finite={lhs['all_finite']}, "
            f"first={lhs['first_values']}] "
            f"{right['backend']}[min={rhs['min']:.6g}, max={rhs['max']:.6g}, "
            f"mean={rhs['mean']:.6g}, finite={rhs['all_finite']}, "
            f"first={rhs['first_values']}] "
            f"{'PASS' if good else 'FAIL'}"
        )
        if first_bad is None and good:
            last_good = name
        elif first_bad is None:
            first_bad = name

    print(f"LAST GOOD MODULE: {last_good}")
    print(f"FIRST BAD MODULE: {first_bad}")
    return 1 if first_bad is not None else 0


def main() -> int:
    args = parse_args()
    if args.compare:
        return compare_dumps(args.compare)
    return run_backend(args)


if __name__ == "__main__":
    raise SystemExit(main())
