#!/usr/bin/env python3
"""
Ultralytics YOLO operator discovery for the experimental GM45 backend.

This script does not silently run unsupported GM45 tensor arithmetic on CPU.
For Phase 1 discovery it uses PyTorch FakeTensor/meta execution plus a
TorchDispatchMode recorder. That advances the YOLO graph far enough to produce
an ATen inventory without computing real tensor values on the CPU.

The report is the authority for which GM45/OpenGL kernels to implement next.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
from ultralytics import YOLO

from drivers import matrixman as gm45


ARITHMETIC_OPS = {
    "aten.convolution.default",
    "aten.native_batch_norm.default",
    "aten.silu.default",
    "aten.silu_.default",
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.div.Tensor",
    "aten.sigmoid.default",
    "aten.sigmoid_.default",
    "aten.max_pool2d_with_indices.default",
    "aten.upsample_nearest2d.default",
}

METADATA_OPS = {
    "prim.device.default",
    "aten.view.default",
    "aten.reshape.default",
    "aten.squeeze.dim",
    "aten.unsqueeze.default",
    "aten.permute.default",
    "aten.transpose.int",
    "aten.flatten.using_ints",
    "aten.expand.default",
    "aten.select.int",
}

LAYOUT_STORAGE_OPS = {
    "aten.empty.memory_format",
    "aten.cat.default",
    "aten.split.Tensor",
    "aten.chunk.default",
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.slice.Tensor",
    "aten.stack.default",
}

SETUP_DECODE_OPS = {
    "aten.arange.default",
    "aten.full.default",
    "aten._local_scalar_dense.default",
}


@dataclass
class OpRecord:
    calls: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)


def classify_op(name: str) -> str:
    if name in ARITHMETIC_OPS:
        return "A. GPU arithmetic"
    if name in METADATA_OPS:
        return "B. metadata/view"
    if name in LAYOUT_STORAGE_OPS:
        return "C. layout/storage"
    if name in SETUP_DECODE_OPS:
        return "D. setup/decode"
    return "unknown"


def tensor_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, (list, tuple)):
        return [tensor_summary(v) for v in value]
    if isinstance(value, dict):
        return {str(k): tensor_summary(v) for k, v in value.items()}
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.Size):
        return list(value)
    return repr(value)


class AtenRecorder(TorchDispatchMode):
    def __init__(self, max_examples: int = 3):
        self.records: dict[str, OpRecord] = defaultdict(OpRecord)
        self.max_examples = max_examples

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        record = self.records[name]
        record.calls += 1
        if len(record.examples) < self.max_examples:
            record.examples.append(
                {
                    "args": tensor_summary(args),
                    "kwargs": tensor_summary(kwargs or {}),
                    "classification": classify_op(name),
                }
            )
        return func(*args, **(kwargs or {}))


def run_fake_yolo_trace(model_spec: str, image_size: int, mode: str) -> tuple[AtenRecorder, str | None, dict[str, Any]]:
    model = YOLO(model_spec).model
    if mode == "train":
        model.train()
    else:
        model.eval()

    progress = {
        "total_layers": len(getattr(model, "model", [])),
        "entered": 0,
        "completed": 0,
        "last_entered": None,
        "last_completed": None,
    }
    hooks = []
    for index, module in enumerate(getattr(model, "model", [])):
        def pre_hook(_module, _args, index=index):
            progress["entered"] = max(progress["entered"], index + 1)
            progress["last_entered"] = f"{index}:{_module.__class__.__name__}"

        def post_hook(_module, _args, _out, index=index):
            progress["completed"] = max(progress["completed"], index + 1)
            progress["last_completed"] = f"{index}:{_module.__class__.__name__}"

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    recorder = AtenRecorder()
    failure = None
    try:
        with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True):
            x = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
            with recorder:
                _ = model(x)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        for hook in hooks:
            hook.remove()
    return recorder, failure, progress


def run_real_gm45_first_blocker(model_spec: str, image_size: int) -> dict[str, Any]:
    gm45.reset_unsupported_report()
    try:
        model = YOLO(model_spec).model.eval()
        x_cpu = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)
        x = gm45.to_device(x_cpu)
        with torch.no_grad():
            _ = model(x)
        return {"status": "unexpectedly completed YOLO forward", "unsupported": gm45.unsupported_report()}
    except Exception as exc:
        return {
            "status": "blocked during real GM45 YOLO forward",
            "reason": f"{type(exc).__name__}: {exc}",
            "unsupported": gm45.unsupported_report(),
        }


def print_report(title: str, recorder: AtenRecorder, failure: str | None, progress: dict[str, Any], limit: int) -> None:
    total = sum(r.calls for r in recorder.records.values())
    print(f"\n{title}")
    print(f"  unique ATen ops: {len(recorder.records)}")
    print(f"  total ATen calls: {total}")
    print(
        "  model layers reached: "
        f"{progress['entered']} entered / {progress['completed']} completed / {progress['total_layers']} total"
    )
    print(f"  last entered layer: {progress['last_entered']}")
    print(f"  last completed layer: {progress['last_completed']}")
    print(f"  trace status: {'complete' if failure is None else 'stopped: ' + failure}")

    by_bucket: dict[str, list[tuple[str, OpRecord]]] = defaultdict(list)
    for name, record in recorder.records.items():
        by_bucket[classify_op(name)].append((name, record))

    for bucket in [
        "A. GPU arithmetic",
        "B. metadata/view",
        "C. layout/storage",
        "D. setup/decode",
        "unknown",
    ]:
        items = sorted(by_bucket.get(bucket, []), key=lambda item: (-item[1].calls, item[0]))
        if not items:
            continue
        print(f"\n{bucket}")
        for name, record in items[:limit]:
            print(f"{name}")
            print(f"    calls: {record.calls}")
            if record.examples:
                example = record.examples[0]
                print(f"    example args: {example['args']}")
                if example["kwargs"]:
                    print(f"    example kwargs: {example['kwargs']}")


def print_progress_summary(recorder: AtenRecorder, failure: str | None, progress: dict[str, Any]) -> None:
    supported_now = {
        "aten.add.Tensor",
        "aten.max_pool2d_with_indices.default",
        "aten.upsample_nearest2d.default",
        "aten.arange.default",
        "aten.mm.default",
        "aten.convolution.default",
        "aten.native_batch_norm.default",
        "aten.silu_.default",
        "aten.split.Tensor",
        "aten.cat.default",
        "aten.stack.default",
        "aten.fill_.Scalar",
        "aten._to_copy.default",
        "aten.detach.default",
        "aten.view.default",
        "aten.reshape.default",
        "aten.flatten.using_ints",
        "aten.squeeze.default",
        "aten.squeeze.dim",
        "aten.unsqueeze.default",
        "aten.empty.memory_format",
    }
    supported_calls = sum(record.calls for name, record in recorder.records.items() if name in supported_now)
    unsupported = [
        (name, record.calls, classify_op(name))
        for name, record in recorder.records.items()
        if name not in supported_now
    ]
    unsupported.sort(key=lambda item: (-item[1], item[0]))
    next_unsupported = unsupported[0] if unsupported else None
    arithmetic = Counter(classify_op(name) for name in recorder.records)

    print("\nGM45 YOLO compatibility:")
    print("  model: yolov8n.yaml")
    print(
        "  model layers reached: "
        f"{progress['entered']} entered / {progress['completed']} completed / {progress['total_layers']} total"
    )
    print("  supported ATen calls with current GM45 backend:", supported_calls)
    print("  unique unsupported/discovered ATen ops:", len(unsupported))
    print("  arithmetic ops discovered:", arithmetic.get("A. GPU arithmetic", 0))
    if next_unsupported:
        print(f"  highest-frequency unsupported op: {next_unsupported[0]} ({next_unsupported[1]} calls, {next_unsupported[2]})")
    print(f"  full eval-mode blocker: {failure or 'none'}")
    print("  current real-GM45 progress: NCHW input upload works; forward stops at first unsupported GM45 op")


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover Ultralytics YOLO ATen ops for GM45 backend work")
    parser.add_argument("--model", default="yolov8n.yaml", help="local Ultralytics model YAML/PT, default: yolov8n.yaml")
    parser.add_argument("--imgsz", type=int, default=64, help="tiny square dummy input size, default: 64")
    parser.add_argument("--limit", type=int, default=20, help="max ops per classification bucket to print")
    args = parser.parse_args()

    print("Ultralytics YOLO GM45 operation discovery")
    print("  model spec:", args.model)
    print("  dummy input:", [1, 3, args.imgsz, args.imgsz], "float32")
    print("  discovery execution: PyTorch FakeTensor/meta; no CPU tensor arithmetic fallback is claimed as GM45")

    real_blocker = run_real_gm45_first_blocker(args.model, args.imgsz)
    print("\nReal MatrixManTensor attempt:")
    print("  status:", real_blocker["status"])
    if "reason" in real_blocker:
        print("  reason:", real_blocker["reason"])

    train_recorder, train_failure, train_progress = run_fake_yolo_trace(args.model, args.imgsz, "train")
    eval_recorder, eval_failure, eval_progress = run_fake_yolo_trace(args.model, args.imgsz, "eval")

    print_report("Raw/head-training-mode trace inventory", train_recorder, train_failure, train_progress, args.limit)
    print_report("Eval/inference-mode trace inventory", eval_recorder, eval_failure, eval_progress, args.limit)
    print_progress_summary(eval_recorder if eval_recorder.records else train_recorder, eval_failure, eval_progress)
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
