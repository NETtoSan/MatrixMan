"""Small tracing and unsupported-operation reporting service."""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import torch

from ...config import set_trace as _set_trace
from ...config import trace_enabled as _trace_enabled
from ...config import trace_log
from ...tensor import MatrixManTensor


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"}


unsupported_counts: Counter[str] = Counter()
unsupported_examples: dict[str, list[dict]] = defaultdict(list)


def set_trace(enabled: bool = True) -> None:
    _set_trace(enabled)


def debug_enabled() -> bool:
    # MATRIXMAN_DEBUG remains a legacy low-level OpenGL debug switch; it does
    # not turn on the new high-level operation trace.
    return _env_flag("MATRIXMAN_DEBUG") or _trace_enabled()


def trace(message: str) -> None:
    trace_log(message)


def kernel_log(message: str) -> None:
    trace_log(f"[MatrixMan/OpenGL] {message}")


def error_log(message: str) -> None:
    print(f"[MatrixMan ERROR] {message}")


def shape_text(shape) -> str:
    return "[" + ",".join(str(int(v)) for v in shape) + "]"


def summarize_dispatch_value(value):
    if isinstance(value, MatrixManTensor):
        return {
            "kind": "MatrixManTensor", "shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "texture": value._owner.texture,
            "storage": value._owner.layout.kind, "storage_offset": value._storage_offset,
            "logical_strides": list(value._logical_strides),
        }
    if isinstance(value, torch.Tensor):
        return {"kind": "torch.Tensor", "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (list, tuple)):
        return [summarize_dispatch_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): summarize_dispatch_value(v) for k, v in value.items()}
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    return repr(value)


def record_unsupported(func, args, kwargs) -> None:
    name = str(func)
    unsupported_counts[name] += 1
    if len(unsupported_examples[name]) < 3:
        unsupported_examples[name].append({"args": summarize_dispatch_value(args), "kwargs": summarize_dispatch_value(kwargs or {})})


def reset_unsupported_report() -> None:
    unsupported_counts.clear()
    unsupported_examples.clear()


def unsupported_report() -> dict[str, dict]:
    return {name: {"calls": count, "examples": unsupported_examples.get(name, [])} for name, count in sorted(unsupported_counts.items())}
