"""Opt-in audit of MatrixMan tensor paths that materialize on the CPU."""

from __future__ import annotations

import atexit
import traceback
from contextlib import contextmanager
from collections import Counter

from .config import config

_counts = Counter()
_elements = Counter()
_bytes = Counter()
_registered = False
_summary_printed = False
_readbacks: list[dict] = []
_readback_context = None


def enabled() -> bool:
    return bool(config.auditCpuLeaks)


@contextmanager
def readback_context(category: str):
    """Label an explicit readback without changing its execution behavior."""
    global _readback_context
    previous = _readback_context
    _readback_context = category
    try:
        yield
    finally:
        _readback_context = previous


def record_readback(*, tensor, op: str | None, reason: str) -> None:
    if not enabled():
        return
    owner = getattr(tensor, "_owner", None)
    layout = getattr(owner, "layout", None)
    itemsize = 4
    _readbacks.append({
        "category": _readback_context or "explicit",
        "op": op,
        "reason": reason,
        "shape": [int(v) for v in tensor.shape],
        "bytes": int(tensor.numel()) * itemsize,
        "physical_bytes": int(getattr(layout, "texture_width", 0)) * int(getattr(layout, "texture_height", 0)) * 16,
        "frame": getattr(_active_frame, "value", None),
    })


_active_frame = __import__("threading").local()


@contextmanager
def frame(number: int):
    previous = getattr(_active_frame, "value", None)
    _active_frame.value = number
    try:
        yield
    finally:
        _active_frame.value = previous


def readback_report() -> dict:
    records = list(_readbacks)
    return {
        "count": len(records),
        "logical_bytes": sum(item["bytes"] for item in records),
        "physical_bytes": sum(item["physical_bytes"] for item in records),
        "by_category": {category: sum(1 for item in records if item["category"] == category)
                        for category in sorted({item["category"] for item in records})},
        "records": records,
    }


def reset_readbacks() -> None:
    """Discard prior readback records, normally after benchmark warmup."""
    _readbacks.clear()


def count(category: str) -> int:
    return int(_counts[category])


def record(category: str, *, op: str | None = None, tensor=None, shape=None,
           dtype=None, numel=None, storage=None, reason: str = "",
           cpu_arithmetic: bool = False) -> None:
    if not enabled():
        return
    if tensor is not None:
        shape = list(int(v) for v in tensor.shape)
        dtype = tensor.dtype
        numel = int(tensor.numel())
        owner = getattr(tensor, "_owner", None)
        storage = getattr(owner, "storage_description", storage)
    shape = [] if shape is None else list(shape)
    numel = int(numel or 0)
    _counts[category] += 1
    _elements[category] += numel
    if dtype is not None:
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize:
            _bytes[category] += numel * int(itemsize)
    warning = category == "unexpected_cpu_materialization" or (cpu_arithmetic and numel > 0)
    prefix = "[MatrixMan/Audit][WARNING]" if warning else "[MatrixMan/Audit]"
    print(f"{prefix} {category}")
    if not warning:
        return
    print(f"  op={op or 'unknown'}")
    print(f"  shape={shape}")
    print(f"  dtype={dtype or 'unknown'}")
    print(f"  numel={numel}")
    if storage:
        print(f"  storage={storage}")
    print(f"  reason={reason or 'unspecified'}")
    print(f"  cpu_arithmetic={'yes' if cpu_arithmetic else 'no'}")
    for line in traceback.format_stack(limit=6)[:-1]:
        print("  " + line.rstrip())
    if cpu_arithmetic and numel > 0:
        print("  warning_detail=nonzero CPU arithmetic observed on MatrixMan-originating data")


def summary() -> None:
    global _summary_printed
    if not enabled():
        return
    if _summary_printed:
        return
    _summary_printed = True
    print("MatrixMan CPU audit summary:")
    for category in ("allowed_bookkeeping", "explicit_readback", "explicit_cpu_transfer", "placeholder_metadata", "unexpected_cpu_materialization"):
        print(f"  {category}: {_counts[category]}")
    if _elements:
        print(f"  nonzero elements: {sum(_elements.values())}")
        print(f"  estimated bytes: {sum(_bytes.values())}")


def register_exit_summary() -> None:
    global _registered
    if enabled() and not _registered:
        atexit.register(summary)
        _registered = True
