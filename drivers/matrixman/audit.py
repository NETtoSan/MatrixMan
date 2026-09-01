"""Opt-in audit of MatrixMan tensor paths that materialize on the CPU."""

from __future__ import annotations

import atexit
import os
import traceback
from collections import Counter


_enabled = os.environ.get("MATRIXMAN_AUDIT_CPU_LEAKS", "").strip().lower() not in {"", "0", "false", "no", "off"}
_counts = Counter()
_elements = Counter()
_bytes = Counter()
_registered = False
_summary_printed = False


def enabled() -> bool:
    return _enabled


def count(category: str) -> int:
    return int(_counts[category])


def record(category: str, *, op: str | None = None, tensor=None, shape=None,
           dtype=None, numel=None, storage=None, reason: str = "",
           cpu_arithmetic: bool = False) -> None:
    if not _enabled:
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
    warning = category == "unexpected_cpu_materialization"
    prefix = "[MatrixMan/AUDIT][WARNING]" if warning else "[MatrixMan/AUDIT]"
    print(f"{prefix} category={category}")
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
        print("[MatrixMan/AUDIT][WARNING] nonzero CPU arithmetic observed on MatrixMan-originating data")


def summary() -> None:
    global _summary_printed
    if not _enabled:
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
    if _enabled and not _registered:
        atexit.register(summary)
        _registered = True
