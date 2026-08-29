"""Opt-in MatrixMan profiling state and instrumentation."""

from __future__ import annotations

import functools
import os
import time
from collections import defaultdict

from . import gpumatrix as gm


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"}


enabled = _env_flag("MATRIXMAN_PROFILE")
detail = _env_flag("MATRIXMAN_PROFILE_DETAIL")
started = time.perf_counter()
ops: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "total": 0.0, "max": 0.0})
counters: dict[str, float] = defaultdict(float)
parameters: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "bytes": 0, "repeated": 0})
parameter_keys: set[tuple[str, int]] = set()
conv: dict[str, float] = defaultdict(float)


def reset() -> None:
    global started
    started = time.perf_counter()
    ops.clear()
    counters.clear()
    parameters.clear()
    parameter_keys.clear()
    conv.clear()


def _profile_gl_begin(mode):
    if not enabled:
        return _profile_gl_begin.original(mode)
    counters["draw_calls"] += 1
    return _profile_gl_begin.original(mode)


def _profile_gl_finish():
    if not enabled:
        return _profile_gl_finish.original()
    begin = time.perf_counter()
    result = _profile_gl_finish.original()
    counters["glFinish_calls"] += 1
    counters["glFinish_seconds"] += time.perf_counter() - begin
    return result


def _profile_gl_flush():
    if not enabled:
        return _profile_gl_flush.original()
    begin = time.perf_counter()
    result = _profile_gl_flush.original()
    counters["glFlush_calls"] += 1
    counters["glFlush_seconds"] += time.perf_counter() - begin
    return result


_profile_gl_begin.original = gm.glBegin
_profile_gl_finish.original = gm.glFinish
_profile_gl_flush.original = gm.glFlush
if enabled:
    gm.glBegin = _profile_gl_begin
    gm.glFinish = _profile_gl_finish
    gm.glFlush = _profile_gl_flush


def dispatch_timer(fn):
    @functools.wraps(fn)
    def wrapped(cls, func, types, args=(), kwargs=None):
        if not enabled:
            return fn(cls, func, types, args, kwargs)
        name = str(func).removeprefix("aten.")
        begin = time.perf_counter()
        try:
            return fn(cls, func, types, args, kwargs)
        finally:
            elapsed = time.perf_counter() - begin
            record = ops[name]
            record["calls"] += 1
            record["total"] += elapsed
            record["max"] = max(record["max"], elapsed)
            if detail:
                print(f"[MatrixMan profile] {name}: {elapsed:.6f}s")
    return wrapped


def report() -> None:
    if not enabled:
        return
    elapsed = time.perf_counter() - started
    print("\nMatrixMan profile\n-----------------")
    print(f"total backend time: {elapsed:.3f}s")
    names = ("convolution.default", "native_batch_norm.default", "silu_.default", "add.Tensor", "mul.Tensor", "div.Tensor", "sigmoid.default", "mm.default", "cat.default", "max_pool2d_with_indices.default", "upsample_nearest2d.default", "_softmax.default")
    for name in names:
        record = ops.get(name)
        if record:
            print(f"{name}: calls={int(record['calls'])} total={record['total']:.3f}s average={record['total']/record['calls']:.3f}s max={record['max']:.3f}s")
    print("OpenGL:")
    print(f"  draw calls: {int(counters['draw_calls'])}")
    print(f"  tiled convolution draw calls: {int(counters['tiled_draw_calls'])}")
    print(f"  consolidation draw calls: {int(counters['consolidation_draw_calls'])}")
    print(f"  glFinish: {int(counters['glFinish_calls'])} ({counters['glFinish_seconds']:.3f}s)")
    print(f"  glFlush: {int(counters['glFlush_calls'])} ({counters['glFlush_seconds']:.3f}s)")
    print(f"  tiled convolution sync mode: {os.environ.get('MATRIXMAN_TILE_SYNC', 'per_tile')}")
    print(f"  physical tile limit: {os.environ.get('MATRIXMAN_TILE_LIMIT', '256')}")
    print(f"  texture allocations: {int(counters['texture_allocations'])}")
    print(f"  texture uploads: {int(counters['texture_uploads'])} ({int(counters['texture_upload_bytes'])} bytes, {counters['texture_upload_seconds']:.3f}s)")
    print("parameter uploads:")
    print(f"  count: {int(counters['parameter_uploads'])}")
    print(f"  bytes: {int(counters['parameter_upload_bytes'])}")
    print(f"  repeated: {int(counters['repeated_parameter_uploads'])}")
    print("readback:")
    print(f"  calls: {int(counters['readback_calls'])} bytes: {int(counters['readback_bytes'])}")
    print(f"  sync/wait: {counters['readback_sync_seconds']:.3f}s")
    print(f"  transfer: {counters['readback_transfer_seconds']:.3f}s")
    print(f"  conversion/tensor creation: {counters['readback_conversion_seconds']:.3f}s")
    print(f"  total: {counters['readback_total_seconds']:.3f}s")
    if conv:
        print("Conv2D breakdown (aggregate):")
        for key in ("prepare", "parameter_upload", "shader_setup", "tile_render", "sync", "consolidation"):
            print(f"  {key}: {conv[key]:.3f}s")
        print(f"  tiled calls: {int(counters['tiled_conv_calls'])} tiles: {int(counters['tiled_conv_tiles'])} max physical tile: {int(counters['tiled_conv_max_tile_width'])}x{int(counters['tiled_conv_max_tile_height'])}")
    slow = sorted(ops.items(), key=lambda item: item[1]["total"], reverse=True)[:3]
    if slow:
        print("Top slow operations:")
        for index, (name, record) in enumerate(slow, 1):
            print(f"  {index}. {name}: {record['total']:.3f}s ({int(record['calls'])} calls)")
