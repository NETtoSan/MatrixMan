"""Opt-in MatrixMan profiling state and instrumentation."""

from __future__ import annotations

import functools
import atexit
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

from . import gpumatrix as gm
from ...config import profiling_enabled


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"}


enabled = profiling_enabled()
detail = _env_flag("MATRIXMAN_PROFILE_DETAIL")
gpu_timing_enabled = _env_flag("MATRIXMAN_GPU_TIMING")
started = time.perf_counter()
ops: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "total": 0.0, "max": 0.0})
counters: dict[str, float] = defaultdict(float)
parameters: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "bytes": 0, "repeated": 0})
parameter_keys: set[tuple[str, int]] = set()
conv: dict[str, float] = defaultdict(float)

_GL_EXTENSIONS = 0x1F03
_GL_TIME_ELAPSED = 0x88BF
_GL_QUERY_RESULT = 0x8866
_GL_QUERY_RESULT_AVAILABLE = 0x8867
_gpu_timer_capable = False
_gpu_timer_api = "unavailable"
_gpu_timer_reason = "disabled"
_gpu_timer_gen = None
_gpu_timer_delete = None
_gpu_timer_begin = None
_gpu_timer_end = None
_gpu_timer_available = None
_gpu_timer_result = None
_gpu_timer_free: list[int] = []
_gpu_timer_all: set[int] = set()
_gpu_timer_pending: list[tuple[int, str, dict | None]] = []
_gpu_timer_dropped = 0
gpu_timings: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "total": 0.0, "max": 0.0})
gpu_timing_samples: list[dict] = []
_exit_hook_registered = False


def set_enabled(value: bool) -> None:
    """Update the OpenGL profiler without importing another backend."""
    global enabled
    enabled = bool(value)
    if enabled:
        register_exit_hook()
    if enabled or gpu_timing_enabled:
        gm.glBegin = _profile_gl_begin
        gm.glFinish = _profile_gl_finish
        gm.glFlush = _profile_gl_flush
    # These compatibility aliases are consumed by the OpenGL operation
    # modules. Keep them synchronized when Python configuration changes after
    # backend import.
    for module_name in (
        "drivers.matrixman.backends.opengl.backend",
        "drivers.matrixman.backends.opengl.implementation",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "_profile_enabled"):
            module._profile_enabled = enabled


def is_enabled() -> bool:
    """Return the OpenGL profiler's current runtime state."""
    return enabled


def register_exit_hook() -> None:
    """Register OpenGL reporting only after OpenGL has been selected."""
    global _exit_hook_registered
    if _exit_hook_registered:
        return

    def report_if_used() -> None:
        from ...backend import active_backend

        active = active_backend()
        if (
            active is not None
            and active.name == "opengl"
            and enabled
            and (ops or counters or gpu_timings or conv)
        ):
            report()

    atexit.register(report_if_used)
    _exit_hook_registered = True


def initialize_gpu_timing() -> None:
    """Probe optional timer-query support after an OpenGL context exists."""
    global _gpu_timer_capable, _gpu_timer_api, _gpu_timer_reason
    global _gpu_timer_gen, _gpu_timer_delete, _gpu_timer_begin, _gpu_timer_end
    global _gpu_timer_available, _gpu_timer_result
    if not gpu_timing_enabled:
        _gpu_timer_reason = "disabled (set MATRIXMAN_GPU_TIMING=1)"
        return
    try:
        extensions = gm.glGetString(_GL_EXTENSIONS) or b""
        extension_names = set(extensions.decode("ascii", "replace").split())
        if "GL_ARB_timer_query" in extension_names:
            _gpu_timer_api = "GL_ARB_timer_query / GL_TIME_ELAPSED"
        elif "GL_EXT_timer_query" in extension_names:
            _gpu_timer_api = "GL_EXT_timer_query / GL_TIME_ELAPSED_EXT"
        else:
            _gpu_timer_reason = "timer-query extension unavailable"
            return
        import ctypes
        def load(names, restype, *argtypes):
            for name in names:
                try:
                    return gm.proc(name, restype, *argtypes)
                except Exception:
                    continue
            raise RuntimeError(f"none of the query entry points are available: {names}")

        _gpu_timer_gen = load(("glGenQueries", "glGenQueriesARB"), None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        _gpu_timer_delete = load(("glDeleteQueries", "glDeleteQueriesARB"), None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        _gpu_timer_begin = load(("glBeginQuery", "glBeginQueryARB"), None, ctypes.c_uint, ctypes.c_uint)
        _gpu_timer_end = load(("glEndQuery", "glEndQueryARB"), None, ctypes.c_uint)
        _gpu_timer_available = load(("glGetQueryObjectiv", "glGetQueryObjectivARB"), None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        _gpu_timer_result = load(("glGetQueryObjectui64v", "glGetQueryObjectui64vEXT"), None, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64))
        _gpu_timer_capable = True
        _gpu_timer_reason = "available"
    except Exception as exc:
        _gpu_timer_capable = False
        _gpu_timer_api = "unavailable"
        _gpu_timer_reason = f"query API unavailable: {exc}"


def collect_gpu_timing() -> int:
    """Collect only completed queries; never waits for a query result."""
    if not _gpu_timer_capable:
        return 0
    import ctypes
    collected = 0
    remaining = []
    for query, label, metadata in _gpu_timer_pending:
        available = ctypes.c_int()
        _gpu_timer_available(query, _GL_QUERY_RESULT_AVAILABLE, ctypes.byref(available))
        if not available.value:
            remaining.append((query, label, metadata))
            continue
        nanoseconds = ctypes.c_uint64()
        _gpu_timer_result(query, _GL_QUERY_RESULT, ctypes.byref(nanoseconds))
        elapsed = nanoseconds.value / 1_000_000_000.0
        record = gpu_timings[label]
        record["calls"] += 1
        record["total"] += elapsed
        record["max"] = max(record["max"], elapsed)
        gpu_timing_samples.append({"label": label, "elapsed": elapsed, "metadata": metadata})
        _gpu_timer_free.append(query)
        collected += 1
    _gpu_timer_pending[:] = remaining
    return collected


@contextmanager
def gpu_timer(label: str, metadata: dict | None = None):
    """Time one non-nested GPU operation without synchronizing the CPU."""
    global _gpu_timer_dropped
    if not _gpu_timer_capable:
        yield
        return
    import ctypes
    query = _gpu_timer_free.pop() if _gpu_timer_free else 0
    try:
        if not query:
            value = ctypes.c_uint()
            _gpu_timer_gen(1, ctypes.byref(value))
            query = int(value.value)
            _gpu_timer_all.add(query)
        _gpu_timer_begin(_GL_TIME_ELAPSED, query)
    except Exception:
        _gpu_timer_dropped += 1
        if query:
            _gpu_timer_free.append(query)
        yield
        return
    try:
        yield
    finally:
        try:
            _gpu_timer_end(_GL_TIME_ELAPSED)
            _gpu_timer_pending.append((query, label, metadata))
        except Exception:
            _gpu_timer_dropped += 1
            _gpu_timer_free.append(query)


def shutdown_gpu_timing() -> None:
    if _gpu_timer_delete is None or not _gpu_timer_all:
        return
    import ctypes
    values = (ctypes.c_uint * len(_gpu_timer_all))(*_gpu_timer_all)
    _gpu_timer_delete(len(values), values)
    _gpu_timer_all.clear()
    _gpu_timer_free.clear()
    _gpu_timer_pending.clear()


def reset() -> None:
    global started, _gpu_timer_dropped
    started = time.perf_counter()
    ops.clear()
    counters.clear()
    parameters.clear()
    parameter_keys.clear()
    conv.clear()
    gpu_timings.clear()
    gpu_timing_samples.clear()
    _gpu_timer_dropped = 0


def _profile_gl_begin(mode):
    if not enabled:
        return _profile_gl_begin.original(mode)
    counters["draw_calls"] += 1
    return _profile_gl_begin.original(mode)


def _profile_gl_finish():
    if not enabled and not gpu_timing_enabled:
        return _profile_gl_finish.original()
    begin = time.perf_counter()
    result = _profile_gl_finish.original()
    if enabled:
        counters["glFinish_calls"] += 1
        counters["glFinish_seconds"] += time.perf_counter() - begin
    collect_gpu_timing()
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
if enabled or gpu_timing_enabled:
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
    collect_gpu_timing()
    if not enabled and not gpu_timing_enabled:
        return
    elapsed = time.perf_counter() - started
    print("\nMatrixMan profile\n-----------------")
    print(f"total backend time: {elapsed:.3f}s")
    names = ("convolution.default", "native_batch_norm.default", "silu_.default", "add.Tensor", "mul.Tensor", "div.Tensor", "sigmoid.default", "mm.default", "cat.default", "max_pool2d_with_indices.default", "upsample_nearest2d.default", "_softmax.default")
    for name in names:
        record = ops.get(name)
        if record:
            print(f"{name}: calls={int(record['calls'])} total={record['total']:.3f}s average={record['total']/record['calls']:.3f}s max={record['max']:.3f}s")
    print("CPU-side dispatch timing:")
    print("  timings below measure Python/dispatch duration and are not GPU execution time")
    print("OpenGL:")
    print(f"  draw calls: {int(counters['draw_calls'])}")
    print(f"  tiled convolution draw calls: {int(counters['tiled_draw_calls'])}")
    print(f"  consolidation draw calls: {int(counters['consolidation_draw_calls'])}")
    print(f"  glFinish: {int(counters['glFinish_calls'])} ({counters['glFinish_seconds']:.3f}s)")
    print(f"  glFlush: {int(counters['glFlush_calls'])} ({counters['glFlush_seconds']:.3f}s)")
    print(f"  pre-consolidation glFinish executed: {int(counters['pre_consolidation_sync_calls'])}")
    print(f"  pre-consolidation glFinish skipped: {int(counters['pre_consolidation_sync_skips'])}")
    print(f"  tiled convolution sync mode: {os.environ.get('MATRIXMAN_TILE_SYNC', 'per_tile')}")
    print(f"  physical tile limit: {os.environ.get('MATRIXMAN_TILE_LIMIT', '256')}")
    print("GPU timing:")
    print(f"  timer-query capability: {'available' if _gpu_timer_capable else 'unavailable'}")
    print(f"  API: {_gpu_timer_api}")
    print(f"  reason: {_gpu_timer_reason}")
    print(f"  query count: {len(_gpu_timer_all)}")
    print(f"  unresolved queries: {len(_gpu_timer_pending)}")
    print(f"  dropped queries: {_gpu_timer_dropped}")
    print(f"  total measured GPU time: {sum(item['total'] for item in gpu_timings.values()):.6f}s")
    for label, record in gpu_timings.items():
        print(f"  {label}: calls={int(record['calls'])} total={record['total']:.6f}s average={record['total']/record['calls']:.6f}s max={record['max']:.6f}s")
    conv_samples = [sample for sample in gpu_timing_samples if sample["label"] == "Conv2D" and sample["metadata"]]
    if conv_samples:
        total_conv = sum(sample["elapsed"] for sample in conv_samples)
        total_gpu = sum(item["total"] for item in gpu_timings.values())
        print("Top Conv2D GPU operations:")
        for index, sample in enumerate(sorted(conv_samples, key=lambda item: item["elapsed"], reverse=True)[:10], 1):
            metadata = sample["metadata"]
            print(
                f"  {index}. input={metadata['input_shape']} -> output={metadata['output_shape']} "
                f"weight={metadata['weight_shape']} kernel={metadata['kernel']} "
                f"stride={metadata['stride']} padding={metadata['padding']} dilation={metadata['dilation']} "
                f"groups={metadata['groups']} elements={metadata['logical_output_elements']} "
                f"atlas={metadata['atlas']} macs/output={metadata['macs_per_output']} "
                f"tiled={metadata['tiled']} physical_tiles={metadata['physical_tile_count']} "
                f"gpu={sample['elapsed']:.6f}s conv_share={sample['elapsed']/total_conv*100:.2f}% "
                f"total_gpu_share={sample['elapsed']/total_gpu*100:.2f}%"
            )
        grouped = defaultdict(lambda: {"count": 0, "total": 0.0, "metadata": None})
        for sample in conv_samples:
            key = tuple(sorted((key, repr(value)) for key, value in sample["metadata"].items()))
            grouped[key]["count"] += 1
            grouped[key]["total"] += sample["elapsed"]
            grouped[key]["metadata"] = sample["metadata"]
        print("Repeated Conv2D signatures:")
        for group in sorted(grouped.values(), key=lambda item: item["total"], reverse=True)[:10]:
            metadata = group["metadata"]
            print(
                f"  calls={group['count']} input={metadata['input_shape']} -> output={metadata['output_shape']} "
                f"kernel={metadata['kernel']} groups={metadata['groups']} atlas={metadata['atlas']} "
                f"tiled={metadata['tiled']} physical_tiles={metadata['physical_tile_count']} "
                f"aggregate_gpu={group['total']:.6f}s"
            )
    print(f"  texture allocations: {int(counters['texture_allocations'])}")
    print(f"  scratch texture allocations: {int(counters['scratch_texture_allocations'])}")
    print(f"  scratch texture reuses: {int(counters['scratch_texture_reuses'])}")
    print(f"  scratch texture releases: {int(counters['scratch_texture_releases'])}")
    print(f"  scratch texture evictions: {int(counters['scratch_texture_evictions'])}")
    print(f"  texture uploads: {int(counters['texture_uploads'])} ({int(counters['texture_upload_bytes'])} bytes, {counters['texture_upload_seconds']:.3f}s)")
    print("parameter uploads:")
    print(f"  count: {int(counters['parameter_uploads'])}")
    print(f"  bytes: {int(counters['parameter_upload_bytes'])}")
    print(f"  repeated: {int(counters['repeated_parameter_uploads'])}")
    print(f"  cache hits: {int(counters['parameter_cache_hits'])}")
    print(f"  cache misses: {int(counters['parameter_cache_misses'])}")
    print(f"  cache invalidations: {int(counters['parameter_cache_invalidations'])}")
    print(f"  cache bypasses: {int(counters['parameter_cache_bypasses'])}")
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
