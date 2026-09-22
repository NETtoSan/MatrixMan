"""Opt-in MatrixMan profiling state and instrumentation."""

from __future__ import annotations

import functools
import atexit
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

from . import gpumatrix as gm
from ...config import config, profiling_enabled
from ... import frontend_profiling


enabled = profiling_enabled()


def _gpu_timing_enabled() -> bool:
    return bool(config.gpuTiming)


def _profile_detail() -> bool:
    return bool(config.profileDetail)


def detailed_enabled() -> bool:
    """Return whether high-volume CPU-stage/GL-call instrumentation is active."""
    return bool(enabled and _profile_detail())


_thread_cpu_clock = getattr(time, "thread_time", None)


def thread_cpu_time() -> float:
    """Return calling-thread CPU time, with a documented fallback."""
    return (_thread_cpu_clock or time.process_time)()


def thread_cpu_supported() -> bool:
    return _thread_cpu_clock is not None


def sync_finish(reason: str):
    """Run one glFinish and attribute its wall time to a sync reason."""
    if not enabled:
        return gm.glFinish()
    wall_started = time.perf_counter()
    cpu_started = thread_cpu_time()
    try:
        return gm.glFinish()
    finally:
        record = sync_timings[str(reason)]
        record["calls"] += 1
        record["wall_seconds"] += time.perf_counter() - wall_started
        record["thread_cpu_seconds"] += thread_cpu_time() - cpu_started


started = time.perf_counter()
ops: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "total": 0.0, "thread_cpu_seconds": 0.0, "max": 0.0}
)
counters: dict[str, float] = defaultdict(float)
parameters: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "bytes": 0, "repeated": 0})
parameter_keys: set[tuple[str, int]] = set()
parameter_groups: dict[tuple, dict[str, float]] = defaultdict(
    lambda: {
        "upload_count": 0,
        "repeated_count": 0,
        "bytes": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_invalidations": 0,
    }
)
conv: dict[str, float] = defaultdict(float)
stage_timings: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "wall_seconds": 0.0, "thread_cpu_seconds": 0.0}
)
gl_call_timings: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "redundant_calls": 0, "wall_seconds": 0.0, "thread_cpu_seconds": 0.0}
)
conv_stage_timings: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "wall_seconds": 0.0, "thread_cpu_seconds": 0.0}
)
sync_timings: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "wall_seconds": 0.0, "thread_cpu_seconds": 0.0}
)
conv_cost_records: list[dict] = []
_gl_state = {
    "program": 0,
    "active_texture": 0,
    "textures": {},
    "framebuffer": 0,
    "viewport": None,
    "uniforms": {},
}

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
    if enabled or _gpu_timing_enabled() or frontend_profiling.enabled:
        gm.glBegin = _profile_gl_begin
        gm.glFinish = _profile_gl_finish
        gm.glFlush = _profile_gl_flush
    install_gl_profilers()
    install_program_profiler()
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
    if not _gpu_timing_enabled():
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
    parameter_groups.clear()
    conv.clear()
    stage_timings.clear()
    gl_call_timings.clear()
    conv_stage_timings.clear()
    sync_timings.clear()
    conv_cost_records.clear()
    _gl_state.update({
        "program": 0,
        "active_texture": 0,
        "textures": {},
        "framebuffer": 0,
        "viewport": None,
        "uniforms": {},
    })
    gpu_timings.clear()
    gpu_timing_samples.clear()
    _gpu_timer_dropped = 0
    try:
        from . import glstate
        glstate.reset_counters()
    except ImportError:
        pass


def record_convolution_cost(input_shape, output_shape, descriptor, source=None) -> None:
    """Record static Conv MAC metadata without touching GPU tensor contents."""
    if not enabled:
        return
    input_shape = tuple(int(value) for value in input_shape)
    output_shape = tuple(int(value) for value in output_shape)
    n, cin, _hin, _win = input_shape
    _nout, cout, hout, wout = output_shape
    kh, kw = descriptor.kernel
    groups = int(descriptor.groups)
    macs = int(n * hout * wout * cout * (cin // groups) * kh * kw)
    if source is None:
        source_name = "<unknown>"
        source_path = "<unknown>"
        source_class = "<unknown>"
    else:
        source_name = str(source.qualified_name)
        source_path = str(source.module_path)
        source_class = source_name.rsplit(".", 1)[-1]
    conv_cost_records.append({
        "source_module": source_name,
        "source_path": source_path,
        "source_class": source_class,
        "family": {"dense": "Conv2D", "depthwise": "DWConv", "grouped": "GroupedConv"}.get(
            descriptor.path, str(descriptor.path)
        ),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "kernel": tuple(int(value) for value in descriptor.kernel),
        "stride": tuple(int(value) for value in descriptor.stride),
        "groups": groups,
        "cin": cin,
        "cout": cout,
        "macs": macs,
    })


def _parameter_group_key(operation: str | None, parameter_role: str, shape, cacheable: bool) -> tuple:
    return (
        operation or "unknown",
        parameter_role,
        tuple(int(value) for value in shape),
        "cacheable" if cacheable else "bypassed",
    )


def record_parameter_upload(
    array,
    parameter_kind: str,
    *,
    operation: str | None = None,
    parameter_role: str | None = None,
    cacheable: bool = False,
) -> None:
    """Record one parameter upload and its operation/role attribution."""
    if not enabled:
        return
    key = (parameter_kind, int(array.__array_interface__["data"][0]))
    counters["parameter_uploads"] += 1
    counters["parameter_upload_bytes"] += array.nbytes
    parameters[parameter_kind]["count"] += 1
    parameters[parameter_kind]["bytes"] += array.nbytes
    repeated = key in parameter_keys
    if repeated:
        counters["repeated_parameter_uploads"] += 1
        parameters[parameter_kind]["repeated"] += 1
    parameter_keys.add(key)

    group = parameter_groups[_parameter_group_key(operation, parameter_role or parameter_kind, array.shape, cacheable)]
    group["upload_count"] += 1
    group["bytes"] += array.nbytes
    if repeated:
        group["repeated_count"] += 1


def record_parameter_cache_event(
    operation: str | None,
    parameter_role: str,
    shape,
    event: str,
) -> None:
    """Attribute cache hits, misses, and invalidations to a parameter group."""
    if not enabled:
        return
    key = _parameter_group_key(operation, parameter_role, shape, True)
    group = parameter_groups[key]
    field = {
        "hit": "cache_hits",
        "miss": "cache_misses",
        "invalidation": "cache_invalidations",
    }.get(event)
    if field is not None:
        group[field] += 1


def record_timing(name: str, wall_seconds: float, thread_cpu_seconds: float, *, count: int = 1) -> None:
    """Record wall and calling-thread CPU timing for a diagnostic stage."""
    if not detailed_enabled():
        return
    record = stage_timings[name]
    record["calls"] += count
    record["wall_seconds"] += wall_seconds
    record["thread_cpu_seconds"] += thread_cpu_seconds


@contextmanager
def stage(name: str):
    """Measure a CPU-side stage with elapsed and calling-thread CPU clocks."""
    if not detailed_enabled():
        yield
        return
    wall_started = time.perf_counter()
    cpu_started = thread_cpu_time()
    try:
        yield
    finally:
        record_timing(
            name,
            time.perf_counter() - wall_started,
            thread_cpu_time() - cpu_started,
        )


@contextmanager
def conv_stage(name: str):
    """Measure a Conv2D setup/render sub-stage and calling-thread CPU time."""
    if not detailed_enabled():
        yield
        return
    wall_started = time.perf_counter()
    cpu_started = thread_cpu_time()
    try:
        yield
    finally:
        wall_seconds = time.perf_counter() - wall_started
        thread_cpu_seconds = thread_cpu_time() - cpu_started
        record_timing(f"Conv2D/{name}", wall_seconds, thread_cpu_seconds)
        record = conv_stage_timings[name]
        record["calls"] += 1
        record["wall_seconds"] += wall_seconds
        record["thread_cpu_seconds"] += thread_cpu_seconds


class ProgramCache(dict):
    """Runtime program dictionary that measures membership/cache lookups."""

    def __init__(self, label: str, initial=None):
        super().__init__(initial or {})
        self.label = label

    def __contains__(self, key):
        if not detailed_enabled():
            return dict.__contains__(self, key)
        wall_started = time.perf_counter()
        cpu_started = thread_cpu_time()
        hit = dict.__contains__(self, key)
        wall_seconds = time.perf_counter() - wall_started
        thread_cpu_seconds = thread_cpu_time() - cpu_started
        record_timing("program_lookup", wall_seconds, thread_cpu_seconds)
        counters["program_cache_lookups"] += 1
        counters["program_cache_hits" if hit else "program_cache_misses"] += 1
        return hit


def install_program_cache_profilers(runtime_state) -> None:
    """Wrap runtime program maps without changing their lookup semantics."""
    names = (
        name for name in vars(runtime_state)
        if name.endswith("_programs")
    )
    for name in names:
        value = getattr(runtime_state, name)
        if not isinstance(value, ProgramCache):
            setattr(runtime_state, name, ProgramCache(name, value))


def _record_gl_call(name: str, original, args, redundant: bool):
    if not detailed_enabled():
        if frontend_profiling.enabled and name in {"glBegin", "glEnd", "glVertex2f"}:
            with frontend_profiling.component("OpenGL API/draw submission"):
                return original(*args)
        return original(*args)
    wall_started = time.perf_counter()
    cpu_started = thread_cpu_time()
    try:
        return original(*args)
    finally:
        wall_seconds = time.perf_counter() - wall_started
        thread_cpu_seconds = thread_cpu_time() - cpu_started
        record = gl_call_timings[name]
        record["calls"] += 1
        record["redundant_calls"] += int(redundant)
        record["wall_seconds"] += wall_seconds
        record["thread_cpu_seconds"] += thread_cpu_seconds
        stage_name = {
            "glUseProgram": "glUseProgram",
            "glUniform1i": "uniform_setup",
            "glGetUniformLocation": "program_lookup",
            "glActiveTexture": "texture_binding",
            "glBindTexture": "texture_binding",
            "glBindFramebuffer": "fbo_output_binding",
            "glFramebufferTexture2D": "fbo_output_binding",
            "glCheckFramebufferStatus": "fbo_output_binding",
            "glViewport": "viewport_state_setup",
            "glBegin": "draw_submission",
            "glEnd": "draw_submission",
            "glVertex2f": "draw_submission",
            "glFinish": "synchronization_wait",
            "glFlush": "synchronization_wait",
        }.get(name)
        if stage_name:
            record_timing(stage_name, wall_seconds, thread_cpu_seconds)


def _gl_redundant(name: str, args) -> bool:
    if name == "glUseProgram":
        value = int(args[0])
        redundant = value == _gl_state["program"]
        _gl_state["program"] = value
        return redundant
    if name == "glActiveTexture":
        value = int(args[0])
        redundant = value == _gl_state["active_texture"]
        _gl_state["active_texture"] = value
        return redundant
    if name == "glBindTexture":
        target, texture = int(args[0]), int(args[1])
        key = (_gl_state["active_texture"], target)
        redundant = _gl_state["textures"].get(key) == texture
        _gl_state["textures"][key] = texture
        return redundant
    if name == "glBindFramebuffer":
        value = int(args[1])
        redundant = value == _gl_state["framebuffer"]
        _gl_state["framebuffer"] = value
        return redundant
    if name == "glViewport":
        value = tuple(int(item) for item in args)
        redundant = value == _gl_state["viewport"]
        _gl_state["viewport"] = value
        return redundant
    if name == "glUniform1i":
        key = (_gl_state["program"], int(args[0]))
        value = int(args[1])
        redundant = _gl_state["uniforms"].get(key) == value
        _gl_state["uniforms"][key] = value
        return redundant
    return False


def _make_gl_profiler(name: str, original):
    @functools.wraps(original)
    def wrapped(*args):
        redundant = _gl_redundant(name, args) if detailed_enabled() else False
        return _record_gl_call(name, original, args, redundant)
    wrapped._matrixman_profile_wrapper = True
    wrapped.original = original
    return wrapped


def install_gl_profilers() -> None:
    """Install diagnostic wrappers after context-specific GL functions load."""
    if not detailed_enabled() and not frontend_profiling.enabled:
        return
    names = (
        "glUseProgram", "glUniform1i", "glGetUniformLocation",
        "glActiveTexture", "glBindTexture", "glBindFramebuffer",
        "glFramebufferTexture2D", "glCheckFramebufferStatus", "glViewport",
        "glEnd", "glVertex2f",
    )
    for name in names:
        original = getattr(gm, name, None)
        if original is not None and not getattr(original, "_matrixman_profile_wrapper", False):
            setattr(gm, name, _make_gl_profiler(name, original))


def install_program_profiler() -> None:
    """Time actual shader compile/link calls, which only occur on cache misses."""
    if not detailed_enabled():
        return
    original = gm.make_program
    if getattr(original, "_matrixman_profile_wrapper", False):
        return

    @functools.wraps(original)
    def wrapped(fragment_source):
        if not detailed_enabled():
            return original(fragment_source)
        with stage("shader_compile_link"):
            result = original(fragment_source)
        counters["shader_compile_link_calls"] += 1
        return result

    wrapped._matrixman_profile_wrapper = True
    wrapped.original = original
    gm.make_program = wrapped


def diagnostic_snapshot() -> dict:
    """Return JSON-safe stage and GL-call telemetry for benchmark frame deltas."""
    from . import glstate
    return {
        "stages": {name: dict(value) for name, value in stage_timings.items()},
        "gl_calls": {name: dict(value) for name, value in gl_call_timings.items()},
        "synchronization": {name: dict(value) for name, value in sync_timings.items()},
        "sync_counters": {
            name: int(value)
            for name, value in counters.items()
            if "pool" in name or "scratch" in name or "consolidation" in name or "tiled" in name
        },
        "state_cache_skips": glstate.snapshot(),
    }


def parameter_group_snapshot() -> list[dict]:
    """Return JSON-safe operation/role/shape parameter attribution."""
    result = []
    for (operation, parameter_role, shape, cache_status), values in sorted(parameter_groups.items(), key=str):
        result.append({
            "operation": operation,
            "parameter_role": parameter_role,
            "tensor_shape": list(shape),
            "cacheable": cache_status == "cacheable",
            **{name: int(value) for name, value in values.items()},
        })
    return result


def _profile_gl_begin(mode):
    if not enabled:
        return _profile_gl_begin.original(mode)
    counters["draw_calls"] += 1
    return _record_gl_call("glBegin", _profile_gl_begin.original, (mode,), False)


def _profile_gl_finish():
    if not enabled and not _gpu_timing_enabled():
        return _profile_gl_finish.original()
    synchronization_scope = (
        frontend_profiling.component("OpenGL synchronization wait")
        if frontend_profiling.enabled else nullcontext()
    )
    if detailed_enabled():
        before = gl_call_timings["glFinish"]["wall_seconds"]
        with synchronization_scope:
            result = _record_gl_call("glFinish", _profile_gl_finish.original, (), False)
        elapsed = gl_call_timings["glFinish"]["wall_seconds"] - before
    else:
        wall_started = time.perf_counter()
        with synchronization_scope:
            result = _profile_gl_finish.original()
        elapsed = time.perf_counter() - wall_started
    if enabled:
        counters["glFinish_calls"] += 1
        counters["glFinish_seconds"] += elapsed
    collect_gpu_timing()
    return result


def _profile_gl_flush():
    if not enabled:
        return _profile_gl_flush.original()
    if detailed_enabled():
        before = gl_call_timings["glFlush"]["wall_seconds"]
        result = _record_gl_call("glFlush", _profile_gl_flush.original, (), False)
        elapsed = gl_call_timings["glFlush"]["wall_seconds"] - before
    else:
        wall_started = time.perf_counter()
        result = _profile_gl_flush.original()
        elapsed = time.perf_counter() - wall_started
    counters["glFlush_calls"] += 1
    counters["glFlush_seconds"] += elapsed
    return result


_profile_gl_begin.original = gm.glBegin
_profile_gl_finish.original = gm.glFinish
_profile_gl_flush.original = gm.glFlush
if enabled or _gpu_timing_enabled() or frontend_profiling.enabled:
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
        cpu_begin = thread_cpu_time()
        try:
            with stage("matrixman_python_op_wrapper"):
                return fn(cls, func, types, args, kwargs)
        finally:
            elapsed = time.perf_counter() - begin
            cpu_elapsed = thread_cpu_time() - cpu_begin
            record = ops[name]
            record["calls"] += 1
            record["total"] += elapsed
            record["thread_cpu_seconds"] += cpu_elapsed
            record["max"] = max(record["max"], elapsed)
            if _profile_detail():
                print(f"[MatrixMan profile] {name}: {elapsed:.6f}s")
    return wrapped


def report(frame_count: int | None = None) -> None:
    collect_gpu_timing()
    if not enabled and not _gpu_timing_enabled():
        return
    elapsed = time.perf_counter() - started
    print("\nMatrixMan profile\n-----------------")
    print(f"total backend time: {elapsed:.3f}s")
    if conv_cost_records:
        total_macs = sum(record["macs"] for record in conv_cost_records)
        print("CONVOLUTION COST (static MACs)")
        print(f"  calls={len(conv_cost_records)} total_MACs={total_macs:,}")
        if frame_count:
            print(f"  calls/frame={len(conv_cost_records) / frame_count:.2f} MACs/frame={total_macs / frame_count:,.0f}")
        print("  ranked convolution operations:")
        for rank, record in enumerate(sorted(conv_cost_records, key=lambda item: item["macs"], reverse=True), 1):
            percent = (100.0 * record["macs"] / total_macs) if total_macs else 0.0
            print(
                f"    {rank:>3}. {record['source_module']} "
                f"class={record['source_class']} path={record['source_path']} "
                f"family={record['family']} "
                f"input={list(record['input_shape'])} output={list(record['output_shape'])} "
                f"kernel={record['kernel'][0]}x{record['kernel'][1]} "
                f"stride={record['stride'][0]}x{record['stride'][1]} "
                f"groups={record['groups']} MACs={record['macs']:,} "
                f"percent={percent:.2f}%"
            )

        def _aggregate(key: str):
            totals = defaultdict(lambda: {"calls": 0, "macs": 0})
            for record in conv_cost_records:
                item = totals[record[key]]
                item["calls"] += 1
                item["macs"] += record["macs"]
            return sorted(totals.items(), key=lambda item: item[1]["macs"], reverse=True)

        print("  by primitive family:")
        for name, values in _aggregate("family"):
            percent = 100.0 * values["macs"] / total_macs if total_macs else 0.0
            print(f"    {name}: calls={values['calls']} MACs={values['macs']:,} percent={percent:.2f}%")
        print("  by source module class:")
        for name, values in _aggregate("source_class"):
            percent = 100.0 * values["macs"] / total_macs if total_macs else 0.0
            print(f"    {name}: calls={values['calls']} MACs={values['macs']:,} percent={percent:.2f}%")
    names = ("convolution.default", "native_batch_norm.default", "silu_.default", "add.Tensor", "mul.Tensor", "div.Tensor", "sigmoid.default", "mm.default", "cat.default", "max_pool2d_with_indices.default", "upsample_nearest2d.default", "_softmax.default")
    for name in names:
        record = ops.get(name)
        if record:
            print(
                f"{name}: calls={int(record['calls'])} total={record['total']:.3f}s "
                f"thread_cpu={record['thread_cpu_seconds']:.3f}s average={record['total']/record['calls']:.3f}s "
                f"max={record['max']:.3f}s"
            )
    print("CPU-side dispatch timing:")
    print("  wall time and calling-thread CPU time; these are not GPU execution time")
    print(f"  calling-thread CPU clock: {'time.thread_time' if thread_cpu_supported() else 'time.process_time fallback'}")
    print("  whole-process process_time is reported only by benchmark/frame summaries")
    print("OpenGL CPU dispatch profile:")
    for name, record in sorted(stage_timings.items()):
        per_frame = (
            f" per_frame_calls={record['calls'] / frame_count:.2f}"
            f" per_frame_wall={record['wall_seconds'] / frame_count:.6f}s"
            f" per_frame_thread_cpu={record['thread_cpu_seconds'] / frame_count:.6f}s"
            if frame_count else ""
        )
        print(
            f"  {name}: calls={int(record['calls'])} wall={record['wall_seconds']:.6f}s "
            f"thread_cpu={record['thread_cpu_seconds']:.6f}s{per_frame}"
        )
    print("OpenGL call counts and redundant-state calls:")
    for name, record in sorted(gl_call_timings.items()):
        per_frame = (
            f" per_frame_calls={record['calls'] / frame_count:.2f}"
            f" per_frame_redundant={record['redundant_calls'] / frame_count:.2f}"
            if frame_count else ""
        )
        print(
            f"  {name}: calls={int(record['calls'])} redundant={int(record['redundant_calls'])} "
            f"wall={record['wall_seconds']:.6f}s thread_cpu={record['thread_cpu_seconds']:.6f}s{per_frame}"
        )
    print(f"  program cache: lookups={int(counters['program_cache_lookups'])} "
          f"hits={int(counters['program_cache_hits'])} misses={int(counters['program_cache_misses'])}")
    print(f"  shader compile/link calls: {int(counters['shader_compile_link_calls'])}")
    print(f"  prepared convolution deferrals: {int(counters['prepared_convolution_deferrals'])}")
    print(f"  fused Conv+BatchNorm calls: {int(counters['fused_batch_norm_calls'])}")
    print(f"  fused Conv+BatchNorm+SiLU calls: {int(counters['fused_silu_calls'])}")
    print(
        "  Conv families: "
        f"dense={int(counters['dense_conv_calls'])} "
        f"dense_1x1_specialized={int(counters['dense_1x1_specialized_calls'])} "
        f"dense_3x3_specialized={int(counters['dense_3x3_specialized_calls'])} "
        f"grouped={int(counters['grouped_conv_calls'])} "
        f"grouped_specialized={int(counters['grouped_conv_specialized_calls'])} "
        f"depthwise_specialized={int(counters['depthwise_specialized_calls'])} "
        f"depthwise_generic_fallback={int(counters['depthwise_generic_fallback_calls'])}"
    )
    print(
        "  depthwise Conv: "
        f"specialized={int(counters['depthwise_specialized_calls'])} "
        f"generic_fallback={int(counters['depthwise_generic_fallback_calls'])} "
        f"program_cache_hits={int(counters['depthwise_program_cache_hits'])} "
        f"program_cache_misses={int(counters['depthwise_program_cache_misses'])} "
        f"tiled_cache_hits={int(counters['depthwise_tile_program_cache_hits'])} "
        f"tiled_cache_misses={int(counters['depthwise_tile_program_cache_misses'])}"
    )
    print(
        "  dense 1x1 program cache: "
        f"hits={int(counters['dense_1x1_program_cache_hits'])} "
        f"misses={int(counters['dense_1x1_program_cache_misses'])}"
    )
    print(
        "  dense 3x3 program cache: "
        f"hits={int(counters['dense_3x3_program_cache_hits'])} "
        f"misses={int(counters['dense_3x3_program_cache_misses'])}"
    )
    print(
        "  grouped 3x3 program cache: "
        f"hits={int(counters['grouped_conv_program_cache_hits'])} "
        f"misses={int(counters['grouped_conv_program_cache_misses'])}"
    )
    print(f"  PrivateUse1 dispatch calls: {int(counters['privateuse1_dispatch_calls'])}")
    try:
        from . import glstate
        skips = glstate.snapshot()
        if skips:
            print("  state-cache-suppressed GL calls: " + ", ".join(
                f"{name}={count}" for name, count in sorted(skips.items())
            ))
    except ImportError:
        pass
    print("Conv2D detailed CPU stages:")
    for name, record in sorted(conv_stage_timings.items()):
        print(
            f"  {name}: calls={int(record['calls'])} wall={record['wall_seconds']:.6f}s "
            f"thread_cpu={record['thread_cpu_seconds']:.6f}s"
        )
    print("OpenGL:")
    print(f"  draw calls: {int(counters['draw_calls'])}")
    print(f"  tiled convolution draw calls: {int(counters['tiled_draw_calls'])}")
    print(f"  consolidation draw calls: {int(counters['consolidation_draw_calls'])}")
    print(f"  glFinish: {int(counters['glFinish_calls'])} ({counters['glFinish_seconds']:.3f}s)")
    print(f"  glFlush: {int(counters['glFlush_calls'])} ({counters['glFlush_seconds']:.3f}s)")
    print(f"  tiled per-tile glFinish calls: {int(counters['tiled_per_tile_sync_calls'])}")
    print(f"  pre-consolidation glFinish executed: {int(counters['pre_consolidation_sync_calls'])}")
    print(f"  pre-consolidation glFinish skipped: {int(counters['pre_consolidation_sync_skips'])}")
    print(f"  post-consolidation glFinish calls: {int(counters['post_consolidation_sync_calls'])}")
    print(
        "  pool-safety glFinish calls: "
        f"{int(counters['activation_pool_reuse_sync_calls'] + counters['activation_pool_eviction_sync_calls'] + counters['scratch_pool_reuse_sync_calls'] + counters['scratch_pool_eviction_sync_calls'])}"
    )
    print(f"  tiled convolution sync mode: {config.tileSync}")
    print(f"  physical tile limit: {config.resolvedTileLimit}")
    print("SYNC PROFILE")
    print(f"  glFinish total: {int(counters['glFinish_calls'])} calls, {counters['glFinish_seconds']:.3f}s")
    sync_order = (
        "per_tile", "pre_consolidation", "post_consolidation",
        "activation_reuse", "activation_release", "activation_eviction",
        "scratch_reuse", "scratch_release", "scratch_eviction",
        "final_output_readback", "backend_synchronize",
    )
    for reason in sync_order:
        record = sync_timings.get(reason, {"calls": 0, "wall_seconds": 0.0, "thread_cpu_seconds": 0.0})
        print(
            f"  {reason}: calls={int(record['calls'])} "
            f"wall={record['wall_seconds']:.3f}s "
            f"thread_cpu={record['thread_cpu_seconds']:.3f}s"
        )
    other_syncs = sorted(set(sync_timings) - set(sync_order))
    for reason in other_syncs:
        record = sync_timings[reason]
        print(
            f"  {reason}: calls={int(record['calls'])} "
            f"wall={record['wall_seconds']:.3f}s "
            f"thread_cpu={record['thread_cpu_seconds']:.3f}s"
        )
    def _frame_value(value):
        return value / frame_count if frame_count else value

    print("TILED CONV")
    tiled_calls = counters["tiled_conv_calls"]
    tiled_tiles = counters["tiled_conv_tiles"]
    if frame_count:
        tiled_call_rate = f"{_frame_value(tiled_calls):.2f}/frame"
        tile_draw_rate = f"{_frame_value(counters['tiled_draw_calls']):.2f}/frame"
    else:
        tiled_call_rate = "total; frame count unavailable"
        tile_draw_rate = "total; frame count unavailable"
    print(
        f"  logical calls={int(tiled_calls)} ({tiled_call_rate}) "
        f"tile draws={int(counters['tiled_draw_calls'])} "
        f"({tile_draw_rate}) "
        f"avg tiles/Conv={tiled_tiles / tiled_calls if tiled_calls else 0.0:.2f}"
    )
    print(f"  consolidation draws={int(counters['consolidation_draw_calls'])}")
    for stage_name in (
        "tile_render", "tile_render_submission", "consolidation_draw_submission",
        "post_consolidation_synchronization",
    ):
        record = conv_stage_timings.get(stage_name, {"calls": 0, "wall_seconds": 0.0})
        print(f"  {stage_name}: wall={record['wall_seconds']:.3f}s calls={int(record['calls'])}")
    for reason in ("per_tile", "pre_consolidation", "post_consolidation"):
        record = sync_timings.get(reason, {"calls": 0, "wall_seconds": 0.0})
        print(f"  {reason} wait: wall={record['wall_seconds']:.3f}s calls={int(record['calls'])}")
    print("ACTIVATION POOL")
    print(
        f"  acquires={int(counters['activation_pool_acquires'])} "
        f"reuses={int(counters['activation_pool_reuses'])} "
        f"allocations={int(counters['activation_pool_allocations'])} "
        f"releases={int(counters['activation_pool_releases'])}"
    )
    for reason in ("activation_reuse", "activation_release", "activation_eviction"):
        record = sync_timings.get(reason, {"calls": 0, "wall_seconds": 0.0})
        print(f"  {reason}: syncs={int(record['calls'])} wall={record['wall_seconds']:.3f}s")
    print("SCRATCH POOL")
    print(
        f"  acquires={int(counters['scratch_pool_acquires'])} "
        f"reuses={int(counters['scratch_pool_reuses'])} "
        f"allocations={int(counters['scratch_pool_allocations'])} "
        f"releases={int(counters['scratch_texture_releases'])} "
        f"pool releases={int(counters['scratch_pool_releases'])}"
    )
    print(
        f"  fresh allocations={int(counters['scratch_fresh_allocations'])} "
        f"fresh releases={int(counters['scratch_fresh_releases'])} "
        f"cross-epoch reuses={int(counters['scratch_cross_epoch_reuses'])} "
        f"same-epoch reuse attempts={int(counters['scratch_same_epoch_reuse_attempts'])} "
        f"reused clears={int(counters['scratch_reused_clears'])} "
        f"retired={int(counters['scratch_retired_textures'])} "
        f"retired bytes={int(counters['scratch_retired_bytes'])} "
        f"current retired={int(counters['scratch_retired_current'])} "
        f"current bytes={int(counters['scratch_retired_bytes_current'])} "
        f"reclaims={int(counters['scratch_retired_reclaims'])} "
        f"promotions={int(counters['scratch_retired_promotions'])} "
        f"safe-point evictions={int(counters['scratch_retired_evictions'])}"
    )
    print(
        f"  reusable pool textures={int(counters['scratch_pool_current'])} "
        f"bytes={int(counters['scratch_pool_bytes'])}"
    )
    for reason in ("scratch_reuse", "scratch_release", "scratch_eviction"):
        record = sync_timings.get(reason, {"calls": 0, "wall_seconds": 0.0})
        print(f"  {reason}: syncs={int(record['calls'])} wall={record['wall_seconds']:.3f}s")
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
    try:
        from .resources import activation_pool_stats
        pool = activation_pool_stats()
        print(
            f"  activation pool: {pool['current_pooled_textures']} textures "
            f"({pool['current_pooled_bytes']} bytes)"
        )
        print(
            f"  activation pool peak: {pool['peak_pooled_textures']} textures "
            f"({pool['peak_pooled_bytes']} bytes)"
        )
        print(f"  activation allocation avoidance: {pool['allocation_avoidance_rate']:.1%}")
    except RuntimeError:
        print("  activation pool: unavailable")
    print(f"  texture uploads: {int(counters['texture_uploads'])} ({int(counters['texture_upload_bytes'])} bytes, {counters['texture_upload_seconds']:.3f}s)")
    print("parameter uploads:")
    print(f"  count: {int(counters['parameter_uploads'])}")
    print(f"  bytes: {int(counters['parameter_upload_bytes'])}")
    print(f"  repeated: {int(counters['repeated_parameter_uploads'])}")
    print(f"  cache hits: {int(counters['parameter_cache_hits'])}")
    print(f"  cache misses: {int(counters['parameter_cache_misses'])}")
    print(f"  cache invalidations: {int(counters['parameter_cache_invalidations'])}")
    print(f"  cache bypasses: {int(counters['parameter_cache_bypasses'])}")
    print("parameter upload groups:")
    for group in parameter_group_snapshot():
        shape = group["tensor_shape"]
        cache_status = "cacheable" if group["cacheable"] else "bypassed"
        print(
            f"  {group['operation']} {group['parameter_role']} shape={shape} "
            f"cache={cache_status}: uploads={group['upload_count']} "
            f"repeated={group['repeated_count']} bytes={group['bytes']} "
            f"hits={group['cache_hits']} misses={group['cache_misses']} "
            f"invalidations={group['cache_invalidations']}"
        )
    try:
        from .resources import persistent_parameter_resources
        current = persistent_parameter_resources()
        print(f"  persistent parameter resources: {current}")
        print(f"  peak persistent parameter resources: {int(counters['persistent_parameter_resources_peak'])}")
    except RuntimeError:
        print("  persistent parameter resources: unavailable (runtime already cleaned up)")
        print(f"  peak persistent parameter resources: {int(counters['persistent_parameter_resources_peak'])}")
    print("readback:")
    print(f"  calls: {int(counters['readback_calls'])} bytes: {int(counters['readback_bytes'])}")
    print(f"  sync/wait: {counters['readback_sync_seconds']:.3f}s")
    print(f"  transfer: {counters['readback_transfer_seconds']:.3f}s")
    print(f"  conversion/tensor creation: {counters['readback_conversion_seconds']:.3f}s")
    print(f"  total: {counters['readback_total_seconds']:.3f}s")
    if conv:
        print("Conv2D breakdown (aggregate):")
        for key in ("prepare", "parameter_upload", "tile_render", "sync", "consolidation"):
            print(f"  {key}: {conv[key]:.3f}s")
        print(f"  tiled calls: {int(counters['tiled_conv_calls'])} tiles: {int(counters['tiled_conv_tiles'])} max physical tile: {int(counters['tiled_conv_max_tile_width'])}x{int(counters['tiled_conv_max_tile_height'])}")
    slow = sorted(ops.items(), key=lambda item: item[1]["total"], reverse=True)[:3]
    if slow:
        print("Top slow operations:")
        for index, (name, record) in enumerate(slow, 1):
            print(f"  {index}. {name}: {record['total']:.3f}s ({int(record['calls'])} calls)")
