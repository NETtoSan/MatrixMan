"""Low-overhead, opt-in profiling for the PyTorch-facing MatrixMan path."""

from __future__ import annotations

import atexit
import functools
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


# MATRIXMAN_PROFILE remains the main profiling switch.  The dedicated
# variable allows frontend-only runs when backend profiling is not desired.
enabled = _truthy(os.environ.get("MATRIXMAN_PROFILE")) or _truthy(
    os.environ.get("MATRIXMAN_PROFILE_DISPATCH")
)
started = time.perf_counter()
ops: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "wall": 0.0, "cpu": 0.0}
)
stages: dict[str, dict[str, float]] = defaultdict(
    lambda: {"calls": 0, "wall": 0.0, "cpu": 0.0, "exclusive_wall": 0.0, "exclusive_cpu": 0.0}
)
exclusive_by_op: dict[str, dict[str, dict[str, float]]] = defaultdict(
    lambda: defaultdict(lambda: {"calls": 0, "wall": 0.0, "cpu": 0.0})
)
counters = defaultdict(int)
_current_category = None
_current_operator = None
_scope_stack = []
_last_report_dispatches = 0

_METADATA = {
    "aten.view.default", "aten.reshape.default", "aten.flatten.using_ints",
    "aten.squeeze.default", "aten.squeeze.dim", "aten.unsqueeze.default",
    "aten.expand.default", "aten.transpose.int", "aten.split.Tensor",
    "aten.detach.default", "aten.alias.default",
}
_FACTORIES = {
    "aten.new_full.default", "aten.arange.out", "aten.fill_.Scalar",
}
_TRANSFERS = {"aten._to_copy.default"}
_BOOKKEEPING = {"aten.detach.default", "aten.alias.default"}


def _name(func) -> str:
    return str(func)


def _category(name: str) -> str:
    if name in _METADATA:
        return "metadata/view ops"
    if name in _FACTORIES:
        return "factory ops"
    if name in _TRANSFERS:
        return "explicit transfer/readback ops"
    if name in _BOOKKEEPING:
        return "bookkeeping ops"
    return "compute ops"


def _record_stage(name: str, wall: float, cpu: float) -> None:
    item = stages[name]
    item["calls"] += 1
    item["wall"] += wall
    item["cpu"] += cpu


@contextmanager
def component(name: str):
    """Record inclusive and exclusive calling-thread time for nested scopes."""
    if not enabled:
        yield
        return
    wall_start = time.perf_counter()
    cpu_start = time.thread_time()
    scope = {"name": name, "child_wall": 0.0, "child_cpu": 0.0}
    _scope_stack.append(scope)
    try:
        yield
    finally:
        _scope_stack.pop()
        wall = time.perf_counter() - wall_start
        cpu = time.thread_time() - cpu_start
        exclusive_wall = max(0.0, wall - scope["child_wall"])
        exclusive_cpu = max(0.0, cpu - scope["child_cpu"])
        record = stages[name]
        record["calls"] += 1
        record["wall"] += wall
        record["cpu"] += cpu
        record["exclusive_wall"] += exclusive_wall
        record["exclusive_cpu"] += exclusive_cpu
        if _current_operator is not None:
            op_record = exclusive_by_op[_current_operator][name]
            op_record["calls"] += 1
            op_record["wall"] += exclusive_wall
            op_record["cpu"] += exclusive_cpu
        if _scope_stack:
            _scope_stack[-1]["child_wall"] += wall
            _scope_stack[-1]["child_cpu"] += cpu


def method_timer(name: str):
    """Decorate a backend method without adding a disabled-path wrapper."""
    def decorate(fn):
        if not enabled:
            return fn
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with component(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def record_external(name: str, wall: float, cpu: float) -> None:
    """Record a non-nested backend timer as an exclusive component."""
    if not enabled:
        return
    record = stages[name]
    record["calls"] += 1
    record["wall"] += wall
    record["cpu"] += cpu
    record["exclusive_wall"] += wall
    record["exclusive_cpu"] += cpu
    if _current_operator is not None:
        op_record = exclusive_by_op[_current_operator][name]
        op_record["calls"] += 1
        op_record["wall"] += wall
        op_record["cpu"] += cpu
    if _scope_stack:
        _scope_stack[-1]["child_wall"] += wall
        _scope_stack[-1]["child_cpu"] += cpu


def dispatch_call(func, callback):
    """Time one complete MatrixManTensor.__torch_dispatch__ callback."""
    name = _name(func)
    wall_start = time.perf_counter()
    cpu_start = time.thread_time()
    global _current_category, _current_operator
    previous_category = _current_category
    previous_operator = _current_operator
    _current_category = _category(name)
    _current_operator = name
    try:
        with component("torch_dispatch callback"):
            return callback()
    finally:
        _current_category = previous_category
        _current_operator = previous_operator
        wall = time.perf_counter() - wall_start
        cpu = time.thread_time() - cpu_start
        item = ops[name]
        item["calls"] += 1
        item["wall"] += wall
        item["cpu"] += cpu
        counters[f"category:{_category(name)}"] += 1


def wrapper_created(metadata: bool | None = None) -> None:
    counters["matrixman_tensor_objects_created"] += 1
    if metadata is True or (metadata is None and _current_category == "metadata/view ops"):
        counters["metadata_views_created"] += 1


def wrapper_timing(wall: float, cpu: float) -> None:
    record_external("tensor wrapper creation", wall, cpu)


def metadata_call(callback, *args, **kwargs):
    with component("metadata/view handling"):
        return callback(*args, **kwargs)


def wrap_shared(fn):
    """Add a shared-dispatch timing layer, only when enabled at import time."""
    if not enabled:
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with component("shared MatrixMan dispatch"):
            return fn(*args, **kwargs)
    return wrapped


def wrap_backend(fn):
    """Time a backend dispatch bridge (CUDA's bridge is shared with routing)."""
    if not enabled:
        return fn

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with component("backend dispatch"):
            return fn(*args, **kwargs)
    return wrapped


def wrap_cuda_backend(fn):
    """Time CUDA's backend portion, which currently shares the routing module."""
    if not enabled:
        return fn
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        from .backend import active_backend
        active = active_backend()
        if active is not None and active.name == "cuda":
            with component("CUDA backend dispatch/selection"):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapped


def reset() -> None:
    global _last_report_dispatches
    started_reset = time.perf_counter()
    globals()["started"] = started_reset
    ops.clear()
    stages.clear()
    counters.clear()
    exclusive_by_op.clear()
    _scope_stack.clear()
    _last_report_dispatches = 0


def _print_table(title, records, frame_count=None):
    print(title)
    for name, item in sorted(records.items(), key=lambda pair: (-pair[1]["cpu"], pair[0])):
        calls = int(item["calls"])
        if not calls:
            continue
        exclusive = item.get("exclusive_cpu", item["cpu"])
        suffix = (
            f" calls/frame={calls / frame_count:.2f}"
            f" exclusive_cpu_ms/frame={exclusive / frame_count * 1000.0:.3f}"
            if frame_count else ""
        )
        print(f"  {name}: calls={calls}{suffix} avg_us/call={exclusive / calls * 1e6:.2f} "
              f"exclusive_cpu={exclusive:.6f}s wall={item['wall']:.6f}s")


def _frontend_component_records():
    """Aggregate exclusive component records that occurred inside dispatch."""
    records = defaultdict(lambda: {"calls": 0, "wall": 0.0, "cpu": 0.0})
    for components in exclusive_by_op.values():
        for name, value in components.items():
            record = records[name]
            record["calls"] += value["calls"]
            record["wall"] += value["wall"]
            record["cpu"] += value["cpu"]
    return records


def report(frame_count: int | None = None) -> None:
    global _last_report_dispatches
    if not enabled:
        return
    _last_report_dispatches = sum(int(v["calls"]) for v in ops.values())
    print("\nMatrixMan frontend dispatch profile\n-----------------------------------")
    print("Timing uses the calling Python thread; backend GPU execution is not included.")
    print(f"dispatch callbacks: {sum(int(v['calls']) for v in ops.values())}")
    print(f"MatrixManTensor objects created: {counters['matrixman_tensor_objects_created']}")
    print(f"metadata-only views created: {counters['metadata_views_created']}")
    _print_table("ATen operators:", ops, frame_count)
    print("Dispatch categories:")
    for category in ("compute ops", "metadata/view ops", "factory ops",
                     "explicit transfer/readback ops", "bookkeeping ops"):
        print(f"  {category}: {counters[f'category:{category}']}")
    frontend_components = _frontend_component_records()
    _print_table("Frontend components inside dispatch (exclusive CPU):", frontend_components, frame_count)
    total = sum(item["cpu"] for item in ops.values())
    if total:
        print(f"total measured frontend CPU time: {total:.6f}s")
        exclusive_total = sum(item["cpu"] for item in frontend_components.values())
        print(f"total measured exclusive component CPU time: {exclusive_total:.6f}s")
        print("Exclusive component CPU share:")
        for name, item in sorted(frontend_components.items(), key=lambda pair: -pair[1]["cpu"]):
            print(f"  {name}: {item['cpu'] / exclusive_total * 100.0:.2f}%")
        print("Per-operator exclusive frontend CPU:")
        for op, components in sorted(exclusive_by_op.items(), key=lambda pair: -sum(v["cpu"] for v in pair[1].values())):
            op_total = sum(value["cpu"] for value in components.values())
            print(f"  {op}: total={op_total:.6f}s calls={int(ops[op]['calls'])}")
            for name, value in sorted(components.items(), key=lambda pair: -pair[1]["cpu"]):
                print(f"    {name}: {value['cpu']:.6f}s")
    print("Exclusive scopes are non-overlapping by subtracting nested child scopes; uninstrumented work remains attributed to its parent scope.")


def _report_at_exit() -> None:
    if enabled and ops and sum(int(v["calls"]) for v in ops.values()) > _last_report_dispatches:
        report()


if enabled:
    atexit.register(_report_at_exit)
