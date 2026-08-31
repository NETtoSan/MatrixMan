"""Optional aggregate profiling for the legacy CUDA execution boundary."""

from __future__ import annotations

import atexit
import time
from collections import defaultdict

from ...config import profiling_enabled


def _enabled() -> bool:
    return profiling_enabled(legacy_cuda=True)


enabled = _enabled()
records: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "bytes": 0})
batch_norm = defaultdict(float)
conv2d = defaultdict(float)
activation = defaultdict(float)
parameter_cache = defaultdict(float)
conv2d_signatures = {}
_exit_hook_registered = False


def set_enabled(value: bool) -> None:
    """Update profiling for an explicit Python configuration change."""
    global enabled
    enabled = bool(value)
    if enabled:
        register_exit_hook()


def is_enabled() -> bool:
    """Return the CUDA profiler's current runtime state."""
    return enabled


def reset() -> None:
    """Reset aggregate CUDA profiling state for the next measurement window."""
    records.clear()
    batch_norm.clear()
    conv2d.clear()
    activation.clear()
    parameter_cache.clear()
    conv2d_signatures.clear()


def register_exit_hook() -> None:
    """Register CUDA reporting only after CUDA has been selected."""
    global _exit_hook_registered
    if _exit_hook_registered:
        return

    def report_if_used() -> None:
        from ...backend import active_backend

        active = active_backend()
        if (
            active is not None
            and active.name == "cuda"
            and enabled
            and (records or batch_norm or conv2d or activation or parameter_cache)
        ):
            report()

    atexit.register(report_if_used)
    _exit_hook_registered = True


def start() -> float | None:
    return time.perf_counter() if enabled else None


def observe(label: str, started: float | None, byte_count: int = 0) -> float | None:
    if not enabled or started is None:
        return None
    elapsed = time.perf_counter() - started
    record = records[label]
    record["calls"] += 1
    record["seconds"] += elapsed
    record["bytes"] += int(byte_count)
    return elapsed


def count(label: str, byte_count: int = 0) -> None:
    if enabled:
        record = records[label]
        record["calls"] += 1
        record["bytes"] += int(byte_count)


def count_batch_norm(name: str, byte_count: int = 0) -> None:
    if enabled:
        batch_norm[name] += 1
        if byte_count:
            batch_norm[f"{name}_bytes"] += int(byte_count)


def count_conv2d(name: str, byte_count: int = 0) -> None:
    if enabled:
        conv2d[name] += 1
        if byte_count:
            conv2d[f"{name}_bytes"] += int(byte_count)


def count_activation(name: str, byte_count: int = 0) -> None:
    if enabled:
        activation[name] += 1
        if byte_count:
            activation[f"{name}_bytes"] += int(byte_count)


def parameter_cache_event(name: str, byte_count: int = 0) -> None:
    if enabled:
        parameter_cache[name] += 1
        if byte_count:
            parameter_cache[f"{name}_bytes"] += int(byte_count)


def parameter_cache_adjust(name: str, delta: int) -> None:
    if enabled:
        parameter_cache[name] += int(delta)


def observe_conv2d_signature(
    signature: tuple[int, ...], elapsed: float | None, variant: str = "generic"
) -> None:
    if not enabled or elapsed is None:
        return
    record = conv2d_signatures.setdefault(
        (variant, signature), {"calls": 0, "seconds": 0.0}
    )
    record["calls"] += 1
    record["seconds"] += elapsed


def _format_conv2d_signature(signature: tuple[int, ...]) -> str:
    (
        n, cin, hin, win, cout, hout, wout,
        kh, kw, stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups,
    ) = signature
    return (
        f"[{n},{cin},{hin},{win}] -> [{n},{cout},{hout},{wout}] "
        f"k={kh}x{kw} s={stride_h}x{stride_w} "
        f"p={pad_h}x{pad_w} d={dilation_h}x{dilation_w} g={groups}"
    )


def report() -> None:
    if not enabled:
        return
    print("\nMatrixMan CUDA profile")
    print("----------------------------------------")
    for label, record in records.items():
        calls = int(record["calls"])
        total_ms = record["seconds"] * 1000.0
        average_ms = total_ms / calls if calls else 0.0
        suffix = f" bytes={int(record['bytes'])}" if record["bytes"] else ""
        print(f"{label:<24} calls={calls:<4} total={total_ms:9.3f} ms avg={average_ms:8.3f} ms{suffix}")
    if batch_norm:
        print("BatchNorm parameter traffic")
        print(f"  invocations: {int(batch_norm['invocations'])}")
        print(f"  parameter uploads: {int(batch_norm['parameter_uploads'])}")
        print(f"  parameter upload bytes: {int(batch_norm['parameter_uploads_bytes'])}")
        print(f"  temporary allocations: {int(batch_norm['temporary_allocations'])}")
        print(f"  temporary frees: {int(batch_norm['temporary_frees'])}")
    if conv2d:
        print("Conv2D parameter traffic")
        print(f"  weight uploads: {int(conv2d['weight_uploads'])}")
        print(f"  weight bytes: {int(conv2d['weight_uploads_bytes'])}")
        print(f"  bias uploads: {int(conv2d['bias_uploads'])}")
        print(f"  bias bytes: {int(conv2d['bias_uploads_bytes'])}")
        print(f"  parameter temporary allocations: {int(conv2d['temporary_allocations'])}")
        print(f"  parameter temporary frees: {int(conv2d['temporary_frees'])}")
        print(f"  output allocations: {int(conv2d['output_allocations'])}")
        print(f"  output allocation bytes: {int(conv2d['output_allocations_bytes'])}")

    if activation:
        print("Activation upload traffic")
        print(f"  uploads: {int(activation['uploads'])}")
        print(f"  upload bytes: {int(activation['uploads_bytes'])}")

    if parameter_cache:
        print("Parameter cache")
        print(f"  hits: {int(parameter_cache['hits'])}")
        print(f"  misses: {int(parameter_cache['misses'])}")
        print(f"  retained allocations: {int(parameter_cache['retained_allocations'])}")
        print(f"  retained bytes: {int(parameter_cache['retained_bytes'])}")

    h2d = records.get("HtoD", {"calls": 0, "bytes": 0})
    attributed_calls = int(batch_norm["parameter_uploads"] + conv2d["weight_uploads"] + conv2d["bias_uploads"])
    attributed_bytes = int(batch_norm["parameter_uploads_bytes"] + conv2d["weight_uploads_bytes"] + conv2d["bias_uploads_bytes"])
    print("HtoD attribution")
    print(f"  attributed parameter calls: {attributed_calls}")
    print(f"  attributed parameter bytes: {attributed_bytes}")
    print(f"  other HtoD calls: {max(0, int(h2d['calls']) - attributed_calls)}")
    print(f"  other HtoD bytes: {max(0, int(h2d['bytes']) - attributed_bytes)}")

    if conv2d_signatures:
        total_conv_seconds = records.get("Conv2D", {}).get("seconds", 0.0)
        print("Conv2D profile by signature")
        print("------------------------------------------------------------------------")
        for (variant, signature), record in sorted(
            conv2d_signatures.items(), key=lambda item: item[1]["seconds"], reverse=True
        ):
            calls = int(record["calls"])
            seconds = record["seconds"]
            total_ms = seconds * 1000.0
            average_ms = total_ms / calls if calls else 0.0
            percent = (seconds / total_conv_seconds * 100.0) if total_conv_seconds else 0.0
            n, cin, hin, win, cout, hout, wout, kh, kw = signature[:9]
            groups = signature[-1]
            macs = n * cout * hout * wout * (cin // groups) * kh * kw
            gmacs = macs / (seconds * 1e9) if seconds else 0.0
            print(f"{_format_conv2d_signature(signature)} variant={variant}")
            print(
                f"    calls={calls} total={total_ms:.3f} ms "
                f"({percent:.1f}%) avg={average_ms:.3f} ms "
                f"output_elements={n * cout * hout * wout} "
                f"MACs={macs} GMAC/s={gmacs:.3f}"
            )
