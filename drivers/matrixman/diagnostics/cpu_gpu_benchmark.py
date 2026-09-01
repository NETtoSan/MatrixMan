#!/usr/bin/env python3
"""
Backend-neutral MatrixMan versus CPU matrix multiplication benchmark.

This compares the same operation on the same float32 inputs:

    C = A x B

CPU path:
    NumPy A @ B, used as both benchmark and correctness reference.

MatrixMan path:
    CPU matrices -> selected MatrixMan backend -> backend-aware readback.

The GPU operation under test does not fall back to CPU arithmetic.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch

from drivers import matrixman
from drivers.matrixman.backend import get_backend
from drivers.matrixman.diagnostics.backend_helpers import readback_tensor


gm = None
gpu_stress = None


def _load_opengl_benchmark_modules() -> None:
    """Load the legacy OpenGL stress implementation after backend gating."""
    global gm, gpu_stress
    if gm is not None:
        return
    from drivers.matrixman import gpumatrix as gm_module
    from drivers.matrixman import gpu_stress as stress_module

    gm = gm_module
    gpu_stress = stress_module


@dataclass
class BenchResult:
    label: str
    size: int
    avg_seconds: float
    matmuls_per_sec: float
    gflops: float
    max_abs_error: float | None = None
    matches: bool | None = None
    error: str | None = None
    phases: dict[str, float] | None = None
    phase_calls: dict[str, int] | None = None


def make_benchmark_inputs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Scale by sqrt(N) so output magnitudes stay comparable across sizes.
    scale = 1.0 / math.sqrt(n)
    a = rng.uniform(-scale, scale, (n, n)).astype(np.float32)
    b = rng.uniform(-scale, scale, (n, n)).astype(np.float32)
    return a, b


def flops_per_matmul(n: int) -> float:
    return 2.0 * n * n * n


def result_from_total(label: str, n: int, total_seconds: float, count: int, **kwargs) -> BenchResult:
    avg = total_seconds / count
    rate = count / total_seconds
    return BenchResult(label, n, avg, rate, rate * flops_per_matmul(n) / 1e9, **kwargs)


def benchmark_cpu(a: np.ndarray, b: np.ndarray, min_seconds: float) -> tuple[np.ndarray, BenchResult]:
    n = a.shape[0]
    # Warm up BLAS/cache behavior before measurement.
    reference = a @ b

    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < min_seconds:
        reference = a @ b
        count += 1
    elapsed = time.perf_counter() - started
    return reference, result_from_total("CPU NumPy", n, elapsed, max(count, 1))


def _synchronize() -> None:
    """Complete selected-backend work before wall-clock measurements/readback."""
    get_backend().synchronize()


def _matrixman_matmul(a_cpu: np.ndarray, b_cpu: np.ndarray):
    a = matrixman.to_device(torch.from_numpy(a_cpu))
    b = matrixman.to_device(torch.from_numpy(b_cpu))
    return a @ b


def benchmark_matrixman_compute_only(
    a_cpu: np.ndarray, b_cpu: np.ndarray, n: int, min_seconds: float
) -> BenchResult:
    """Measure selected-backend matmul with inputs already resident."""
    a = matrixman.to_device(torch.from_numpy(a_cpu))
    b = matrixman.to_device(torch.from_numpy(b_cpu))
    _synchronize()
    _ = a @ b
    _synchronize()
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < min_seconds:
        _ = a @ b
        _synchronize()
        count += 1
    elapsed = time.perf_counter() - started
    return result_from_total(f"MatrixMan/{get_backend().name} compute only", n, elapsed, max(count, 1))


def benchmark_matrixman_with_transfers(
    a_cpu: np.ndarray, b_cpu: np.ndarray, n: int, min_seconds: float
) -> BenchResult:
    """Measure uploads, selected-backend matmul, and explicit CPU readback."""
    count = 0
    phase_names = ("allocation/setup + HtoD", "matmul dispatch", "backend synchronization", "DtoH readback", "python reference release")
    phases = {name: 0.0 for name in phase_names}
    phase_calls = {name: 0 for name in phase_names}
    cuda_profile = None
    profile_before = None
    if get_backend().name == "cuda":
        from drivers.matrixman.backends.cuda import profiling as cuda_profile_module

        if cuda_profile_module.is_enabled():
            cuda_profile = cuda_profile_module
            profile_before = {
                label: dict(record) for label, record in cuda_profile.records.items()
            }
    started = time.perf_counter()
    while time.perf_counter() - started < min_seconds:
        phase_start = time.perf_counter()
        left = matrixman.to_device(torch.from_numpy(a_cpu))
        right = matrixman.to_device(torch.from_numpy(b_cpu))
        phases["allocation/setup + HtoD"] += time.perf_counter() - phase_start
        phase_calls["allocation/setup + HtoD"] += 1

        phase_start = time.perf_counter()
        result = left @ right
        phases["matmul dispatch"] += time.perf_counter() - phase_start
        phase_calls["matmul dispatch"] += 1

        phase_start = time.perf_counter()
        _synchronize()
        phases["backend synchronization"] += time.perf_counter() - phase_start
        phase_calls["backend synchronization"] += 1

        phase_start = time.perf_counter()
        _ = readback_tensor(result)
        phases["DtoH readback"] += time.perf_counter() - phase_start
        phase_calls["DtoH readback"] += 1

        phase_start = time.perf_counter()
        del result, left, right
        release_elapsed = time.perf_counter() - phase_start
        phases["python reference release"] += release_elapsed
        phase_calls["python reference release"] += 1
        count += 1
    elapsed = time.perf_counter() - started
    if cuda_profile is not None and profile_before is not None:
        for label in ("Alloc", "HtoD", "Free", "DtoH"):
            before = profile_before.get(label, {})
            after = cuda_profile.records.get(label, {})
            phases[f"CUDA profiler {label}"] = after.get("seconds", 0.0) - before.get("seconds", 0.0)
            phase_calls[f"CUDA profiler {label}"] = int(after.get("calls", 0) - before.get("calls", 0))
    return result_from_total(f"MatrixMan/{get_backend().name} incl upload/readback", n, elapsed, max(count, 1), phases=phases, phase_calls=phase_calls)


def print_transfer_breakdown(n: int, result: BenchResult) -> None:
    if not result.phases:
        return
    print(f"\nTransfer-phase breakdown ({result.label}, {n}x{n})")
    print("  phase                                      total ms     avg ms")
    print("  " + "-" * 62)
    for name, seconds in result.phases.items():
        if name.endswith(" calls"):
            continue
        calls = (result.phase_calls or {}).get(name, 0)
        average = seconds / calls if calls else seconds
        print(f"  {name:<40} {seconds * 1000.0:10.3f} {average * 1000.0:10.3f}")
    h2d_seconds = result.phases.get("CUDA profiler HtoD")
    dtoh_seconds = result.phases.get("CUDA profiler DtoH")
    if h2d_seconds and dtoh_seconds is not None:
        h2d_calls = (result.phase_calls or {}).get("CUDA profiler HtoD", 0)
        dtoh_calls = (result.phase_calls or {}).get("CUDA profiler DtoH", 0)
        h2d_bytes = 2 * n * n * 4 * max(1, h2d_calls // 2)
        dtoh_bytes = n * n * 4 * max(1, dtoh_calls)
        print(f"  HtoD bytes={h2d_bytes} bandwidth={h2d_bytes / h2d_seconds / 1048576.0:.2f} MiB/s")
        print(f"  DtoH bytes={dtoh_bytes} bandwidth={dtoh_bytes / dtoh_seconds / 1048576.0:.2f} MiB/s")
    else:
        print("  Exact CUDA copy timing: enable MATRIXMAN_PROFILE=1 for Driver API attribution")


def verify_matrixman(a_cpu: np.ndarray, b_cpu: np.ndarray, reference: np.ndarray) -> tuple[float, bool]:
    result = _matrixman_matmul(a_cpu, b_cpu)
    _synchronize()
    result_cpu = readback_tensor(result).numpy()
    max_error = float(np.max(np.abs(result_cpu - reference)))
    matches = bool(np.allclose(result_cpu, reference, rtol=5e-4, atol=5e-4))
    return max_error, matches


def print_comparison_table(results: list[BenchResult]) -> None:
    print("\nCPU vs GPU matrix multiplication")
    print(
        f"{'size':>8}  {'path':<26} {'avg ms':>10} {'matmul/s':>10} "
        f"{'GFLOPS':>10} {'max err':>12} {'valid':>7}"
    )
    print("-" * 91)
    for r in results:
        if r.error:
            print(f"{r.size:>8}  {r.label:<26} {'FAILED':>10} {'-':>10} {'-':>10} {r.error}")
            continue
        err = "-" if r.max_abs_error is None else f"{r.max_abs_error:.3g}"
        valid = "-" if r.matches is None else str(r.matches)
        print(
            f"{r.size:>8}  {r.label:<26} {r.avg_seconds * 1000.0:10.3f} "
            f"{r.matmuls_per_sec:10.3f} {r.gflops:10.3f} {err:>12} {valid:>7}"
        )

    print("\nFaster path by size:")
    for n in sorted({r.size for r in results}):
        cpu = next((r for r in results if r.size == n and r.label == "CPU NumPy" and not r.error), None)
        gpu = next((r for r in results if r.size == n and "compute only" in r.label and not r.error), None)
        gpu_total = next((r for r in results if r.size == n and "incl upload/readback" in r.label and not r.error), None)
        if cpu and gpu:
            winner = "CPU" if cpu.avg_seconds < gpu.avg_seconds else gpu.label
            factor = max(cpu.avg_seconds, gpu.avg_seconds) / min(cpu.avg_seconds, gpu.avg_seconds)
            print(f"  {n}x{n}: {winner} is {factor:.2f}x faster than the other compute-only path")
        if cpu and gpu_total:
            winner = "CPU" if cpu.avg_seconds < gpu_total.avg_seconds else gpu_total.label
            factor = max(cpu.avg_seconds, gpu_total.avg_seconds) / min(cpu.avg_seconds, gpu_total.avg_seconds)
            print(f"  {n}x{n}: {winner} is {factor:.2f}x faster when GPU transfers/readback are included")


def run_size(n: int, seconds_per_bench: float, seed: int) -> list[BenchResult]:
    print(f"\nBenchmarking {n}x{n}")
    a, b = make_benchmark_inputs(n, seed)
    reference, cpu = benchmark_cpu(a, b, seconds_per_bench)

    try:
        max_error, matches = verify_matrixman(a, b, reference)
        gpu_compute = benchmark_matrixman_compute_only(a, b, n, seconds_per_bench)
        gpu_total = benchmark_matrixman_with_transfers(a, b, n, seconds_per_bench)
        gpu_compute.max_abs_error = max_error
        gpu_compute.matches = matches
        gpu_total.max_abs_error = max_error
        gpu_total.matches = matches
        return [cpu, gpu_compute, gpu_total]
    except Exception as exc:
        return [cpu, BenchResult(f"MatrixMan/{get_backend().name} compute only", n, math.nan, math.nan, math.nan, error=str(exc))]


def run_512_stress(seconds: float) -> None:
    print(f"\nStarting {seconds:.0f}-second GPU-only stress test at 512x512 without fallback")
    try:
        state = gpu_stress.StressState(512)
    except Exception as exc:
        print(f"512x512 stress setup failed exactly: {exc}")
        return

    started = time.monotonic()
    deadline = started + seconds
    next_report = started + 1.0
    next_validate = started + 5.0
    iterations = 0
    matmuls = 0
    shader_ops = 0
    max_error = 0.0
    last_validation = "not yet checked"
    gl_errors_seen = False
    one_second_gflops: list[float] = []
    last_matmuls = 0
    last_time = started

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                break

            if now >= next_validate:
                err, valid = state.validate_next_iteration()
                max_error = max(max_error, err)
                last_validation = f"valid={valid}, max_abs_error={err:.6g}"
                next_validate += 5.0
            else:
                state.iteration()
                gm.glFinish()

            iterations += 1
            matmuls += 2
            shader_ops += 4

            now = time.monotonic()
            if now >= next_report:
                gl_error = state.gl_error()
                if gl_error:
                    gl_errors_seen = True
                    gl_error_text = f"0x{gl_error:04x}"
                else:
                    gl_error_text = "none"
                interval = max(now - last_time, 1e-9)
                interval_matmuls = matmuls - last_matmuls
                interval_gflops = (interval_matmuls / interval) * flops_per_matmul(512) / 1e9
                one_second_gflops.append(interval_gflops)
                elapsed = now - started
                print(
                    f"t={elapsed:6.2f}s size=512 ops={shader_ops} "
                    f"matmul/s={matmuls / elapsed:7.3f} "
                    f"est_GFLOPS={matmuls / elapsed * flops_per_matmul(512) / 1e9:7.3f} "
                    f"correctness={last_validation} gl_errors={gl_error_text}",
                    flush=True,
                )
                last_time = now
                last_matmuls = matmuls
                next_report += 1.0

        gm.glFinish()
        runtime = time.monotonic() - started
        avg_matmuls = matmuls / runtime
        avg_gflops = avg_matmuls * flops_per_matmul(512) / 1e9
        if len(one_second_gflops) >= 2:
            first = statistics.mean(one_second_gflops[: min(10, len(one_second_gflops))])
            last = statistics.mean(one_second_gflops[-min(10, len(one_second_gflops)) :])
            change = ((last - first) / first * 100.0) if first else 0.0
            perf_note = f"{change:+.1f}% last-window vs first-window GFLOPS"
        else:
            perf_note = "not enough samples"

        print("\n512x512 GPU-only stress summary:")
        print(f"  total runtime: {runtime:.3f}s")
        print(f"  total matmuls: {matmuls}")
        print(f"  total shader operations: {shader_ops}")
        print(f"  average matmuls/sec: {avg_matmuls:.3f}")
        print(f"  estimated average GFLOPS: {avg_gflops:.6f}")
        print(f"  maximum observed numerical error: {max_error:.6g}")
        print(f"  last correctness status: {last_validation}")
        print(f"  any OpenGL errors: {gl_errors_seen}")
        print(f"  performance change: {perf_note}")
    except Exception as exc:
        print(f"512x512 stress failed exactly: {exc}")
    finally:
        state.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CPU vs selected MatrixMan backend matrix multiplication benchmark"
    )
    parser.add_argument("--seconds-per-bench", type=float, default=3.0)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--stress-seconds", type=float, default=60.0)
    args = parser.parse_args()

    selected = get_backend()
    print(f"Selected backend: {selected.name.upper()}")
    print("Benchmark core: MatrixMan frontend matmul with explicit backend synchronization")

    results: list[BenchResult] = []
    for i, n in enumerate([256, 512], start=1):
        results.extend(run_size(n, args.seconds_per_bench, seed=100 + i))
    print_comparison_table(results)
    for result in results:
        if result.phases:
            print_transfer_breakdown(result.size, result)

    if not args.skip_stress:
        if selected.name == "opengl":
            _load_opengl_benchmark_modules()
            gpu_stress.print_capabilities()
            gpu_stress.print_telemetry_availability()
            run_512_stress(args.stress_seconds)
        else:
            print("Skipping legacy OpenGL stress: it is not a selected-backend benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
