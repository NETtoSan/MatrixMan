#!/usr/bin/env python3
"""
OpenGL-only fragment-shader matrix multiplication benchmark.

This compares the same operation on the same float32 inputs:

    C = A x B

CPU path:
    NumPy A @ B, used as both benchmark and correctness reference.

OpenGL path:
    CPU matrices -> OpenGL RGBA32F textures -> GLSL 1.20 fragment shader
    -> framebuffer texture -> optional glReadPixels validation.

The GPU operation under test does not fall back to CPU arithmetic.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import statistics
import time
from dataclasses import dataclass

import numpy as np

from drivers.matrixman.backend import get_backend


gm = None
gpu_stress = None


def _load_opengl_benchmark_modules() -> None:
    """Load the legacy OpenGL benchmark implementation after backend gating."""
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


class GpuMatmulBench:
    """Reusable single-matmul OpenGL GLSL benchmark state for one matrix size."""

    def __init__(self, n: int, a: np.ndarray, b: np.ndarray):
        self.n = n
        self.program = gm.make_program(gpu_stress.shader_source(gpu_stress.MUL_SHADER, n))
        self.left_loc = gm.glGetUniformLocation(self.program, b"left_tex")
        self.right_loc = gm.glGetUniformLocation(self.program, b"right_tex")
        self.fbo = ctypes.c_uint()
        gm.glGenFramebuffers(1, ctypes.byref(self.fbo))

        self.tex_a = gpu_stress.create_texture(n, a)
        self.tex_b = gpu_stress.create_texture(n, b)
        self.tex_out = gpu_stress.create_texture(n)
        self.textures = [self.tex_a, self.tex_b, self.tex_out]

        gm.glViewport(0, 0, n, n)
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, self.fbo.value)
        gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, self.tex_out, 0)
        status = gm.glCheckFramebufferStatus(gm.GL_FRAMEBUFFER)
        if status != gm.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"framebuffer incomplete for {n}x{n}: 0x{status:04x}")

    def upload_inputs(self, a: np.ndarray, b: np.ndarray) -> None:
        gpu_stress.update_texture(self.tex_a, a)
        gpu_stress.update_texture(self.tex_b, b)

    def dispatch(self) -> None:
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, self.fbo.value)
        gm.glFramebufferTexture2D(gm.GL_FRAMEBUFFER, gm.GL_COLOR_ATTACHMENT0, gm.GL_TEXTURE_2D, self.tex_out, 0)
        gm.glUseProgram(self.program)

        gm.glActiveTexture(gm.GL_TEXTURE0)
        gm.glBindTexture(gm.GL_TEXTURE_2D, self.tex_a)
        gm.glUniform1i(self.left_loc, 0)

        gm.glActiveTexture(gm.GL_TEXTURE1)
        gm.glBindTexture(gm.GL_TEXTURE_2D, self.tex_b)
        gm.glUniform1i(self.right_loc, 1)

        gm.glBegin(gm.GL_QUADS)
        gm.glVertex2f(-1.0, -1.0)
        gm.glVertex2f(1.0, -1.0)
        gm.glVertex2f(1.0, 1.0)
        gm.glVertex2f(-1.0, 1.0)
        gm.glEnd()

    def read_result(self) -> np.ndarray:
        return gpu_stress.read_texture(self.tex_out, self.fbo, self.n)

    def cleanup(self) -> None:
        gm.glUseProgram(0)
        gm.glBindFramebuffer(gm.GL_FRAMEBUFFER, 0)
        textures = (ctypes.c_uint * len(self.textures))(*self.textures)
        gm.glDeleteTextures(len(self.textures), textures)
        gm.glDeleteFramebuffers(1, ctypes.byref(self.fbo))
        gm.glDeleteProgram(self.program)


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


def benchmark_gpu_compute_only(bench: GpuMatmulBench, n: int, min_seconds: float) -> BenchResult:
    # Inputs are already uploaded. Measure only draw + shader execution + finish.
    bench.dispatch()
    gm.glFinish()

    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < min_seconds:
        bench.dispatch()
        gm.glFinish()
        count += 1
    elapsed = time.perf_counter() - started
    return result_from_total("OpenGL shader only", n, elapsed, max(count, 1))


def benchmark_gpu_with_transfers(
    bench: GpuMatmulBench,
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    min_seconds: float,
) -> BenchResult:
    # Reuse allocations, but include input upload and output readback every pass.
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < min_seconds:
        bench.upload_inputs(a, b)
        bench.dispatch()
        gm.glFinish()
        _ = bench.read_result()
        count += 1
    elapsed = time.perf_counter() - started
    return result_from_total("OpenGL incl upload/readback", n, elapsed, max(count, 1))


def verify_gpu(bench: GpuMatmulBench, reference: np.ndarray) -> tuple[float, bool]:
    bench.dispatch()
    gm.glFinish()
    result = bench.read_result()
    max_error = float(np.max(np.abs(result - reference)))
    matches = bool(np.allclose(result, reference, rtol=5e-4, atol=5e-4))
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
        gpu = next((r for r in results if r.size == n and r.label == "OpenGL shader only" and not r.error), None)
        gpu_total = next((r for r in results if r.size == n and r.label == "OpenGL incl upload/readback" and not r.error), None)
        if cpu and gpu:
            winner = "CPU" if cpu.avg_seconds < gpu.avg_seconds else "OpenGL shader only"
            factor = max(cpu.avg_seconds, gpu.avg_seconds) / min(cpu.avg_seconds, gpu.avg_seconds)
            print(f"  {n}x{n}: {winner} is {factor:.2f}x faster than the other compute-only path")
        if cpu and gpu_total:
            winner = "CPU" if cpu.avg_seconds < gpu_total.avg_seconds else "OpenGL total"
            factor = max(cpu.avg_seconds, gpu_total.avg_seconds) / min(cpu.avg_seconds, gpu_total.avg_seconds)
            print(f"  {n}x{n}: {winner} is {factor:.2f}x faster when GPU transfers/readback are included")


def run_size(n: int, seconds_per_bench: float, seed: int) -> list[BenchResult]:
    print(f"\nBenchmarking {n}x{n}")
    a, b = make_benchmark_inputs(n, seed)
    reference, cpu = benchmark_cpu(a, b, seconds_per_bench)

    bench = None
    try:
        bench = GpuMatmulBench(n, a, b)
        max_error, matches = verify_gpu(bench, reference)
        gpu_compute = benchmark_gpu_compute_only(bench, n, seconds_per_bench)
        gpu_total = benchmark_gpu_with_transfers(bench, a, b, n, seconds_per_bench)
        gpu_compute.max_abs_error = max_error
        gpu_compute.matches = matches
        gpu_total.max_abs_error = max_error
        gpu_total.matches = matches
        gl_error = gm.glGetError()
        if gl_error:
            gpu_compute.error = f"OpenGL error after benchmark: 0x{gl_error:04x}"
        return [cpu, gpu_compute, gpu_total]
    except Exception as exc:
        return [cpu, BenchResult("OpenGL shader only", n, math.nan, math.nan, math.nan, error=str(exc))]
    finally:
        if bench is not None:
            bench.cleanup()


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
    parser = argparse.ArgumentParser(description="CPU vs OpenGL GLSL matmul benchmark (OpenGL only)")
    parser.add_argument("--seconds-per-bench", type=float, default=3.0)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--stress-seconds", type=float, default=60.0)
    args = parser.parse_args()

    selected = get_backend()
    if selected.name != "opengl":
        print("cpu_gpu_benchmark currently implements the OpenGL benchmark only.")
        print(f"Selected backend: {selected.name.upper()}")
        print("No benchmark was run.")
        return 0

    _load_opengl_benchmark_modules()

    gm.sdl_check(gm.sdl.SDL_Init(gm.SDL_INIT_VIDEO) == 0, "SDL_Init failed")
    window = None
    context = None
    try:
        gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_MAJOR_VERSION, 2)
        gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_MINOR_VERSION, 1)
        gm.sdl.SDL_GL_SetAttribute(gm.SDL_GL_CONTEXT_PROFILE_MASK, gm.SDL_GL_CONTEXT_PROFILE_COMPATIBILITY)
        window = gm.sdl.SDL_CreateWindow(
            b"cpu_gpu_benchmark",
            0,
            0,
            64,
            64,
            gm.SDL_WINDOW_OPENGL | gm.SDL_WINDOW_HIDDEN,
        )
        gm.sdl_check(bool(window), "SDL_CreateWindow failed")
        context = gm.sdl.SDL_GL_CreateContext(window)
        gm.sdl_check(bool(context), "SDL_GL_CreateContext failed")

        gpu_stress.print_capabilities()
        gpu_stress.print_telemetry_availability()

        results: list[BenchResult] = []
        for i, n in enumerate([256, 512], start=1):
            results.extend(run_size(n, args.seconds_per_bench, seed=100 + i))
        print_comparison_table(results)

        if not args.skip_stress:
            run_512_stress(args.stress_seconds)
        return 0
    finally:
        if context:
            gm.sdl.SDL_GL_DeleteContext(context)
        if window:
            gm.sdl.SDL_DestroyWindow(window)
        gm.sdl.SDL_Quit()


if __name__ == "__main__":
    raise SystemExit(main())
