"""CUDA-only diagnostic for representative fused Conv2D + SiLU kernels.

This intentionally bypasses MatrixMan's production operator routing.  It loads
the opt-in PTX entry points, compares them with the existing specialized Conv
plus SiLU path, and uses CUDA events for device-side elapsed time.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as functional

from drivers.matrixman.backends.cuda.gpumatrix import CudaExecutionBackend


CASES = (
    ("80x80", 8, 80, "small_c8"),
    ("40x40", 24, 40, "small_c24"),
    ("20x20", 10, 20, "small_c10"),
)


def _launch_separate(execution, pointers, dims):
    input_pointer, weight_pointer, bias_pointer, intermediate_pointer, output_pointer, variant = pointers
    n, channels, height, width, filters, out_height, out_width = dims
    execution.convolution(
        input_pointer, weight_pointer, bias_pointer, intermediate_pointer,
        n, channels, height, width, filters, 3, 3, out_height, out_width,
        1, 1, 1, 1, 1, 1, 1,
        **{f"specialized_3x3_{variant}": True},
    )
    execution.silu(intermediate_pointer, output_pointer, n * filters * out_height * out_width)


def _launch_fused(execution, pointers, dims, variant):
    input_pointer, weight_pointer, bias_pointer, _, output_pointer = pointers
    n, channels, height, width, filters, out_height, out_width = dims
    execution.diagnostic_fused_convolution_silu(
        input_pointer, weight_pointer, bias_pointer, output_pointer,
        n, channels, height, width, filters, 3, 3, out_height, out_width,
        1, 1, 1, 1, 1, 1, 1, variant,
    )


def _timed(execution, launcher, warmup, iterations):
    for _ in range(warmup):
        launcher()
    execution.synchronize("diagnostic-warmup")
    samples = []
    for _ in range(iterations):
        execution.gpu_timing_start()
        launcher()
        execution.gpu_timing_stop()
        execution.synchronize("diagnostic-timed")
        samples.append(execution.gpu_timing_elapsed_ms())
    return float(np.median(samples)), float(np.mean(samples))


def _attributes(execution, variant):
    separate = getattr(execution, f"convolution_{variant}_function")
    fused = getattr(execution, f"diagnostic_fused_{variant}_silu_function")
    return execution.function_attributes(separate), execution.function_attributes(fused)


def run(warmup: int = 5, iterations: int = 30) -> int:
    execution = CudaExecutionBackend()
    pointers_to_free = []
    try:
        if not execution._gpu_timing_supported:
            raise RuntimeError("CUDA event timing is unavailable on this driver")
        print(f"CUDA device: {execution.info['name']}")
        print(f"warmup={warmup} iterations={iterations} timing=CUDA events")
        for label, channels, spatial, variant in CASES:
            n = 1
            filters = channels
            shape = (n, channels, spatial, spatial)
            weight_shape = (filters, channels, 3, 3)
            generator = torch.Generator(device="cpu").manual_seed(20260909 + spatial)
            input_cpu = torch.randn(shape, generator=generator, dtype=torch.float32)
            weight_cpu = torch.randn(weight_shape, generator=generator, dtype=torch.float32)
            bias_cpu = torch.randn((filters,), generator=generator, dtype=torch.float32)
            output_shape = (n, filters, spatial, spatial)
            dims = (n, channels, spatial, spatial, filters, spatial, spatial)

            input_pointer = execution.to_device(input_cpu.numpy(), category="activation")
            weight_pointer = execution.to_device(weight_cpu.numpy(), category="parameter")
            bias_pointer = execution.to_device(bias_cpu.numpy(), category="parameter")
            intermediate_pointer = execution.allocate(int(np.prod(output_shape)) * 4)
            separate_output_pointer = execution.allocate(int(np.prod(output_shape)) * 4)
            fused_output_pointer = execution.allocate(int(np.prod(output_shape)) * 4)
            pointers_to_free.extend((input_pointer, weight_pointer, bias_pointer,
                                     intermediate_pointer, separate_output_pointer,
                                     fused_output_pointer))
            separate_pointers = (input_pointer, weight_pointer, bias_pointer,
                                 intermediate_pointer, separate_output_pointer, variant)
            fused_pointers = (input_pointer, weight_pointer, bias_pointer,
                              intermediate_pointer, fused_output_pointer)

            separate_median, separate_mean = _timed(
                execution, lambda: _launch_separate(execution, separate_pointers, dims),
                warmup, iterations,
            )
            separate_output = execution.from_device(separate_output_pointer, output_shape)
            fused_median, fused_mean = _timed(
                execution, lambda: _launch_fused(execution, fused_pointers, dims, variant),
                warmup, iterations,
            )
            fused_output = execution.from_device(fused_output_pointer, output_shape)

            reference = functional.silu(
                functional.conv2d(input_cpu, weight_cpu, bias_cpu, padding=1)
            ).numpy()
            difference = np.abs(fused_output - separate_output)
            reference_difference = np.abs(fused_output - reference)
            separate_attributes, fused_attributes = _attributes(execution, variant)
            improvement = (separate_median - fused_median) / separate_median * 100.0
            intermediate_bytes = int(np.prod(output_shape)) * 4
            print(f"\n[{label}] variant={variant} output={output_shape}")
            print(f"  event_ms separate median={separate_median:.4f} mean={separate_mean:.4f}")
            print(f"  event_ms fused    median={fused_median:.4f} mean={fused_mean:.4f}")
            print(f"  improvement={separate_median - fused_median:.4f} ms ({improvement:.2f}%)")
            print(f"  correctness fused-vs-separate max_abs={difference.max():.8g} "
                  f"mean_abs={difference.mean():.8g} allclose={np.allclose(fused_output, separate_output, rtol=1e-5, atol=1e-5)}")
            print(f"  correctness fused-vs-torch   max_abs={reference_difference.max():.8g} "
                  f"mean_abs={reference_difference.mean():.8g}")
            print("  launches separate=2 fused=1 intermediate_bytes_avoided="
                  f"{intermediate_bytes}")
            print(f"  registers separate={separate_attributes['num_regs']} fused={fused_attributes['num_regs']} "
                  f"max_threads separate={separate_attributes['max_threads_per_block']} "
                  f"fused={fused_attributes['max_threads_per_block']}")
            print(f"  shared_bytes separate={separate_attributes['shared_size_bytes']} "
                  f"fused={fused_attributes['shared_size_bytes']}")
    finally:
        for pointer in reversed(pointers_to_free):
            try:
                execution.free(pointer)
            except Exception:
                pass
        execution.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args(argv)
    return run(args.warmup, args.iterations)


if __name__ == "__main__":
    sys.exit(main())
