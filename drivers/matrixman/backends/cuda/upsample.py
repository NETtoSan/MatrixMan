#!/usr/bin/env python3
"""Standalone legacy-CUDA nearest-neighbor resize diagnostic."""

import numpy as np

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def nearest_reference(values, output_height, output_width):
    input_height, input_width = values.shape[-2:]
    y = (np.arange(output_height) * input_height // output_height).astype(np.int64)
    x = (np.arange(output_width) * input_width // output_width).astype(np.int64)
    return values[:, :, y[:, None], x[None, :]]


def check_resize(cuda, values, output_height, output_width, label):
    expected = nearest_reference(values, output_height, output_width)
    input_pointer = cuda.to_device(values)
    output_pointer = cuda.allocate(expected.nbytes)
    try:
        cuda.upsample_nearest2d(
            input_pointer,
            output_pointer,
            expected.size,
            values.shape[1],
            values.shape[2],
            values.shape[3],
            output_height,
            output_width,
        )
        actual = cuda.from_device(output_pointer, expected.shape)
        print_check(label, actual, expected)
    finally:
        cuda.free(input_pointer)
        cuda.free(output_pointer)


def run_diagnostic() -> int:
    exact_input = np.arange(1, 1 + 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    non_integer_input = np.arange(1, 1 + 1 * 1 * 3 * 4, dtype=np.float32).reshape(1, 1, 3, 4)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        check_resize(cuda, exact_input, 4, 6, "NCHW nearest 2x resize")
        check_resize(cuda, non_integer_input, 5, 7, "NCHW nearest non-integer resize")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
