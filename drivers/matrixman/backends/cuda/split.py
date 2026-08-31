#!/usr/bin/env python3
"""Standalone legacy-CUDA split diagnostic."""

import numpy as np

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def run_split(cuda, input_array, split_size, dimension):
    input_pointer = cuda.to_device(input_array)
    owners = []
    offset = 0
    shape = input_array.shape
    for chunk_size in range(0, shape[dimension], split_size):
        current_size = min(split_size, shape[dimension] - chunk_size)
        output_shape = shape[:dimension] + (current_size,) + shape[dimension + 1:]
        output_pointer = cuda.allocate(np.prod(output_shape, dtype=np.int64) * 4)
        padded_input = (1,) * (4 - input_array.ndim) + shape
        padded_input_strides = (0,) * (4 - input_array.ndim) + tuple(
            int(value) // 4 for value in input_array.strides
        )
        padded_output = (1,) * (4 - input_array.ndim) + output_shape
        cuda.split_copy(
            input_pointer,
            output_pointer,
            int(np.prod(output_shape)),
            dimension + 4 - input_array.ndim,
            offset,
            padded_input,
            padded_input_strides,
            padded_output,
        )
        owners.append((output_pointer, output_shape))
        offset += current_size
    return input_pointer, owners


def check_split(cuda, input_array, split_size, dimension, expected_parts, label):
    input_pointer, outputs = run_split(cuda, input_array, split_size, dimension)
    try:
        for index, (pointer, output_shape) in enumerate(outputs):
            actual = cuda.from_device(pointer, output_shape)
            print_check(f"{label} part {index}", actual, expected_parts[index])
    finally:
        cuda.free(input_pointer)
        for pointer, _ in outputs:
            cuda.free(pointer)


def run_diagnostic() -> int:
    channel_input = np.arange(1, 1 + 1 * 16 * 4 * 4, dtype=np.float32).reshape(1, 16, 4, 4)
    uneven_input = np.arange(1, 1 + 10 * 2 * 3, dtype=np.float32).reshape(1, 10, 2, 3)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        check_split(
            cuda,
            channel_input,
            8,
            1,
            (channel_input[:, :8], channel_input[:, 8:]),
            "channel split 16 -> 8,8",
        )
        check_split(
            cuda,
            uneven_input,
            4,
            1,
            (uneven_input[:, :4], uneven_input[:, 4:8], uneven_input[:, 8:]),
            "channel split 10 -> 4,4,2",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
