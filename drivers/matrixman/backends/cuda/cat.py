#!/usr/bin/env python3
"""Standalone legacy-CUDA concatenation diagnostic."""

import numpy as np

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def run_cat(cuda, inputs, dimension):
    output_shape = list(inputs[0].shape)
    output_shape[dimension] = sum(array.shape[dimension] for array in inputs)
    output_shape = tuple(output_shape)
    padded_output = (1,) * (4 - len(output_shape)) + output_shape
    output_pointer = cuda.allocate(int(np.prod(output_shape)) * 4)
    input_pointers = []
    offset = 0
    try:
        for array in inputs:
            pointer = cuda.to_device(array)
            input_pointers.append(pointer)
            padded_input = (1,) * (4 - array.ndim) + array.shape
            cuda.cat_copy(
                pointer,
                output_pointer,
                int(np.prod(output_shape)),
                dimension + 4 - array.ndim,
                offset,
                padded_input,
                padded_output,
            )
            offset += array.shape[dimension]
        return output_pointer, output_shape
    except Exception:
        cuda.free(output_pointer)
        raise
    finally:
        for pointer in input_pointers:
            cuda.free(pointer)


def check_cat(cuda, inputs, dimension, label):
    expected = np.concatenate(inputs, axis=dimension)
    output_pointer, output_shape = run_cat(cuda, inputs, dimension)
    try:
        actual = cuda.from_device(output_pointer, output_shape)
        print_check(label, actual, expected)
    finally:
        cuda.free(output_pointer)


def run_diagnostic() -> int:
    channel_inputs = (
        np.arange(1, 1 + 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3),
        np.arange(100, 100 + 1 * 3 * 2 * 3, dtype=np.float32).reshape(1, 3, 2, 3),
        np.arange(200, 200 + 1 * 1 * 2 * 3, dtype=np.float32).reshape(1, 1, 2, 3),
    )
    height_inputs = (
        np.arange(1, 1 + 1 * 2 * 1 * 3, dtype=np.float32).reshape(1, 2, 1, 3),
        np.arange(50, 50 + 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3),
    )
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        check_cat(cuda, channel_inputs, 1, "NCHW channel cat")
        check_cat(cuda, height_inputs, 2, "NCHW height cat")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
