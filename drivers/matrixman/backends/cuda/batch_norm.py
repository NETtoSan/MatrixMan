#!/usr/bin/env python3
"""Standalone legacy-CUDA BatchNorm inference diagnostic."""

import numpy as np

from .gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    input_array = np.arange(1, 1 + 1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    running_mean = np.array([2.0, 8.0, 20.0], dtype=np.float32)
    running_var = np.array([1.0, 4.0, 9.0], dtype=np.float32)
    weight = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    bias = np.array([0.0, -1.0, 3.0], dtype=np.float32)
    eps = 1e-5
    expected = (
        (input_array - running_mean.reshape(1, 3, 1, 1))
        / np.sqrt(running_var.reshape(1, 3, 1, 1) + eps)
        * weight.reshape(1, 3, 1, 1)
        + bias.reshape(1, 3, 1, 1)
    ).astype(np.float32)

    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        input_pointer = cuda.to_device(input_array)
        mean_pointer = cuda.to_device(running_mean)
        var_pointer = cuda.to_device(running_var)
        weight_pointer = cuda.to_device(weight)
        bias_pointer = cuda.to_device(bias)
        output_pointer = cuda.allocate(expected.nbytes)
        try:
            cuda.batch_norm(
                input_pointer,
                mean_pointer,
                var_pointer,
                weight_pointer,
                bias_pointer,
                output_pointer,
                input_array.size,
                3,
                2 * 4,
                eps,
            )
            actual = cuda.from_device(output_pointer, expected.shape)
            print_check("BatchNorm inference", actual, expected)
        finally:
            for pointer in (
                input_pointer,
                mean_pointer,
                var_pointer,
                weight_pointer,
                bias_pointer,
                output_pointer,
            ):
                cuda.free(pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
