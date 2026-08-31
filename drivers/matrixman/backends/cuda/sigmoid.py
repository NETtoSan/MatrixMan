#!/usr/bin/env python3
"""Direct CUDA backend diagnostic for elementwise sigmoid."""

import numpy as np

from .gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    values = np.array([[-20.0, -2.0, -0.5, 0.0, 0.5, 2.0, 20.0]], dtype=np.float32)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        input_pointer = cuda.to_device(values)
        output_pointer = cuda.allocate(values.nbytes)
        try:
            cuda.sigmoid(input_pointer, output_pointer, values.size, (1, 1, 1, 7), (0, 0, 7, 1), 0)
            expected = 1.0 / (1.0 + np.exp(-values))
            print_check("Sigmoid", cuda.from_device(output_pointer, values.shape), expected)
        finally:
            cuda.free(input_pointer)
            cuda.free(output_pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
