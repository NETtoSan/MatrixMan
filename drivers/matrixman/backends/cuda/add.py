#!/usr/bin/env python3
"""Standalone legacy-CUDA tensor-add diagnostic."""

import numpy as np

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    left = np.arange(1, 1 + 1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    right = np.arange(100, 100 + left.size, dtype=np.float32).reshape(left.shape)
    expected = left + right
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        left_pointer = cuda.to_device(left)
        right_pointer = cuda.to_device(right)
        output_pointer = cuda.allocate(expected.nbytes)
        try:
            cuda.add(left_pointer, right_pointer, output_pointer, expected.size)
            print_check("NCHW add", cuda.from_device(output_pointer, expected.shape), expected)
        finally:
            for pointer in (left_pointer, right_pointer, output_pointer):
                cuda.free(pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
