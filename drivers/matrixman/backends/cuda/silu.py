#!/usr/bin/env python3
"""Standalone legacy-CUDA SiLU diagnostic."""

import numpy as np

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    values = np.array([-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0], dtype=np.float32)
    expected = (values / (1.0 + np.exp(-values))).astype(np.float32)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])

        input_pointer = cuda.to_device(values)
        output_pointer = cuda.allocate(values.nbytes)
        inplace_pointer = cuda.to_device(values)
        try:
            cuda.silu(input_pointer, output_pointer, values.size)
            print_check("SiLU out-of-place", cuda.from_device(output_pointer, values.shape), expected)
            cuda.silu(inplace_pointer, inplace_pointer, values.size)
            print_check("SiLU inplace", cuda.from_device(inplace_pointer, values.shape), expected)
        finally:
            for pointer in (input_pointer, output_pointer, inplace_pointer):
                cuda.free(pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
