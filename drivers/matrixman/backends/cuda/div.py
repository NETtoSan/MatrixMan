#!/usr/bin/env python3
"""Direct CUDA backend diagnostic for scalar division."""

import numpy as np

from .gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])

        def run_case(name, values, shape, strides, divisor):
            input_pointer = cuda.to_device(values)
            output_pointer = cuda.allocate(int(np.prod(shape)) * np.dtype(np.float32).itemsize)
            try:
                cuda.div_scalar(
                    input_pointer, output_pointer, int(np.prod(shape)),
                    shape, strides, 0, divisor,
                )
                print_check(name, cuda.from_device(output_pointer, shape), values.reshape(shape) / divisor)
            finally:
                for pointer in (input_pointer, output_pointer):
                    cuda.free(pointer)

        values_2d = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        run_case("C = A / 2 (rank 2)", values_2d, (1, 1, 2, 3), (0, 0, 3, 1), 2.0)
        values = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        run_case("C = A / 2 (rank 3)", values, (1, 2, 3, 4), (0, 12, 4, 1), 2.0)
        run_case("C = A / -2 (rank 3)", values, (1, 2, 3, 4), (0, 12, 4, 1), -2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
