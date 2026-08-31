#!/usr/bin/env python3
"""Direct CUDA backend diagnostic for elementwise subtraction."""

import numpy as np

from .gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])

        def run_case(name, left, right, shape, strides, alpha, expected):
            left_pointer = cuda.to_device(left)
            right_pointer = cuda.to_device(right)
            output_pointer = cuda.allocate(int(np.prod(shape)) * np.dtype(np.float32).itemsize)
            try:
                cuda.sub(
                    left_pointer, right_pointer, output_pointer, int(np.prod(shape)),
                    shape, strides, strides, 0, 0, alpha,
                )
                print_check(name, cuda.from_device(output_pointer, shape), expected)
            finally:
                for pointer in (left_pointer, right_pointer, output_pointer):
                    cuda.free(pointer)

        left_2d = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        right_2d = np.arange(2, 8, dtype=np.float32).reshape(2, 3)
        run_case("C = A - B (rank 2)", left_2d, right_2d, (1, 1, 2, 3), (0, 0, 3, 1), 1.0, left_2d.reshape(1, 1, 2, 3) - right_2d.reshape(1, 1, 2, 3))
        run_case("C = A - 0.5 * B (rank 2)", left_2d, right_2d, (1, 1, 2, 3), (0, 0, 3, 1), 0.5, (left_2d - 0.5 * right_2d).reshape(1, 1, 2, 3))

        left = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)
        right = np.arange(2, 26, dtype=np.float32).reshape(left.shape)
        strides = (0, 12, 4, 1)
        run_case("C = A - B (rank 3)", left, right, (1, 2, 3, 4), strides, 1.0, (left - right).reshape(1, 2, 3, 4))
        run_case("C = A - (-2) * B (rank 3)", left, right, (1, 2, 3, 4), strides, -2.0, (left + 2.0 * right).reshape(1, 2, 3, 4))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
