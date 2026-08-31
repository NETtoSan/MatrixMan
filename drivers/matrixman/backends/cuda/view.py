#!/usr/bin/env python3
"""Standalone CUDA metadata-only view diagnostic."""

import numpy as np

from drivers.matrixman.tensor import MatrixManTensor, contiguous_strides, infer_view_shape
from drivers.matrixman.backends.cuda.backend import CudaTensorOwner
from drivers.matrixman.backends.cuda.gpumatrix import CudaExecutionBackend


def run_diagnostic() -> int:
    shape = (1, 64, 4, 5)
    values = np.arange(1, 1 + np.prod(shape), dtype=np.float32).reshape(shape)
    requested_shape = (1, 16, -1)
    expected_shape = (1, 16, 80)

    with CudaExecutionBackend() as cuda:
        owner = CudaTensorOwner(
            cuda,
            cuda.to_device(values),
            shape,
            contiguous_strides(shape),
        )
        original = MatrixManTensor._from_owner(
            owner,
            shape,
            logical_strides=contiguous_strides(shape),
        )
        resolved_shape = infer_view_shape(shape, requested_shape)
        viewed = MatrixManTensor._from_owner(
            owner,
            resolved_shape,
            storage_offset=original._storage_offset,
            logical_strides=contiguous_strides(resolved_shape),
        )
        if resolved_shape != expected_shape:
            raise AssertionError(f"unexpected view shape: {resolved_shape}")
        if tuple(viewed._logical_strides) != contiguous_strides(expected_shape):
            raise AssertionError(f"unexpected view strides: {viewed._logical_strides}")
        if viewed._owner.pointer.value != original._owner.pointer.value:
            raise AssertionError("view allocated a different CUDA storage pointer")
        actual = cuda.from_device(owner.pointer, shape).reshape(resolved_shape)
        np.testing.assert_array_equal(actual, values.reshape(resolved_shape))
        print("view shape:", list(resolved_shape))
        print("view strides:", list(viewed._logical_strides))
        print("same CUDA storage pointer:", True)
        print("matches NumPy reshape:", True)
        owner.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
