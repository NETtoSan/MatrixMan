#!/usr/bin/env python3
"""Standalone CUDA diagnostic for in-place scalar fill."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor
from .backend import CudaBackend, CudaTensorOwner


def _check(name, tensor, expected, cuda, owner, pointer):
    if not isinstance(tensor, MatrixManTensor) or not isinstance(tensor._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result is not CUDA-backed MatrixManTensor")
    if tensor._owner is not owner or tensor._owner.pointer.value != pointer:
        raise AssertionError(f"{name}: fill replaced the CUDA storage")
    actual = cuda.from_device(tensor._owner.pointer, tuple(tensor.shape))
    expected = np.asarray(expected, dtype=np.float32).reshape(tuple(tensor.shape))
    max_abs = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    print(f"{name}: shape={list(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
          f"same owner=True same pointer=True max abs diff={max_abs:.6g}")
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{name}: result mismatch")


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the fill diagnostic")
    cuda = backend.execution
    owners = []
    try:
        x = torch.empty((2, 3), device=PRIVATEUSE_DEVICE, dtype=torch.float32)
        owner, pointer = x._owner, x._owner.pointer.value
        owners.append(owner)
        returned = x.fill_(2.5)
        if returned is not x:
            raise AssertionError("fill_ did not return the same tensor object")
        _check("fill_(2.5)", x, np.full((2, 3), 2.5), cuda, owner, pointer)
        x.fill_(-3.0)
        _check("fill_(-3.0)", x, np.full((2, 3), -3.0), cuda, owner, pointer)

        full = torch.full((24, 1), 8.0, dtype=torch.float32, device=PRIVATEUSE_DEVICE)
        owners.append(full._owner)
        _check("torch.full((24,1), 8.0)", full, np.full((24, 1), 8.0), cuda, full._owner, full._owner.pointer.value)
        for size, value in ((6400, 8.0), (1600, 16.0), (400, 32.0)):
            result = torch.full((size, 1), value, dtype=torch.float32, device=PRIVATEUSE_DEVICE)
            owners.append(result._owner)
            _check(f"torch.full(({size},1), {value})", result, np.full((size, 1), value), cuda, result._owner, result._owner.pointer.value)

        base = matrixman.to_device(torch.arange(12, dtype=torch.float32))
        owners.append(base._owner)
        offset_view = MatrixManTensor._from_owner(
            base._owner, (3,), storage_offset=2, logical_strides=(1,)
        )
        pointer = base._owner.pointer.value
        offset_view.fill_(7.0)
        actual = cuda.from_device(base._owner.pointer, (12,))
        expected = np.arange(12, dtype=np.float32)
        expected[2:5] = 7.0
        np.testing.assert_array_equal(actual, expected)
        print("nonzero storage offset: logical region only written, PASS")
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
