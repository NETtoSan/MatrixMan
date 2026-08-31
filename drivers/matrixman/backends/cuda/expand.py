#!/usr/bin/env python3
"""Standalone diagnostic for CUDA metadata-only expand and meshgrid."""

import numpy as np
import torch

from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor
from .backend import CudaBackend, CudaTensorOwner


def _assert_expanded(name, tensor, shape, strides, source, expected_source, cuda):
    if not isinstance(tensor, MatrixManTensor) or not isinstance(tensor._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result is not CUDA-backed MatrixManTensor")
    if tuple(tensor.shape) != tuple(shape) or tuple(tensor._logical_strides) != tuple(strides):
        raise AssertionError(
            f"{name}: metadata is {tuple(tensor.shape)}, {tensor._logical_strides}"
        )
    if tensor._owner is not source._owner or tensor._owner.pointer.value != source._owner.pointer.value:
        raise AssertionError(f"{name}: expand did not preserve CUDA storage")
    if tensor._storage_offset != source._storage_offset or tensor.dtype != source.dtype:
        raise AssertionError(f"{name}: expand did not preserve offset or dtype")
    if tensor.device != PRIVATEUSE_DEVICE:
        raise AssertionError(f"{name}: unexpected device {tensor.device}")
    actual_source = cuda.from_device(source._owner.pointer, tuple(source.shape))
    expected_source = np.asarray(expected_source, dtype=np.float32).reshape(tuple(source.shape))
    max_abs = float(np.max(np.abs(actual_source - expected_source))) if expected_source.size else 0.0
    if not np.allclose(actual_source, expected_source, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{name}: source values do not match reference")
    print(
        f"{name}: shape={list(tensor.shape)} strides={list(tensor._logical_strides)} "
        f"offset={tensor._storage_offset} shared pointer=True "
        f"max abs diff={max_abs:.6g}"
    )


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the expand diagnostic")
    cuda = backend.execution

    source_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    source_b = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    owners = []
    try:
        a = __import__("drivers.matrixman", fromlist=["to_device"]).to_device(source_a)
        b = __import__("drivers.matrixman", fromlist=["to_device"]).to_device(source_b)
        owners.extend((a._owner, b._owner))
        expanded_a = a.expand(2, 3)
        expanded_b = b.expand(4, 3)
        _assert_expanded("[3] -> [2,3]", expanded_a, (2, 3), (0, 1), a, [1, 2, 3], cuda)
        _assert_expanded("[1,3] -> [4,3]", expanded_b, (4, 3), (0, 1), b, [[1, 2, 3]], cuda)

        retained = b.expand(-1, 3)
        _assert_expanded("-1 retention", retained, (1, 3), (3, 1), b, [[1, 2, 3]], cuda)
        try:
            b.expand(4, 2)
        except (RuntimeError, ValueError):
            print("invalid non-singleton expansion: rejected")
        else:
            raise AssertionError("invalid non-singleton expansion was accepted")

        h, w = 4, 6
        sx = torch.arange(end=w, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy = torch.arange(end=h, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy_grid, sx_grid = torch.meshgrid(sy, sx, indexing="ij")
        _assert_expanded("meshgrid sy", sy_grid, (h, w), (1, 0), sy, np.arange(h) + 0.5, cuda)
        _assert_expanded("meshgrid sx", sx_grid, (h, w), (0, 1), sx, np.arange(w) + 0.5, cuda)
        sy_values = cuda.from_device(sy._owner.pointer, tuple(sy.shape))
        sx_values = cuda.from_device(sx._owner.pointer, tuple(sx.shape))
        np.testing.assert_allclose(
            np.broadcast_to(sy_values.reshape(h, 1), (h, w)),
            np.broadcast_to((np.arange(h, dtype=np.float32) + 0.5).reshape(h, 1), (h, w)),
        )
        np.testing.assert_allclose(
            np.broadcast_to(sx_values.reshape(1, w), (h, w)),
            np.broadcast_to((np.arange(w, dtype=np.float32) + 0.5).reshape(1, w), (h, w)),
        )
        print("meshgrid: HxW shapes, zero-stride metadata, shared CUDA storage: PASS")
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
