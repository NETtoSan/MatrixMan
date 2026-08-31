#!/usr/bin/env python3
"""CUDA split diagnostic for logical non-contiguous MatrixMan views."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor, contiguous_strides
from .backend import CudaBackend, CudaTensorOwner


def _check_parts(label, parts, expected, cuda):
    if len(parts) != len(expected):
        raise AssertionError(f"{label}: expected {len(expected)} parts, got {len(parts)}")
    for index, (part, reference) in enumerate(zip(parts, expected)):
        if not isinstance(part, MatrixManTensor) or not isinstance(part._owner, CudaTensorOwner):
            raise AssertionError(f"{label} part {index}: not CUDA-backed")
        if tuple(part._logical_strides) != contiguous_strides(tuple(part.shape)):
            raise AssertionError(f"{label} part {index}: output is not contiguous")
        actual = cuda.from_device(part._owner.pointer, tuple(part.shape))
        max_abs = float(np.max(np.abs(actual - reference.numpy()))) if actual.size else 0.0
        print(f"{label} part {index}: shape={list(part.shape)} max abs diff={max_abs:.6g}")
        if not np.allclose(actual, reference.numpy(), rtol=1e-5, atol=1e-6):
            raise AssertionError(f"{label} part {index}: mismatch")


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the split diagnostic")
    cuda = backend.execution
    owners = []
    try:
        x_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        x = matrixman.to_device(x_cpu)
        owners.append(x._owner)
        y_cpu = x_cpu.transpose(1, 2)
        y = x.transpose(1, 2)
        print("transposed shape:", list(y.shape))
        print("transposed logical strides:", list(y._logical_strides))
        parts = torch.chunk(y, 2, dim=1)
        _check_parts("transpose split dim=1", parts, torch.chunk(y_cpu, 2, dim=1), cuda)
        parts = torch.chunk(y, 2, dim=0)
        _check_parts("transpose split dim=0", parts, torch.chunk(y_cpu, 2, dim=0), cuda)

        expanded_base = matrixman.to_device(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        owners.append(expanded_base._owner)
        expanded = expanded_base.expand(3, 4)
        print("expanded logical strides:", list(expanded._logical_strides))
        parts = torch.chunk(expanded, 3, dim=0)
        _check_parts("zero-stride split", parts, torch.chunk(torch.ones(3, 1) * torch.tensor([1.0, 2.0, 3.0, 4.0]), 3, dim=0), cuda)

        offset_base = matrixman.to_device(torch.arange(12, dtype=torch.float32))
        owners.append(offset_base._owner)
        offset_view = MatrixManTensor._from_owner(
            offset_base._owner, (2, 3), storage_offset=2, logical_strides=(3, 1)
        )
        parts = torch.chunk(offset_view, 2, dim=0)
        _check_parts("nonzero offset split", parts, torch.arange(2, 8, dtype=torch.float32).reshape(2, 3).chunk(2, dim=0), cuda)

        distance_cpu = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        distance = matrixman.to_device(distance_cpu)
        owners.append(distance._owner)
        left, right = torch.chunk(distance, 2, dim=1)
        _check_parts("YOLO dist2bbox chunk", (left, right), torch.chunk(distance_cpu, 2, dim=1), cuda)
        print("YOLO distance.chunk(2, dim=1): PASS")
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
