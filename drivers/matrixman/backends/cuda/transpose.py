#!/usr/bin/env python3
"""Standalone CUDA diagnostic for metadata-only transpose."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor
from .backend import CudaBackend, CudaTensorOwner


def _check(name, result, expected_shape, expected_strides, expected_values, cuda, source):
    if not isinstance(result, MatrixManTensor) or not isinstance(result._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result is not CUDA-backed MatrixManTensor")
    if tuple(result.shape) != tuple(expected_shape):
        raise AssertionError(f"{name}: unexpected shape {tuple(result.shape)}")
    if tuple(result._logical_strides) != tuple(expected_strides):
        raise AssertionError(f"{name}: unexpected strides {result._logical_strides}")
    if result._owner is not source._owner or result._owner.pointer.value != source._owner.pointer.value:
        raise AssertionError(f"{name}: transpose did not preserve storage")
    if result._storage_offset != source._storage_offset or result.dtype != source.dtype:
        raise AssertionError(f"{name}: transpose changed offset or dtype")
    if result.device != PRIVATEUSE_DEVICE:
        raise AssertionError(f"{name}: unexpected device {result.device}")
    raw = cuda.from_device(result._owner.pointer, tuple(result._owner.shape))
    logical = np.ndarray(
        shape=tuple(result.shape),
        dtype=np.float32,
        buffer=raw,
        offset=int(result._storage_offset) * 4,
        strides=tuple(int(stride) * 4 for stride in result._logical_strides),
    )
    actual = logical
    expected = np.asarray(expected_values, dtype=np.float32)
    max_abs = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    print(
        f"{name}: shape={list(result.shape)} strides={list(result._logical_strides)} "
        f"same owner/pointer=True max abs diff={max_abs:.6g}"
    )
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{name}: result mismatch")


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the transpose diagnostic")
    cuda = backend.execution
    owners = []
    try:
        x_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        x = matrixman.to_device(x_cpu)
        owners.append(x._owner)
        transposed = x.transpose(0, 1)
        _check("transpose(0,1)", transposed, (3, 2), (1, 3), x_cpu.numpy().T, cuda, x)
        negative = x.transpose(-2, -1)
        _check("transpose(-2,-1)", negative, (3, 2), (1, 3), x_cpu.numpy().T, cuda, x)
        same = x.transpose(1, 1)
        _check("transpose(1,1)", same, (2, 3), (3, 1), x_cpu.numpy(), cuda, x)
        try:
            x.transpose(0, 3)
        except (IndexError, RuntimeError, ValueError):
            print("invalid dimension: rejected")
        else:
            raise AssertionError("transpose accepted an invalid dimension")

        expanded_base = matrixman.to_device(torch.tensor([[1.0, 2.0, 3.0]]))
        owners.append(expanded_base._owner)
        expanded = expanded_base.expand(4, 3)
        expanded_transposed = expanded.transpose(0, 1)
        expected_expanded = np.broadcast_to(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), (4, 3)).T
        _check("expanded transpose", expanded_transposed, (3, 4), (1, 0), expected_expanded, cuda, expanded)

        h, w = 4, 6
        sx_cpu = torch.arange(w, dtype=torch.float32) + 0.5
        sy_cpu = torch.arange(h, dtype=torch.float32) + 0.5
        sx = torch.arange(end=w, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy = torch.arange(end=h, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy_grid, sx_grid = torch.meshgrid(sy, sx, indexing="ij")
        anchor = torch.stack((sx_grid, sy_grid), -1).view(-1, 2)
        anchor_t = anchor.transpose(0, 1)
        cpu_sy_grid, cpu_sx_grid = torch.meshgrid(sy_cpu, sx_cpu, indexing="ij")
        anchor_cpu = torch.stack((cpu_sx_grid, cpu_sy_grid), -1).view(-1, 2)
        owners.extend((sx._owner, sy._owner, anchor._owner))
        _check("YOLO anchor transpose", anchor_t, (2, 24), (1, 2), anchor_cpu.numpy().T, cuda, anchor)
        print("YOLO stack -> view -> transpose: PASS")

        stride_cpu = torch.full((24, 1), 8.0, dtype=torch.float32)
        stride = torch.full((24, 1), 8.0, dtype=torch.float32, device=PRIVATEUSE_DEVICE)
        owners.append(stride._owner)
        stride_t = stride.transpose(0, 1)
        _check("stride tensor transpose", stride_t, (1, 24), (1, 1), stride_cpu.numpy().T, cuda, stride)
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
