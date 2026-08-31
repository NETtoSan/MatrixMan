#!/usr/bin/env python3
"""Standalone CUDA diagnostic for metadata-only unsqueeze."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor
from .backend import CudaBackend, CudaTensorOwner


def _check(name, result, expected, cuda, source):
    if not isinstance(result, MatrixManTensor) or not isinstance(result._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result is not CUDA-backed MatrixManTensor")
    if result._owner is not source._owner or result._owner.pointer.value != source._owner.pointer.value:
        raise AssertionError(f"{name}: unsqueeze did not preserve storage")
    if tuple(result.shape) != tuple(expected.shape) or tuple(result._logical_strides) != expected.stride():
        raise AssertionError(
            f"{name}: metadata is {tuple(result.shape)}, {result._logical_strides}; "
            f"expected {tuple(expected.shape)}, {expected.stride()}"
        )
    if result._storage_offset != source._storage_offset or result.dtype != torch.float32:
        raise AssertionError(f"{name}: unsqueeze changed offset or dtype")
    if result.device != PRIVATEUSE_DEVICE:
        raise AssertionError(f"{name}: unexpected device {result.device}")
    raw = cuda.from_device(result._owner.pointer, tuple(result._owner.shape))
    actual = np.ndarray(
        shape=tuple(result.shape),
        dtype=np.float32,
        buffer=raw,
        offset=int(result._storage_offset) * 4,
        strides=tuple(int(stride) * 4 for stride in result._logical_strides),
    )
    expected_array = expected.numpy()
    max_abs = float(np.max(np.abs(actual - expected_array))) if actual.size else 0.0
    print(f"{name}: shape={list(result.shape)} strides={list(result._logical_strides)} "
          f"same owner/pointer=True max abs diff={max_abs:.6g}")
    if not np.allclose(actual, expected_array, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{name}: values mismatch")


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the unsqueeze diagnostic")
    cuda = backend.execution
    owners = []
    try:
        x_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        x = matrixman.to_device(x_cpu)
        owners.append(x._owner)
        for dimension in (0, 1, -1):
            result = x.unsqueeze(dimension)
            try:
                _check(f"contiguous unsqueeze({dimension})", result, x_cpu.unsqueeze(dimension), cuda, x)
            finally:
                # The owner is shared and released once in the outer cleanup.
                pass
        try:
            x.unsqueeze(3)
        except (IndexError, RuntimeError, ValueError):
            print("invalid dimension: rejected")
        else:
            raise AssertionError("unsqueeze accepted an invalid dimension")

        y_cpu = x_cpu.transpose(0, 1)
        y = x.transpose(0, 1)
        result = y.unsqueeze(0)
        _check("transposed unsqueeze(0)", result, y_cpu.unsqueeze(0), cuda, y)
        print("transposed input shape:", list(y.shape))
        print("transposed unsqueeze strides:", list(result._logical_strides))

        h, w = 4, 6
        sx_cpu = torch.arange(w, dtype=torch.float32) + 0.5
        sy_cpu = torch.arange(h, dtype=torch.float32) + 0.5
        sx = torch.arange(end=w, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy = torch.arange(end=h, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy_grid, sx_grid = torch.meshgrid(sy, sx, indexing="ij")
        anchor = torch.stack((sx_grid, sy_grid), -1).view(-1, 2).transpose(0, 1)
        anchor_u = anchor.unsqueeze(0)
        cpu_sy_grid, cpu_sx_grid = torch.meshgrid(sy_cpu, sx_cpu, indexing="ij")
        anchor_cpu = torch.stack((cpu_sx_grid, cpu_sy_grid), -1).view(-1, 2).transpose(0, 1)
        owners.extend((sx._owner, sy._owner, anchor._owner))
        _check("YOLO anchor unsqueeze(0)", anchor_u, anchor_cpu.unsqueeze(0), cuda, anchor)
        print("YOLO anchor transpose -> unsqueeze: PASS")
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
