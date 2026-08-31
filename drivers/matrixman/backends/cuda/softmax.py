#!/usr/bin/env python3
"""Standalone CUDA diagnostic for stable logical-stride softmax."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor, contiguous_strides
from .backend import CudaBackend, CudaTensorOwner


def _check(name, result, expected, cuda, owner):
    if not isinstance(result, MatrixManTensor) or not isinstance(result._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result is not CUDA-backed MatrixManTensor")
    if result._owner is owner:
        raise AssertionError(f"{name}: softmax reused input storage")
    if tuple(result.shape) != tuple(expected.shape):
        raise AssertionError(f"{name}: unexpected shape {tuple(result.shape)}")
    if tuple(result._logical_strides) != contiguous_strides(tuple(result.shape)):
        raise AssertionError(f"{name}: output is not contiguous")
    if result.device != PRIVATEUSE_DEVICE or result.dtype != torch.float32:
        raise AssertionError(f"{name}: unexpected device or dtype")
    actual = cuda.from_device(result._owner.pointer, tuple(result.shape))
    expected_array = expected.numpy()
    max_abs = float(np.max(np.abs(actual - expected_array))) if actual.size else 0.0
    sums = actual.sum(axis=-1) if actual.ndim else actual
    print(f"{name}: shape={list(result.shape)} max abs diff={max_abs:.6g} "
          f"row-sum range=[{float(sums.min()):.6g}, {float(sums.max()):.6g}]")
    if not np.allclose(actual, expected_array, rtol=1e-4, atol=1e-5):
        raise AssertionError(f"{name}: result mismatch")


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the softmax diagnostic")
    cuda = backend.execution
    owners = []
    try:
        x_cpu = torch.tensor(
            [[-2.0, 0.5, 1.0, 3.0, -1.0], [4.0, -3.0, 0.0, 0.25, 2.0], [1.5, 1.5, -2.0, 0.0, 5.0]],
            dtype=torch.float32,
        )
        x = matrixman.to_device(x_cpu)
        owners.append(x._owner)
        result = x.softmax(1)
        try:
            _check("contiguous dim=1", result, torch.softmax(x_cpu, 1), cuda, x._owner)
        finally:
            result._owner.release()
        result = x.softmax(-1)
        try:
            _check("contiguous dim=-1", result, torch.softmax(x_cpu, -1), cuda, x._owner)
        finally:
            result._owner.release()

        cube_cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) - 8.0
        cube = matrixman.to_device(cube_cpu)
        owners.append(cube._owner)
        result = cube.softmax(1)
        try:
            _check("3D dim=1", result, torch.softmax(cube_cpu, 1), cuda, cube._owner)
        finally:
            result._owner.release()

        transposed_cpu = cube_cpu.transpose(1, 2)
        transposed = cube.transpose(1, 2)
        print("transposed input shape:", list(transposed.shape))
        print("transposed logical strides:", list(transposed._logical_strides))
        result = transposed.softmax(1)
        try:
            _check("transposed dim=1", result, torch.softmax(transposed_cpu, 1), cuda, transposed._owner)
        finally:
            result._owner.release()

        batch, bins, anchors = 1, 16, 7
        dfl_cpu = torch.arange(batch * 4 * bins * anchors, dtype=torch.float32).reshape(
            batch, 4, bins, anchors
        ) / 10.0
        dfl = matrixman.to_device(dfl_cpu)
        owners.append(dfl._owner)
        y = dfl.view(batch, 4, bins, anchors).transpose(2, 1)
        print("DFL transposed shape:", list(y.shape))
        print("DFL transposed logical strides:", list(y._logical_strides))
        result = y.softmax(1)
        try:
            _check("YOLO DFL softmax", result, torch.softmax(dfl_cpu.view(batch, 4, bins, anchors).transpose(2, 1), 1), cuda, y._owner)
        finally:
            result._owner.release()
        print("YOLO DFL transpose -> softmax: PASS")
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
