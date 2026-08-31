#!/usr/bin/env python3
"""Standalone CUDA diagnostic for stack, including meshgrid views."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor, contiguous_strides
from .backend import CudaBackend, CudaTensorOwner


def _check(name, result, expected, cuda, inputs):
    if not isinstance(result, MatrixManTensor) or not isinstance(result._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result is not CUDA-backed MatrixManTensor")
    if tuple(result.shape) != tuple(expected.shape):
        raise AssertionError(f"{name}: unexpected shape {tuple(result.shape)}")
    expected_strides = contiguous_strides(tuple(result.shape))
    if tuple(result._logical_strides) != expected_strides:
        raise AssertionError(f"{name}: unexpected strides {result._logical_strides}")
    if result.device != PRIVATEUSE_DEVICE or result.dtype != torch.float32:
        raise AssertionError(f"{name}: unexpected device or dtype")
    if any(result._owner is tensor._owner for tensor in inputs):
        raise AssertionError(f"{name}: output reused input storage")
    actual = cuda.from_device(result._owner.pointer, tuple(result.shape))
    expected_array = expected.detach().numpy()
    max_abs = float(np.max(np.abs(actual - expected_array))) if actual.size else 0.0
    print(
        f"{name}: shape={list(result.shape)} strides={list(result._logical_strides)} "
        f"new CUDA storage=True max abs diff={max_abs:.6g}"
    )
    if not np.allclose(actual, expected_array, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{name}: result mismatch")


def run_diagnostic() -> int:
    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the stack diagnostic")
    cuda = backend.execution
    owners = []
    try:
        a_cpu = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b_cpu = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        a = matrixman.to_device(a_cpu)
        b = matrixman.to_device(b_cpu)
        owners.extend((a._owner, b._owner))
        dim0 = torch.stack((a, b), 0)
        dim_last = torch.stack((a, b), -1)
        try:
            _check("stack dim=0", dim0, torch.stack((a_cpu, b_cpu), 0), cuda, (a, b))
            _check("stack dim=-1", dim_last, torch.stack((a_cpu, b_cpu), -1), cuda, (a, b))
        finally:
            dim0._owner.release()
            dim_last._owner.release()

        bad = matrixman.to_device(torch.zeros((3, 2)))
        try:
            torch.stack((a, bad), 0)
        except (RuntimeError, ValueError):
            print("mismatched shapes: rejected")
        else:
            raise AssertionError("stack accepted mismatched shapes")
        finally:
            bad._owner.release()

        sx_cpu = torch.arange(6, dtype=torch.float32) + 0.5
        sy_cpu = torch.arange(4, dtype=torch.float32) + 0.5
        sx = torch.arange(end=6, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy = torch.arange(end=4, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
        sy_grid, sx_grid = torch.meshgrid(sy, sx, indexing="ij")
        print("meshgrid sy logical strides:", list(sy_grid._logical_strides))
        print("meshgrid sx logical strides:", list(sx_grid._logical_strides))
        stacked = torch.stack((sx_grid, sy_grid), -1)
        cpu_sy_grid, cpu_sx_grid = torch.meshgrid(sy_cpu, sx_cpu, indexing="ij")
        expected = torch.stack((cpu_sx_grid, cpu_sy_grid), -1)
        _check("YOLO meshgrid stack", stacked, expected, cuda, (sx_grid, sy_grid))
        reshaped = stacked.view(-1, 2)
        actual_reshaped = cuda.from_device(reshaped._owner.pointer, tuple(reshaped.shape))
        np.testing.assert_allclose(actual_reshaped, expected.numpy().reshape(-1, 2))
        print("YOLO stack -> view: shape=[24, 2], values: PASS")
        owners.extend((sx._owner, sy._owner, stacked._owner))
    finally:
        seen = set()
        for owner in owners:
            if id(owner) not in seen:
                owner.release()
                seen.add(id(owner))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
