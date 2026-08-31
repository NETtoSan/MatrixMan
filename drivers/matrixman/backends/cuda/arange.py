#!/usr/bin/env python3
"""Standalone CUDA ``arange`` diagnostic."""

import sys

import numpy as np
import torch

from ...tensor import PRIVATEUSE_DEVICE, MatrixManTensor
from .backend import CudaBackend, CudaTensorOwner


def _check(name, tensor, expected, cuda):
    if not isinstance(tensor, MatrixManTensor):
        raise AssertionError(f"{name}: result is not a MatrixManTensor")
    if not isinstance(tensor._owner, CudaTensorOwner):
        raise AssertionError(f"{name}: result does not own CUDA storage")
    if tensor.device != PRIVATEUSE_DEVICE:
        raise AssertionError(f"{name}: unexpected device {tensor.device}")
    if tensor.dtype != torch.float32 or tuple(tensor.shape) != expected.shape:
        raise AssertionError(f"{name}: unexpected metadata {tensor.shape}, {tensor.dtype}")
    actual = cuda.from_device(tensor._owner.pointer, expected.shape)
    difference = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
          f"owner={type(tensor._owner).__name__}, max abs diff={difference}")
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"{name}: result mismatch")


def run_diagnostic() -> int:
    from ...backend import get_backend

    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the arange diagnostic")
    cuda = backend.execution
    print("CUDA device:", cuda.info["name"])
    print("compute capability:", cuda.info["compute_capability"])
    cases = (
        ("arange(5)", torch.arange(5, device=PRIVATEUSE_DEVICE, dtype=torch.float32), np.arange(5, dtype=np.float32)),
        ("arange(2, 7)", torch.arange(2, 7, device=PRIVATEUSE_DEVICE, dtype=torch.float32), np.arange(2, 7, dtype=np.float32)),
        ("arange(1, 6, 0.5)", torch.arange(1, 6, 0.5, device=PRIVATEUSE_DEVICE, dtype=torch.float32), np.arange(1, 6, 0.5, dtype=np.float32)),
    )
    for name, result, expected in cases:
        try:
            _check(name, result, expected, cuda)
        finally:
            result._owner.release()

    def check_add(name, left, right, expected):
        result = left + right
        try:
            _check(name, result, expected, cuda)
        finally:
            result._owner.release()

    values = torch.arange(5, device=PRIVATEUSE_DEVICE, dtype=torch.float32)
    try:
        check_add("tensor + 0.5", values, 0.5, np.arange(5, dtype=np.float32) + 0.5)
        check_add("0.5 + tensor", 0.5, values, np.arange(5, dtype=np.float32) + 0.5)
        check_add("tensor + (-2.0)", values, -2.0, np.arange(5, dtype=np.float32) - 2.0)
    finally:
        values._owner.release()

    result = torch.arange(end=7, device=PRIVATEUSE_DEVICE, dtype=torch.float32) + 0.5
    expected = np.arange(7, dtype=np.float32) + 0.5
    try:
        _check("YOLO arange(end=7) + 0.5", result, expected, cuda)
    finally:
        result._owner.release()
    return 0


if __name__ == "__main__":
    sys.exit(run_diagnostic())
