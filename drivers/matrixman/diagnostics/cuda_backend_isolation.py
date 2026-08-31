#!/usr/bin/env python3
"""Regression check that CUDA selection does not activate OpenGL factories."""

import sys

import torch

from drivers.matrixman.backend import get_backend
from drivers.matrixman.privateuse import register_privateuse1_backend
from drivers.matrixman.tensor import PRIVATEUSE_DEVICE
from drivers.matrixman.backends.cuda.backend import CudaTensorOwner
from drivers.matrixman.tensor import MatrixManTensor


def run_diagnostic() -> int:
    backend = get_backend()
    if backend.name != "cuda":
        raise RuntimeError(
            "CUDA isolation diagnostic requires MATRIXMAN_BACKEND=cuda and an available CUDA device"
        )
    register_privateuse1_backend()
    if torch._C._get_privateuse1_backend_name() != "matrixman":
        raise AssertionError("MatrixMan did not register the canonical PrivateUse1 name")
    try:
        canonical_device = torch.device("matrixman:0")
    except RuntimeError as exc:
        raise AssertionError("MatrixMan device name matrixman:0 is not accepted") from exc
    if str(canonical_device) != "matrixman:0":
        raise AssertionError(f"unexpected MatrixMan device spelling: {canonical_device}")

    imported_opengl = sorted(
        name for name in sys.modules if name.startswith("drivers.matrixman.backends.opengl")
    )
    if imported_opengl:
        raise AssertionError(f"CUDA selection imported OpenGL modules: {imported_opengl}")

    empty = torch.empty((2, 3), dtype=torch.float32, device=PRIVATEUSE_DEVICE)
    empty_strided = torch.empty_strided(
        (2, 3), (3, 1), dtype=torch.float32, device=PRIVATEUSE_DEVICE
    )
    opaque = torch.empty((8,), dtype=torch.uint8, device=PRIVATEUSE_DEVICE)
    opaque_strided = torch.empty_strided(
        (2, 3), (3, 1), dtype=torch.uint8, device=PRIVATEUSE_DEVICE
    )
    for label, value in (
        ("torch.empty", empty),
        ("torch.empty_strided", empty_strided),
        ("torch.empty uint8", opaque),
        ("torch.empty_strided uint8", opaque_strided),
    ):
        if not isinstance(value, MatrixManTensor):
            raise AssertionError(f"{label} did not return MatrixManTensor")
        if not isinstance(value._owner, CudaTensorOwner):
            raise AssertionError(f"{label} did not return a CUDA owner")
        if value.device != PRIVATEUSE_DEVICE:
            raise AssertionError(f"{label} returned unexpected device {value.device}")
        if not value._owner.pointer.value:
            raise AssertionError(f"{label} did not create a CUDA allocation")
        expected_dtype = torch.uint8 if "uint8" in label else torch.float32
        if value.dtype != expected_dtype:
            raise AssertionError(f"{label} returned unexpected dtype {value.dtype}")
        value._owner.release()

    print("CUDA selected without importing OpenGL modules")
    print("CUDA factories created CUDA-backed MatrixMan tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
