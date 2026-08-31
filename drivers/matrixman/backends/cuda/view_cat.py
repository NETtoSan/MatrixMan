#!/usr/bin/env python3
"""CUDA diagnostic for contiguous metadata-only views followed by cat."""

from __future__ import annotations

import numpy as np
import torch

from drivers import matrixman
from drivers.matrixman.backend import get_backend
from drivers.matrixman.tensor import contiguous_strides


def describe(label, tensor) -> None:
    expected = contiguous_strides(tuple(int(value) for value in tensor.shape))
    is_contiguous = tuple(tensor._logical_strides) == expected
    print(
        f"{label}: shape={list(tensor.shape)} strides={list(tensor._logical_strides)} "
        f"offset={tensor._storage_offset} owner_kind={tensor._owner.layout.kind} "
        f"dtype={tensor.dtype} pointer={tensor._owner.pointer.value} "
        f"matrixman_contiguous={is_contiguous}"
    )
    if not is_contiguous:
        raise AssertionError(f"{label} is not logically contiguous")


def run_diagnostic() -> int:
    backend = get_backend()
    if backend.name != "cuda":
        raise RuntimeError("view -> cat diagnostic requires CUDA selection")

    left_cpu = np.arange(1, 1 + 1 * 64 * 2 * 3, dtype=np.float32).reshape(1, 64, 2, 3)
    right_cpu = np.arange(1000, 1000 + 1 * 64 * 1 * 4, dtype=np.float32).reshape(1, 64, 1, 4)
    left = matrixman.to_device(torch.from_numpy(left_cpu))
    right = matrixman.to_device(torch.from_numpy(right_cpu))
    left_view = left.view(1, 64, -1)
    right_view = right.view(1, 64, -1)

    describe("left view", left_view)
    describe("right view", right_view)
    output = torch.cat((left_view, right_view), dim=-1)
    describe("cat output", output)

    expected = np.concatenate(
        (left_cpu.reshape(1, 64, 6), right_cpu.reshape(1, 64, 4)),
        axis=-1,
    )
    actual = backend.execution.from_device(output._owner.pointer, tuple(output.shape))
    max_abs = float(np.max(np.abs(actual - expected)))
    print("output max abs error:", max_abs)
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        raise RuntimeError("CUDA view -> cat result does not match NumPy reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
