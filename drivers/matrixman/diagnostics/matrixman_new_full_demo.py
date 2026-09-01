#!/usr/bin/env python3
"""Focused OpenGL new_full allocation and scalar-fill diagnostic."""

from __future__ import annotations

import torch

from drivers import matrixman
from drivers.matrixman.backend import get_backend
from drivers.matrixman.diagnostics.backend_helpers import readback_tensor


def main() -> int:
    backend = get_backend()
    if backend.name != "opengl":
        raise RuntimeError("matrixman_new_full_demo requires the OpenGL backend")

    source_cpu = torch.empty((2,), dtype=torch.float32)
    source = matrixman.to_device(source_cpu)
    expected = source_cpu.new_full((5,), 0.5)
    result = source.new_full((5,), 0.5)
    actual = readback_tensor(result)

    print("aten.new_full.default schema:", torch.ops.aten.new_full.default._schema)
    print("result device:", result.device)
    print("result dtype:", result.dtype)
    print("result strides:", tuple(result._logical_strides))
    print("readback:", actual.tolist())
    print("allclose:", torch.allclose(actual, expected))
    if result.dtype != torch.float32 or not torch.allclose(actual, expected):
        raise AssertionError("MatrixMan new_full result does not match CPU")

    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
