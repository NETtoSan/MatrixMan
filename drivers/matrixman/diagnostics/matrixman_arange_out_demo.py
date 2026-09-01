#!/usr/bin/env python3
"""Focused OpenGL arange.out allocation and readback diagnostic."""

from __future__ import annotations

import torch

from drivers import matrixman
from drivers.matrixman.backend import get_backend
from drivers.matrixman.diagnostics.backend_helpers import readback_tensor


def main() -> int:
    if get_backend().name != "opengl":
        raise RuntimeError("matrixman_arange_out_demo requires the OpenGL backend")
    source = matrixman.to_device(torch.empty((1,), dtype=torch.float32))
    out = source.new_full((5,), 0)
    result = torch.arange(5, out=out)
    actual = readback_tensor(result)
    print("aten.arange.out schema:", torch.ops.aten.arange.out._schema)
    print("result is out:", result is out)
    print("result device:", result.device)
    print("result dtype:", result.dtype)
    print("readback:", actual.tolist())
    if result is not out or str(result.device) not in {"matrixman", "matrixman:0", "privateuseone", "privateuseone:0"} or result.dtype != torch.float32:
        raise AssertionError("MatrixMan arange.out returned incorrect output metadata")
    if not torch.equal(actual, torch.arange(5, dtype=torch.float32)):
        raise AssertionError("MatrixMan arange.out result does not match CPU")
    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
