#!/usr/bin/env python3
"""Frontend diagnostic for selected-backend ``aten.mm.default``."""

from __future__ import annotations

import torch

from drivers import matrixman
from drivers.matrixman.backend import get_backend
from drivers.matrixman.diagnostics.backend_helpers import readback_tensor


def main() -> int:
    backend = get_backend()
    print("selected backend:", backend.name)
    print("aten.mm.default schema:", torch.ops.aten.mm.default._schema)
    for left_shape, right_shape in (((4, 4), (4, 4)), ((32, 64), (64, 48)), ((256, 256), (256, 256))):
        left_cpu = torch.randn(left_shape, dtype=torch.float32)
        right_cpu = torch.randn(right_shape, dtype=torch.float32)
        left = matrixman.to_device(left_cpu)
        right = matrixman.to_device(right_cpu)
        result = torch.mm(left, right)
        actual = readback_tensor(result)
        reference = torch.mm(left_cpu, right_cpu)
        error = float((actual - reference).abs().max())
        close = torch.allclose(actual, reference, rtol=5e-4, atol=5e-4)
        print(
            f"mm {list(left_shape)} @ {list(right_shape)} -> {list(result.shape)}: "
            f"max_abs_diff={error:.6g} allclose={close} "
            f"output_strides={list(result._logical_strides)}"
        )
        if not close:
            raise AssertionError("MatrixMan mm result does not match CPU reference")
    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
