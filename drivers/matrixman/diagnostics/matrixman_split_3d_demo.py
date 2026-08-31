#!/usr/bin/env python3
"""Deterministic MatrixMan 3D split test for the YOLO DFL decoder."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported, storage_identity


def main() -> int:
    set_trace_if_supported(gm45, True)
    cpu = torch.arange(1 * 4 * 84, dtype=torch.float32).reshape(1, 4, 84)
    x = gm45.to_device(cpu)
    left, right = torch.chunk(x, 2, dim=1)
    left_cpu = left.cpu()
    right_cpu = right.cpu()
    expected_left, expected_right = torch.chunk(cpu, 2, dim=1)

    print("YOLO DFL 3D split/chunk")
    print(f"  input: shape={list(x.shape)} {describe_storage(x)}")
    for name, value, expected in [("left", left, expected_left), ("right", right, expected_right)]:
        value_cpu = value.cpu()
        error = (value_cpu - expected).abs().max().item()
        print(f"  {name}: shape={list(value.shape)} {describe_storage(value)}")
        print(f"    max_abs_error: {error:.6g}")
        print(f"    allclose: {torch.allclose(value_cpu, expected)}")
    print("  storage reused:", storage_identity(x) == storage_identity(left) == storage_identity(right))
    print("  GPU->CPU readback: only explicit validation calls")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
