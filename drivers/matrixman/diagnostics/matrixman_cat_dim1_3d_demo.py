#!/usr/bin/env python3
"""Deterministic MatrixMan rank-3 channel-cat test for YOLO box decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported, storage_identity


def main() -> int:
    set_trace_if_supported(gm45, True)
    left_cpu = torch.arange(168, dtype=torch.float32).reshape(1, 2, 84)
    right_cpu = 1000.0 + torch.arange(168, dtype=torch.float32).reshape(1, 2, 84)
    left = gm45.to_device(left_cpu)
    right = gm45.to_device(right_cpu)
    output = torch.cat([left, right], dim=1)
    output_cpu = output.cpu()
    expected = torch.cat([left_cpu, right_cpu], dim=1)
    error = (output_cpu - expected).abs().max().item()

    print("MatrixMan rank-3 dim=1 cat")
    print(f"  left:   shape={list(left.shape)} {describe_storage(left)}")
    print(f"  right:  shape={list(right.shape)} {describe_storage(right)}")
    print(f"  output: shape={list(output.shape)} {describe_storage(output)}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(output_cpu, expected)}")
    print("  new output storage:", storage_identity(output) not in {storage_identity(left), storage_identity(right)})
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
