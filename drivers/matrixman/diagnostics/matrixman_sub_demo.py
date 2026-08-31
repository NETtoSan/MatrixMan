#!/usr/bin/env python3
"""Deterministic MatrixMan stride-aware subtraction test for YOLO decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, readback_tensor, set_trace_if_supported


def main() -> int:
    set_trace_if_supported(gm45, True)
    torch.manual_seed(20260829)
    anchor_base_cpu = torch.randn((84, 2), dtype=torch.float32)
    anchor_cpu = anchor_base_cpu.transpose(0, 1).unsqueeze(0)
    distance_cpu = torch.randn((1, 2, 84), dtype=torch.float32)

    anchor = gm45.to_device(anchor_base_cpu).transpose(0, 1).unsqueeze(0)
    distance = gm45.to_device(distance_cpu)
    output = torch.sub(anchor, distance)
    output_cpu = readback_tensor(output)
    expected = anchor_cpu - distance_cpu
    error = (output_cpu - expected).abs().max().item()

    print("MatrixMan stride-aware packed subtraction")
    print(f"  left:  shape={list(anchor.shape)} {describe_storage(anchor)}")
    print(f"  right: shape={list(distance.shape)} {describe_storage(distance)}")
    print(f"  output: shape={list(output.shape)} {describe_storage(output)}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
