#!/usr/bin/env python3
"""Deterministic MatrixMan stride-aware packed addition test for YOLO decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported


def main() -> int:
    set_trace_if_supported(gm45, True)
    torch.manual_seed(20260829)
    anchor_base_cpu = torch.randn((84, 2), dtype=torch.float32)
    anchor_cpu = anchor_base_cpu.transpose(0, 1).unsqueeze(0)
    rb_cpu = torch.randn((1, 2, 84), dtype=torch.float32)

    anchor = gm45.to_device(anchor_base_cpu).transpose(0, 1).unsqueeze(0)
    rb = gm45.to_device(rb_cpu)
    output = torch.add(anchor, rb, alpha=1.0)
    output_cpu = output.cpu()
    expected = anchor_cpu + rb_cpu
    error = (output_cpu - expected).abs().max().item()

    print("MatrixMan stride-aware packed addition")
    print(f"  left:  shape={list(anchor.shape)} {describe_storage(anchor)}")
    print(f"  right: shape={list(rb.shape)} {describe_storage(rb)}")
    print(f"  output: shape={list(output.shape)} {describe_storage(output)}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
