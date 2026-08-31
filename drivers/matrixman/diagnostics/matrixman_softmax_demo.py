#!/usr/bin/env python3
"""Deterministic MatrixMan DFL softmax test for the traced YOLO shape."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported


def main() -> int:
    set_trace_if_supported(gm45, True)
    torch.manual_seed(20260829)
    base_cpu = torch.randn((1, 4, 16, 84), dtype=torch.float32)
    input_cpu = base_cpu.transpose(1, 2)
    input_gpu = gm45.to_device(base_cpu).transpose(1, 2)
    output_gpu = torch.ops.aten._softmax.default(input_gpu, 1, False)
    output_cpu = output_gpu.cpu()
    expected = torch.softmax(input_cpu, dim=1)
    max_error = (output_cpu - expected).abs().max().item()

    print("MatrixMan DFL softmax dim=1")
    print(f"  input shape={list(input_gpu.shape)} {describe_storage(input_gpu)}")
    print(f"  output shape={list(output_gpu.shape)} {describe_storage(output_gpu)}")
    print(f"  max_abs_error: {max_error:.6g}")
    print(f"  allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
