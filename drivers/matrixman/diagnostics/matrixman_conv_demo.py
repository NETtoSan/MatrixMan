#!/usr/bin/env python3
"""
Correctness tests for the first MatrixMan Conv2D backend implementation.

The tested operation is aten.convolution.default via torch.nn.functional.conv2d.
Input and output tensors remain MatrixMan-backed until the explicit .cpu()
validation readback.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported


def run_case(shape, weight_shape, stride, padding, bias: bool) -> None:
    torch.manual_seed(1000 + shape[1] + weight_shape[0] + stride * 10 + padding)
    x_cpu = torch.randn(shape, dtype=torch.float32) * 0.25
    w_cpu = torch.randn(weight_shape, dtype=torch.float32) * 0.25
    b_cpu = torch.randn((weight_shape[0],), dtype=torch.float32) * 0.1 if bias else None

    x = gm45.to_device(x_cpu)
    before_report = gm45.unsupported_report()
    y = F.conv2d(x, w_cpu, b_cpu, stride=stride, padding=padding)
    readback_report = gm45.unsupported_report()
    y_cpu = y.cpu()
    ref = F.conv2d(x_cpu, w_cpu, b_cpu, stride=stride, padding=padding)

    max_error = (y_cpu - ref).abs().max().item()
    print(f"case input={list(shape)} weight={list(weight_shape)} stride={stride} padding={padding} bias={bias}")
    print(f"  output shape: {list(y.shape)}")
    print(f"  input storage: {describe_storage(x)}")
    print(f"  output storage: {describe_storage(y)}")
    print(f"  max_abs_error: {max_error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, ref, rtol=5e-4, atol=5e-4)}")
    print(f"  unsupported report changed during op: {before_report != readback_report}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")


def main() -> int:
    parser = argparse.ArgumentParser(description="MatrixMan Conv2D correctness tests")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    set_trace_if_supported(gm45, args.trace)
    gm45.reset_unsupported_report()

    # YOLO first layer subset: [1,3,H,W], [16,3,3,3], stride 2, pad 1, no bias.
    run_case((1, 3, 8, 8), (4, 3, 3, 3), stride=2, padding=1, bias=False)
    # YOLO common internal 3x3 stride-1 padded conv.
    run_case((1, 4, 8, 8), (5, 4, 3, 3), stride=1, padding=1, bias=False)
    # YOLO common 1x1 conv.
    run_case((1, 4, 8, 8), (6, 4, 1, 1), stride=1, padding=0, bias=False)
    # YOLO detection-head convs include bias.
    run_case((1, 4, 8, 8), (6, 4, 1, 1), stride=1, padding=0, bias=True)

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
