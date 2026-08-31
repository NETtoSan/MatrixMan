#!/usr/bin/env python3
"""Deterministic packed NCHW add tests for the experimental MatrixMan backend."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import (
    describe_storage,
    readback_tensor,
    set_trace_if_supported,
)


def report_case(name: str, y, expected: torch.Tensor, inputs=(), readback_note: str = "explicit backend-aware readback") -> None:
    y_cpu = readback_tensor(y)
    error = (y_cpu - expected).abs().max().item()
    print(name)
    print(f"  input shapes: {[list(value.shape) for value in inputs]}")
    print(f"  input strides: {[list(value._logical_strides) for value in inputs]}")
    print(f"  output shape: {list(y.shape)}")
    print(f"  output strides: {list(y._logical_strides)}")
    print(f"  output storage: {describe_storage(y)}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print(f"  GPU->CPU readback: {readback_note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print MatrixMan dispatch/kernel trace")
    args = parser.parse_args()

    set_trace_if_supported(gm45, args.trace)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)

    a_cpu = torch.randn((1, 16, 16, 16), dtype=torch.float32, generator=generator)
    b_cpu = torch.randn((1, 16, 16, 16), dtype=torch.float32, generator=generator)
    a = gm45.to_device(a_cpu)
    b = gm45.to_device(b_cpu)
    y = torch.add(a, b, alpha=0.5)
    print("Normal packed NCHW add")
    print(f"  left storage: {describe_storage(a)}")
    print(f"  right storage: {describe_storage(b)}")
    report_case("  result", y, a_cpu + 0.5 * b_cpu, (a, b))

    x_cpu = torch.randn((1, 32, 16, 16), dtype=torch.float32, generator=generator)
    x = gm45.to_device(x_cpu)
    left, right = torch.split(x, 16, dim=1)
    split_sum = torch.add(left, right)
    print("\nSplit-derived nonzero-offset add")
    print(f"  source storage: {describe_storage(x)}")
    print(f"  left storage: {describe_storage(left)}")
    print(f"  right storage: {describe_storage(right)}")
    report_case("  result", split_sum, x_cpu[:, 0:16] + x_cpu[:, 16:32], (left, right))

    weight = torch.randn((16, 16, 1, 1), dtype=torch.float32, generator=generator)
    conv = torch.ops.aten.convolution.default(right, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    chained = torch.add(left, conv)
    expected = x_cpu[:, 0:16] + F.conv2d(x_cpu[:, 16:32], weight)
    print("\nChained split -> Conv2D -> add")
    print(f"  split left storage: {describe_storage(left)}")
    print(f"  split right storage: {describe_storage(right)}")
    print(f"  conv output storage: {describe_storage(conv)}")
    report_case("  result", chained, expected, (left, conv))


if __name__ == "__main__":
    main()
