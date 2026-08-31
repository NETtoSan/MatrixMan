#!/usr/bin/env python3
"""Deterministic YOLO-subset max_pool2d tests for the GM45 backend."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45


def report_pool(name: str, x, y, expected: torch.Tensor) -> None:
    y_cpu = y.cpu()
    error = (y_cpu - expected).abs().max().item()
    print(name)
    print(f"  input shape: {list(x.shape)} texture=#{x._owner.texture} offset={x._storage_offset}")
    print(f"  output shape: {list(y.shape)} texture=#{y._owner.texture} offset={y._storage_offset}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print gm45 dispatch/kernel trace")
    args = parser.parse_args()

    gm45.set_trace(args.trace)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)

    # Negative-biased values verify that out-of-bounds padding is treated as
    # negative infinity, not zero.
    x_cpu = torch.randn((1, 128, 2, 2), dtype=torch.float32, generator=generator) - 5.0
    x = gm45.to_device(x_cpu)
    y = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
    expected = F.max_pool2d(x_cpu, kernel_size=5, stride=1, padding=2)
    report_pool("YOLO SPPF max_pool2d values", x, y, expected)

    direct_values, direct_indices = torch.ops.aten.max_pool2d_with_indices.default(
        x, [5, 5], [1, 1], [2, 2], [1, 1], False
    )
    direct_cpu = direct_values.cpu()
    direct_error = (direct_cpu - expected).abs().max().item()
    print("Direct aten.max_pool2d_with_indices schema result")
    print(f"  values shape: {list(direct_values.shape)} texture=#{direct_values._owner.texture}")
    print(f"  indices placeholder shape: {list(direct_indices.shape)} dtype={direct_indices.dtype} device={direct_indices.device}")
    print("  indices consumed by YOLO path: no")
    print(f"  values max_abs_error: {direct_error:.6g}")
    print(f"  values allclose: {torch.allclose(direct_cpu, expected, atol=1e-5, rtol=1e-5)}")

    split_cpu = torch.randn((1, 256, 2, 2), dtype=torch.float32, generator=generator) - 3.0
    split_source = gm45.to_device(split_cpu)
    _, right = torch.split(split_source, 128, dim=1)
    split_pool = F.max_pool2d(right, kernel_size=5, stride=1, padding=2)
    split_expected = F.max_pool2d(split_cpu[:, 128:256], kernel_size=5, stride=1, padding=2)
    report_pool("Split-derived nonzero-offset max_pool2d values", right, split_pool, split_expected)

    weight = torch.randn((16, 128, 1, 1), dtype=torch.float32, generator=generator)
    conv = torch.ops.aten.convolution.default(split_pool, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    conv_expected = F.conv2d(split_expected, weight)
    conv_cpu = conv.cpu()
    conv_error = (conv_cpu - conv_expected).abs().max().item()
    print("Chained split -> max_pool2d -> Conv2D")
    print(f"  pool output texture: #{split_pool._owner.texture} offset={split_pool._storage_offset}")
    print(f"  conv output shape: {list(conv.shape)} texture=#{conv._owner.texture} offset={conv._storage_offset}")
    print(f"  max_abs_error: {conv_error:.6g}")
    print(f"  allclose: {torch.allclose(conv_cpu, conv_expected, atol=2e-4, rtol=1e-4)}")
    print("  GPU->CPU readback: only explicit conv.cpu() validation")


if __name__ == "__main__":
    main()
