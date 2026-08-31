#!/usr/bin/env python3
"""Deterministic packed NCHW channel-cat tests for the MatrixMan backend."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported


def report_cat(name: str, inputs, y, expected: torch.Tensor) -> None:
    y_cpu = y.cpu()
    error = (y_cpu - expected).abs().max().item()
    print(name)
    for index, tensor in enumerate(inputs):
        print(f"  input {index}: shape={list(tensor.shape)} {describe_storage(tensor)}")
    print(f"  output shape: {list(y.shape)}")
    print(f"  output storage: {describe_storage(y)}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print MatrixMan dispatch/kernel trace")
    args = parser.parse_args()

    set_trace_if_supported(gm45, args.trace)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)

    x_cpu = torch.randn((1, 32, 16, 16), dtype=torch.float32, generator=generator)
    x = gm45.to_device(x_cpu)
    left, right = torch.split(x, 16, dim=1)
    residual = torch.add(left, right)
    y = torch.cat([left, right, residual], dim=1)
    expected = torch.cat([x_cpu[:, 0:16], x_cpu[:, 16:32], x_cpu[:, 0:16] + x_cpu[:, 16:32]], dim=1)
    report_cat("Three-input YOLO-style cat with split offset", [left, right, residual], y, expected)

    four_cpu = [torch.randn((1, 32, 8, 8), dtype=torch.float32, generator=generator) for _ in range(4)]
    four_gpu = [gm45.to_device(tensor) for tensor in four_cpu]
    four_y = torch.cat(four_gpu, dim=1)
    report_cat("Four-input YOLO-style channel cat", four_gpu, four_y, torch.cat(four_cpu, dim=1))

    stride_cpu = [
        torch.full((64, 1), 8.0, dtype=torch.float32),
        torch.full((16, 1), 16.0, dtype=torch.float32),
        torch.full((4, 1), 32.0, dtype=torch.float32),
    ]
    gm45_device = gm45.to_device(torch.zeros((1,), dtype=torch.float32)).device
    stride_gpu = [
        torch.full(tensor.shape, float(tensor[0, 0]), device=gm45_device, dtype=torch.float32)
        for tensor in stride_cpu
    ]
    stride_y = torch.cat(stride_gpu, dim=0)
    report_cat("YOLO stride tensor 2D dim-0 cat", stride_gpu, stride_y, torch.cat(stride_cpu, dim=0))

    anchors_cpu = [
        torch.arange(64 * 2, dtype=torch.float32).reshape(64, 2),
        1000.0 + torch.arange(16 * 2, dtype=torch.float32).reshape(16, 2),
        2000.0 + torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
    ]
    anchors_gpu = [gm45.to_device(tensor) for tensor in anchors_cpu]
    anchors_y = torch.cat(anchors_gpu, dim=0)
    report_cat("YOLO anchor_points 2D dim-0 cat", anchors_gpu, anchors_y, torch.cat(anchors_cpu, dim=0))

    weight = torch.randn((8, 48, 1, 1), dtype=torch.float32, generator=generator)
    conv = torch.ops.aten.convolution.default(y, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    conv_expected = F.conv2d(expected, weight)
    conv_cpu = conv.cpu()
    conv_error = (conv_cpu - conv_expected).abs().max().item()
    print("Chained split -> residual add -> cat -> Conv2D")
    print(f"  cat output storage: {describe_storage(y)}")
    print(f"  conv output shape: {list(conv.shape)}")
    print(f"  conv output storage: {describe_storage(conv)}")
    print(f"  max_abs_error: {conv_error:.6g}")
    print(f"  allclose: {torch.allclose(conv_cpu, conv_expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit conv.cpu() validation")


if __name__ == "__main__":
    main()
