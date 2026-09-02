#!/usr/bin/env python3
"""
Correctness tests for the first MatrixMan Conv2D backend implementation.

The tested operation is aten.convolution.default via torch.nn.functional.conv2d.
Input and output tensors remain MatrixMan-backed until the explicit .cpu()
validation readback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import (
    describe_storage,
    reset_unsupported_report_if_supported,
    set_trace_if_supported,
    unsupported_report_if_supported,
)


def _save_channel_image(tensor: torch.Tensor, output_dir: Path, channel: int, name: str) -> Path:
    """Save one logical NCHW output channel as a viewable grayscale image."""
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("pixel visualization requires a batch-1 NCHW tensor")
    if channel < 0 or channel >= tensor.shape[1]:
        raise ValueError(f"--channel must be in [0, {tensor.shape[1] - 1}]")
    pixels = tensor[0, channel].detach().cpu().numpy().astype(np.float32, copy=False)
    low, high = float(pixels.min()), float(pixels.max())
    if high > low:
        image = ((pixels - low) * (255.0 / (high - low))).clip(0, 255).astype(np.uint8)
    else:
        image = np.zeros(pixels.shape, dtype=np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_channel{channel}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not write visualization image: {path}")
    return path


def run_case(shape, weight_shape, stride, padding, bias: bool, visualize_dir: Path | None = None, channel: int = 0) -> None:
    torch.manual_seed(1000 + shape[1] + weight_shape[0] + stride * 10 + padding)
    x_cpu = torch.randn(shape, dtype=torch.float32) * 0.25
    w_cpu = torch.randn(weight_shape, dtype=torch.float32) * 0.25
    b_cpu = torch.randn((weight_shape[0],), dtype=torch.float32) * 0.1 if bias else None

    x = gm45.to_device(x_cpu) # Turn into MatrixMan's tensor

    before_report = unsupported_report_if_supported(gm45)
    y = F.conv2d(x, w_cpu, b_cpu, stride=stride, padding=padding)

    readback_report = unsupported_report_if_supported(gm45)
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
    if channel < 0 or channel >= y_cpu.shape[1]:
        raise ValueError(f"--channel must be in [0, {y_cpu.shape[1] - 1}]")
    print(f"  output channel {channel} pixels:\n{y_cpu[0, channel]}")

    if visualize_dir is not None:
        case_name = (
            f"conv_in{shape[1]}c_{shape[2]}x{shape[3]}"
            f"_out{weight_shape[0]}_k{weight_shape[2]}x{weight_shape[3]}"
            f"_s{stride}_p{padding}_bias{int(bias)}"
        )
        path = _save_channel_image(y_cpu, visualize_dir, channel, case_name)
        print(f"  pixel visualization: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MatrixMan Conv2D correctness tests")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--visualize", type=Path, metavar="DIR", help="save selected output feature maps as PNG images")
    parser.add_argument("--channel", type=int, default=0, help="logical output channel to visualize (default: 0)")
    args = parser.parse_args()
    set_trace_if_supported(gm45, args.trace)
    reset_unsupported_report_if_supported(gm45)

    # YOLO first layer subset: [1,3,H,W], [16,3,3,3], stride 2, pad 1, no bias.
    run_case((1, 3, 320, 320), (4, 3, 3, 3), stride=2, padding=1, bias=False, visualize_dir=args.visualize, channel=args.channel)
    # YOLO common internal 3x3 stride-1 padded conv.
    run_case((1, 4, 320, 320), (5, 4, 3, 3), stride=1, padding=1, bias=False, visualize_dir=args.visualize, channel=args.channel)
    # YOLO common 1x1 conv.
    run_case((1, 4, 320, 320), (6, 4, 1, 1), stride=1, padding=0, bias=False, visualize_dir=args.visualize, channel=args.channel)
    # YOLO detection-head convs include bias.
    run_case((1, 4, 320, 320), (6, 4, 1, 1), stride=1, padding=0, bias=True, visualize_dir=args.visualize, channel=args.channel)

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
