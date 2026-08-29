#!/usr/bin/env python3
"""Deterministic regressions for the traced GM45 grouped Conv2D families."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    torch.manual_seed(20260829)
    cases = [
        ((1, 8, 8, 8), (16, 1, 3, 3), 8, "depthwise multiplier"),
        ((1, 16, 16, 16), (24, 2, 3, 3), 8, "general grouped"),
    ]
    print("GM45 grouped convolution regressions")
    for input_shape, weight_shape, groups, label in cases:
        input_cpu = torch.randn(input_shape, dtype=torch.float32)
        weight = torch.randn(weight_shape, dtype=torch.float32)
        input_gpu = gm45.to_gm45(input_cpu)
        output = F.conv2d(input_gpu, weight, None, stride=2, padding=1, groups=groups)
        output_cpu = output.cpu()
        expected = F.conv2d(input_cpu, weight, None, stride=2, padding=1, groups=groups)
        error = (output_cpu - expected).abs().max().item()
        print(f"  {label}: input={list(input_shape)} weight={list(weight_shape)} groups={groups}")
        print(f"    output: shape={list(output.shape)} strides={output._logical_strides} texture=#{output._owner.texture}")
        print(f"    max_abs_error: {error:.6g}")
        print(f"    allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
