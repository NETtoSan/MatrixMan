#!/usr/bin/env python3
"""Deterministic GM45 stride-aware subtraction test for YOLO decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    torch.manual_seed(20260829)
    anchor_base_cpu = torch.randn((84, 2), dtype=torch.float32)
    anchor_cpu = anchor_base_cpu.transpose(0, 1).unsqueeze(0)
    distance_cpu = torch.randn((1, 2, 84), dtype=torch.float32)

    anchor = gm45.to_device(anchor_base_cpu).transpose(0, 1).unsqueeze(0)
    distance = gm45.to_device(distance_cpu)
    output = torch.sub(anchor, distance)
    output_cpu = output.cpu()
    expected = anchor_cpu - distance_cpu
    error = (output_cpu - expected).abs().max().item()

    print("GM45 stride-aware packed subtraction")
    print(f"  left:  shape={list(anchor.shape)} strides={anchor._logical_strides} texture=#{anchor._owner.texture} offset={anchor._storage_offset}")
    print(f"  right: shape={list(distance.shape)} strides={distance._logical_strides} texture=#{distance._owner.texture} offset={distance._storage_offset}")
    print(f"  output: shape={list(output.shape)} strides={output._logical_strides} texture=#{output._owner.texture} offset={output._storage_offset}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
