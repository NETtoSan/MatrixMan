#!/usr/bin/env python3
"""Deterministic GM45 rank-3 channel-cat test for YOLO box decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    left_cpu = torch.arange(168, dtype=torch.float32).reshape(1, 2, 84)
    right_cpu = 1000.0 + torch.arange(168, dtype=torch.float32).reshape(1, 2, 84)
    left = gm45.to_device(left_cpu)
    right = gm45.to_device(right_cpu)
    output = torch.cat([left, right], dim=1)
    output_cpu = output.cpu()
    expected = torch.cat([left_cpu, right_cpu], dim=1)
    error = (output_cpu - expected).abs().max().item()

    print("GM45 rank-3 dim=1 cat")
    print(f"  left:   shape={list(left.shape)} texture=#{left._owner.texture} offset={left._storage_offset}")
    print(f"  right:  shape={list(right.shape)} texture=#{right._owner.texture} offset={right._storage_offset}")
    print(f"  output: shape={list(output.shape)} strides={output._logical_strides} texture=#{output._owner.texture} offset={output._storage_offset}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(output_cpu, expected)}")
    print("  new output texture:", output._owner.texture not in {left._owner.texture, right._owner.texture})
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
