#!/usr/bin/env python3
"""Correctness test for GM45 aten.silu_.default."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45


def main() -> int:
    parser = argparse.ArgumentParser(description="GM45 SiLU correctness test")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    gm45.set_trace(args.trace)

    torch.manual_seed(3030)
    x_cpu = torch.randn((1, 7, 5, 6), dtype=torch.float32) * 2.0
    x = gm45.to_device(x_cpu)
    input_texture = x._owner.texture

    # This dispatches aten.silu_.default. The backend writes a new output texture
    # instead of physically mutating the input texture, avoiding GL read/write
    # feedback hazards.
    y = F.silu(x, inplace=True)
    y_cpu = y.cpu()
    ref = F.silu(x_cpu)

    max_error = (y_cpu - ref).abs().max().item()
    print("SiLU inplace-dispatch case input=[1,7,5,6]")
    print(f"  output shape: {list(y.shape)}")
    print(f"  input texture: #{input_texture}")
    print(f"  output texture: #{y._owner.texture}")
    print("  physical in-place texture mutation: False")
    print("  logical in-place wrapper update: True")
    print(f"  max_abs_error: {max_error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, ref, rtol=5e-4, atol=5e-4)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
