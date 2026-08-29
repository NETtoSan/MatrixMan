#!/usr/bin/env python3
"""Deterministic packed NCHW add tests for the experimental GM45 backend."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45


def report_case(name: str, y, expected: torch.Tensor, readback_note: str = "only explicit y.cpu() validation") -> None:
    y_cpu = y.cpu()
    error = (y_cpu - expected).abs().max().item()
    print(name)
    print(f"  output shape: {list(y.shape)}")
    print(f"  output texture: #{y._owner.texture} offset={y._storage_offset}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print(f"  GPU->CPU readback: {readback_note}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print gm45 dispatch/kernel trace")
    args = parser.parse_args()

    gm45.set_trace(args.trace)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)

    a_cpu = torch.randn((1, 16, 16, 16), dtype=torch.float32, generator=generator)
    b_cpu = torch.randn((1, 16, 16, 16), dtype=torch.float32, generator=generator)
    a = gm45.to_gm45(a_cpu)
    b = gm45.to_gm45(b_cpu)
    y = torch.add(a, b, alpha=0.5)
    print("Normal packed NCHW add")
    print(f"  left texture: #{a._owner.texture} offset={a._storage_offset}")
    print(f"  right texture: #{b._owner.texture} offset={b._storage_offset}")
    report_case("  result", y, a_cpu + 0.5 * b_cpu)

    x_cpu = torch.randn((1, 32, 16, 16), dtype=torch.float32, generator=generator)
    x = gm45.to_gm45(x_cpu)
    left, right = torch.split(x, 16, dim=1)
    split_sum = torch.add(left, right)
    print("\nSplit-derived nonzero-offset add")
    print(f"  source texture: #{x._owner.texture} offset={x._storage_offset}")
    print(f"  left texture: #{left._owner.texture} offset={left._storage_offset}")
    print(f"  right texture: #{right._owner.texture} offset={right._storage_offset}")
    report_case("  result", split_sum, x_cpu[:, 0:16] + x_cpu[:, 16:32])

    weight = torch.randn((16, 16, 1, 1), dtype=torch.float32, generator=generator)
    conv = torch.ops.aten.convolution.default(right, weight, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
    chained = torch.add(left, conv)
    expected = x_cpu[:, 0:16] + F.conv2d(x_cpu[:, 16:32], weight)
    print("\nChained split -> Conv2D -> add")
    print(f"  split left texture: #{left._owner.texture} offset={left._storage_offset}")
    print(f"  split right texture: #{right._owner.texture} offset={right._storage_offset}")
    print(f"  conv output texture: #{conv._owner.texture} offset={conv._storage_offset}")
    report_case("  result", chained, expected)


if __name__ == "__main__":
    main()
