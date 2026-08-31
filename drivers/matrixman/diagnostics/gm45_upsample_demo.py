#!/usr/bin/env python3
"""Deterministic YOLO-subset nearest upsample tests for the GM45 backend."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45


def report_upsample(name: str, x, y, expected: torch.Tensor) -> None:
    y_cpu = y.cpu()
    error = (y_cpu - expected).abs().max().item()
    print(name)
    print(f"  input shape: {list(x.shape)} texture=#{x._owner.texture} offset={x._storage_offset}")
    print(f"  output shape: {list(y.shape)} texture=#{y._owner.texture} offset={y._storage_offset}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, expected, atol=1e-6, rtol=0)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print gm45 dispatch/kernel trace")
    args = parser.parse_args()

    gm45.set_trace(args.trace)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)

    known_cpu = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
    known = gm45.to_device(known_cpu)
    known_y = F.interpolate(known, size=(4, 4), mode="nearest")
    known_expected = F.interpolate(known_cpu, size=(4, 4), mode="nearest")
    report_upsample("Known-value nearest 2x upsample", known, known_y, known_expected)
    print("  known output:")
    print(known_y.cpu()[0, 0])

    x_cpu = torch.randn((1, 256, 2, 2), dtype=torch.float32, generator=generator)
    x = gm45.to_device(x_cpu)
    y = F.interpolate(x, size=(4, 4), mode="nearest")
    expected = F.interpolate(x_cpu, size=(4, 4), mode="nearest")
    report_upsample("Traced YOLO nearest upsample [1,256,2,2] -> [1,256,4,4]", x, y, expected)

    split_cpu = torch.randn((1, 512, 2, 2), dtype=torch.float32, generator=generator)
    split_source = gm45.to_device(split_cpu)
    _, right = torch.split(split_source, 256, dim=1)
    split_y = F.interpolate(right, size=(4, 4), mode="nearest")
    split_expected = F.interpolate(split_cpu[:, 256:512], size=(4, 4), mode="nearest")
    report_upsample("Split-derived nonzero-offset nearest upsample", right, split_y, split_expected)

    skip_cpu = torch.randn((1, 128, 4, 4), dtype=torch.float32, generator=generator)
    skip = gm45.to_device(skip_cpu)
    cat_y = torch.cat([split_y, skip], dim=1)
    cat_expected = torch.cat([split_expected, skip_cpu], dim=1)
    cat_cpu = cat_y.cpu()
    cat_error = (cat_cpu - cat_expected).abs().max().item()
    print("Chained upsample_nearest2d -> cat")
    print(f"  upsample output texture: #{split_y._owner.texture} offset={split_y._storage_offset}")
    print(f"  skip texture: #{skip._owner.texture} offset={skip._storage_offset}")
    print(f"  cat output shape: {list(cat_y.shape)} texture=#{cat_y._owner.texture} offset={cat_y._storage_offset}")
    print(f"  max_abs_error: {cat_error:.6g}")
    print(f"  allclose: {torch.allclose(cat_cpu, cat_expected, atol=1e-6, rtol=0)}")
    print("  GPU->CPU readback: only explicit cat.cpu() validation")


if __name__ == "__main__":
    main()
