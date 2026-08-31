#!/usr/bin/env python3
"""
OpenGL texture storage and metadata-view smoke tests.

No convolution is implemented here. This verifies that MatrixMan can hold
YOLO-shaped tensors in GPU-resident OpenGL texture storage and reconstruct the
original CPU shape only when .cpu() is explicitly requested.
"""

from __future__ import annotations

import argparse

import torch

from drivers import matrixman as gm45
from drivers.matrixman.backend import get_backend


def check_roundtrip(shape: tuple[int, ...]) -> None:
    x_cpu = torch.randn(shape, dtype=torch.float32)
    x = gm45.to_device(x_cpu)
    x_back = x.cpu()
    print(f"roundtrip {list(shape)}:")
    print(f"  MatrixMan tensor: {x}")
    print(f"  shape ok: {tuple(x.shape) == shape}")
    print(f"  device: {x.device}")
    print(f"  allclose: {torch.allclose(x_cpu, x_back)}")
    print(f"  max_abs_error: {(x_cpu - x_back).abs().max().item():.6g}")


def main() -> int:
    selected = get_backend()
    if selected.name != "opengl":
        print("opengl_storage_demo requires the OpenGL backend (it tests texture storage).")
        print(f"Selected backend: {selected.name.upper()}")
        return 0
    parser = argparse.ArgumentParser(description="OpenGL NCHW texture-storage smoke test")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    gm45.set_trace(args.trace)

    torch.manual_seed(456)
    check_roundtrip((1, 3, 64, 64))
    check_roundtrip((1, 16, 32, 32))
    check_roundtrip((1, 32, 16, 16))
    check_roundtrip((1, 64, 8, 8))

    x_cpu = torch.randn((1, 16, 32, 32), dtype=torch.float32)
    x = gm45.to_device(x_cpu)
    y = x.view(1, 16, 1024)
    z = y.flatten(1)
    s = z.unsqueeze(0).squeeze(0)
    print("\nmetadata-only operations:")
    print("  x:", x)
    print("  view ->", y)
    print("  flatten ->", z)
    print("  unsqueeze+squeeze ->", s)
    print("  final readback allclose:", torch.allclose(s.cpu(), x_cpu.reshape(1, 16384)))

    for label, op in [
        ("permute", lambda: x.permute(0, 2, 3, 1)),
        ("transpose", lambda: x.transpose(1, 2)),
    ]:
        try:
            op()
        except RuntimeError as exc:
            print(f"  {label} unsupported as expected: {exc}")

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
