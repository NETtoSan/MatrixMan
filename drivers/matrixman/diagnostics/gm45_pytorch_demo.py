#!/usr/bin/env python3
"""
Minimal Phase 1 PyTorch-facing GM45 backend demo.

This demonstrates:
  - PrivateUse1 renamed to matrixman
  - CPU -> gm45 upload
  - torch.add on gm45 tensors
  - torch.matmul on gm45 tensors
  - chained gm45 operation without intermediate CPU readback
  - gm45 -> CPU readback
  - clear error for unsupported operations
"""

from __future__ import annotations

import argparse

import torch

from drivers import matrixman as gm45


def check(label: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    max_error = (actual - expected).abs().max().item()
    print(f"{label}: max_abs_error={max_error:.6g}, matches={torch.allclose(actual, expected, rtol=5e-4, atol=5e-4)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal GM45 PyTorch wrapper demo")
    parser.add_argument("--trace", action="store_true", help="print torch_dispatch/OpenGL/GLSL execution trace")
    args = parser.parse_args()
    gm45.set_trace(args.trace)

    print("PrivateUse1 backend name:", torch._C._get_privateuse1_backend_name())
    print("torch.device('matrixman:0') displays as:", torch.device("matrixman:0"))

    torch.manual_seed(123)
    a_cpu = torch.randn((16, 16), dtype=torch.float32)
    b_cpu = torch.randn((16, 16), dtype=torch.float32)

    a = gm45.to_device(a_cpu)
    b = gm45.to_device(b_cpu)
    print("A:", a)
    print("B:", b)

    c = torch.add(a, b)
    d = torch.matmul(a, b)
    e = torch.add(d, a)
    print("C = A + B:", c)
    print("D = A @ B:", d)
    print("E = D + A:", e)

    check("add", c.cpu(), a_cpu + b_cpu)
    check("matmul", d.cpu(), a_cpu @ b_cpu)
    check("chain matmul->add", e.cpu(), (a_cpu @ b_cpu) + a_cpu)

    try:
        torch.sin(a)
    except RuntimeError as exc:
        print("unsupported op error:", exc)

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
