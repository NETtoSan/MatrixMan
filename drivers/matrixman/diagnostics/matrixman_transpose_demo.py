#!/usr/bin/env python3
"""Focused metadata-only transpose and unsqueeze regression test."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported, storage_identity


def main() -> int:
    set_trace_if_supported(gm45, True)
    cpu = torch.arange(168, dtype=torch.float32).reshape(84, 2)
    x = gm45.to_device(cpu)
    y = x.transpose(0, 1)
    z = y.unsqueeze(0)

    y_cpu = y.cpu()
    z_cpu = z.cpu()
    y_ref = cpu.transpose(0, 1)
    z_ref = y_ref.unsqueeze(0)
    y_error = (y_cpu - y_ref).abs().max().item()
    z_error = (z_cpu - z_ref).abs().max().item()

    print("transpose(0, 1) -> unsqueeze(0)")
    print(f"  input:  shape={list(x.shape)} {describe_storage(x)}")
    print(f"  transposed: shape={list(y.shape)} {describe_storage(y)}")
    print(f"  unsqueezed: shape={list(z.shape)} {describe_storage(z)}")
    print(f"  transpose max_abs_error: {y_error:.6g}")
    print(f"  transpose allclose: {torch.allclose(y_cpu, y_ref)}")
    print(f"  unsqueeze max_abs_error: {z_error:.6g}")
    print(f"  unsqueeze allclose: {torch.allclose(z_cpu, z_ref)}")
    print("  storage reused:", storage_identity(x) == storage_identity(y) == storage_identity(z))
    print("  GPU->CPU readback: only explicit validation calls")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
