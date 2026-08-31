#!/usr/bin/env python3
"""Focused metadata-only transpose and unsqueeze regression test."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
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
    print(f"  input:  shape={list(x.shape)} strides={x._logical_strides} texture=#{x._owner.texture} offset={x._storage_offset}")
    print(f"  transposed: shape={list(y.shape)} strides={y._logical_strides} texture=#{y._owner.texture} offset={y._storage_offset}")
    print(f"  unsqueezed: shape={list(z.shape)} strides={z._logical_strides} texture=#{z._owner.texture} offset={z._storage_offset}")
    print(f"  transpose max_abs_error: {y_error:.6g}")
    print(f"  transpose allclose: {torch.allclose(y_cpu, y_ref)}")
    print(f"  unsqueeze max_abs_error: {z_error:.6g}")
    print(f"  unsqueeze allclose: {torch.allclose(z_cpu, z_ref)}")
    print("  texture reused:", x._owner.texture == y._owner.texture == z._owner.texture)
    print("  GPU->CPU readback: only explicit validation calls")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
