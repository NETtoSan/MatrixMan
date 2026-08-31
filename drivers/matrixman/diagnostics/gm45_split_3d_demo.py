#!/usr/bin/env python3
"""Deterministic GM45 3D split test for the YOLO DFL decoder."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    cpu = torch.arange(1 * 4 * 84, dtype=torch.float32).reshape(1, 4, 84)
    x = gm45.to_device(cpu)
    left, right = torch.chunk(x, 2, dim=1)
    left_cpu = left.cpu()
    right_cpu = right.cpu()
    expected_left, expected_right = torch.chunk(cpu, 2, dim=1)

    print("YOLO DFL 3D split/chunk")
    print(f"  input: shape={list(x.shape)} strides={x._logical_strides} texture=#{x._owner.texture} offset={x._storage_offset}")
    for name, value, expected in [("left", left, expected_left), ("right", right, expected_right)]:
        value_cpu = value.cpu()
        error = (value_cpu - expected).abs().max().item()
        print(f"  {name}: shape={list(value.shape)} strides={value._logical_strides} texture=#{value._owner.texture} offset={value._storage_offset}")
        print(f"    max_abs_error: {error:.6g}")
        print(f"    allclose: {torch.allclose(value_cpu, expected)}")
    print("  texture reused:", x._owner.texture == left._owner.texture == right._owner.texture)
    print("  GPU->CPU readback: only explicit validation calls")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
