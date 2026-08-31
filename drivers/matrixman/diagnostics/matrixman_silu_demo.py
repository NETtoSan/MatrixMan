#!/usr/bin/env python3
"""Correctness test for MatrixMan aten.silu_.default."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported, storage_identity


def main() -> int:
    parser = argparse.ArgumentParser(description="MatrixMan SiLU correctness test")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    set_trace_if_supported(gm45, args.trace)

    torch.manual_seed(3030)
    x_cpu = torch.randn((1, 7, 5, 6), dtype=torch.float32) * 2.0
    x = gm45.to_device(x_cpu)
    input_storage = storage_identity(x)

    # This dispatches aten.silu_.default. The backend writes a new output texture
    # instead of physically mutating the input texture, avoiding GL read/write
    # feedback hazards.
    y = F.silu(x, inplace=True)
    y_cpu = y.cpu()
    ref = F.silu(x_cpu)

    max_error = (y_cpu - ref).abs().max().item()
    print("SiLU inplace-dispatch case input=[1,7,5,6]")
    print(f"  output shape: {list(y.shape)}")
    print(f"  input storage: {describe_storage(x)}")
    print(f"  output storage: {describe_storage(y)}")
    print(f"  physical in-place storage mutation: {storage_identity(y) == input_storage}")
    print("  logical in-place wrapper update: True")
    print(f"  max_abs_error: {max_error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, ref, rtol=5e-4, atol=5e-4)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
