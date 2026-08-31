#!/usr/bin/env python3
"""Correctness test for MatrixMan eval/inference BatchNorm."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported


def main() -> int:
    parser = argparse.ArgumentParser(description="MatrixMan BatchNorm inference correctness test")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    set_trace_if_supported(gm45, args.trace)

    torch.manual_seed(2024)
    x_cpu = torch.randn((1, 5, 6, 7), dtype=torch.float32) * 0.5
    weight = torch.randn((5,), dtype=torch.float32) * 0.25 + 1.0
    bias = torch.randn((5,), dtype=torch.float32) * 0.1
    running_mean = torch.randn((5,), dtype=torch.float32) * 0.2
    running_var = torch.rand((5,), dtype=torch.float32) + 0.5
    eps = 1e-3

    x = gm45.to_device(x_cpu)
    y = F.batch_norm(x, running_mean, running_var, weight, bias, training=False, eps=eps)
    y_cpu = y.cpu()
    ref = F.batch_norm(x_cpu, running_mean, running_var, weight, bias, training=False, eps=eps)

    max_error = (y_cpu - ref).abs().max().item()
    print("BatchNorm eval case input=[1,5,6,7]")
    print(f"  output shape: {list(y.shape)}")
    print(f"  input storage: {describe_storage(x)}")
    print(f"  output storage: {describe_storage(y)}")
    print(f"  max_abs_error: {max_error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, ref, rtol=5e-4, atol=5e-4)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")

    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
