#!/usr/bin/env python3
"""Deterministic 3D final-dimension cat tests for the MatrixMan detection-head path."""

from __future__ import annotations

import argparse

import torch

from drivers import matrixman as gm45
from drivers.matrixman.diagnostics.backend_helpers import describe_storage, set_trace_if_supported


def report_cat(name: str, inputs, y, expected: torch.Tensor) -> torch.Tensor:
    y_cpu = y.cpu()
    error = (y_cpu - expected).abs().max().item()
    print(name)
    for index, tensor in enumerate(inputs):
        print(f"  input {index}: shape={list(tensor.shape)} {describe_storage(tensor)}")
    print(f"  output shape: {list(y.shape)} {describe_storage(y)}")
    print(f"  max_abs_error: {error:.6g}")
    print(f"  allclose: {torch.allclose(y_cpu, expected, atol=1e-6, rtol=0)}")
    print("  GPU->CPU readback: only explicit y.cpu() validation")
    return y_cpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print MatrixMan dispatch/kernel trace")
    args = parser.parse_args()

    set_trace_if_supported(gm45, args.trace)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)

    a_cpu = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]], dtype=torch.float32)
    b_cpu = torch.tensor([[[4.0, 5.0], [40.0, 50.0]]], dtype=torch.float32)
    a = gm45.to_device(a_cpu)
    b = gm45.to_device(b_cpu)
    tiny = torch.cat([a, b], dim=-1)
    tiny_expected = torch.cat([a_cpu, b_cpu], dim=-1)
    tiny_cpu = report_cat("Tiny row-wise 3D final-dim cat", [a, b], tiny, tiny_expected)
    print("  tiny output:")
    print(tiny_cpu)

    x0_cpu = torch.randn((1, 64, 64), dtype=torch.float32, generator=generator)
    x1_cpu = torch.randn((1, 64, 16), dtype=torch.float32, generator=generator)
    x2_cpu = torch.randn((1, 64, 4), dtype=torch.float32, generator=generator)
    x0 = gm45.to_device(x0_cpu)
    x1 = gm45.to_device(x1_cpu)
    x2 = gm45.to_device(x2_cpu)
    y = torch.cat([x0, x1, x2], dim=-1)
    expected = torch.cat([x0_cpu, x1_cpu, x2_cpu], dim=-1)
    report_cat("Traced YOLO detection-head 3D final-dim cat", [x0, x1, x2], y, expected)

    split_cpu = torch.randn((1, 128, 8), dtype=torch.float32, generator=generator)
    split_source = gm45.to_device(split_cpu)
    right = gm45.MatrixManTensor._from_owner(split_source._owner, (1, 64, 8), 64 * 8)
    tail_cpu = torch.randn((1, 64, 4), dtype=torch.float32, generator=generator)
    tail = gm45.to_device(tail_cpu)
    split_y = torch.cat([right, tail], dim=-1)
    split_expected = torch.cat([split_cpu[:, 64:128, :], tail_cpu], dim=-1)
    report_cat("Contiguous nonzero-offset 3D final-dim cat", [right, tail], split_y, split_expected)


if __name__ == "__main__":
    main()
