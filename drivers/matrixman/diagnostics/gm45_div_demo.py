#!/usr/bin/env python3
"""Deterministic GM45 scalar division test for YOLO box decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    torch.manual_seed(20260829)
    for anchors in (84, 336, 8400):
        cpu = torch.randn((1, 2, anchors), dtype=torch.float32)
        gpu = gm45.to_device(cpu)
        output = gpu / 2
        output_cpu = output.cpu()
        expected = cpu / 2
        error = (output_cpu - expected).abs().max().item()
        print(f"GM45 packed scalar division shape=[1,2,{anchors}]")
        print(f"  input:  strides={gpu._logical_strides} texture=#{gpu._owner.texture} offset={gpu._storage_offset}")
        print(f"  output: shape={list(output.shape)} strides={output._logical_strides} texture=#{output._owner.texture} offset={output._storage_offset}")
        print(f"  max_abs_error: {error:.6g}")
        print(f"  allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
        print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
