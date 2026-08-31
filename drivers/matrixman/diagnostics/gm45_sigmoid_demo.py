#!/usr/bin/env python3
"""Deterministic GM45 sigmoid test for YOLO detection scores."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    torch.manual_seed(20260829)
    print("GM45 YOLO sigmoid")
    for channels, anchors in ((80, 84), (80, 8400), (10, 33600)):
        cpu = torch.randn((1, channels, anchors), dtype=torch.float32)
        gpu = gm45.to_device(cpu)
        output = torch.sigmoid(gpu)
        output_cpu = output.cpu()
        expected = torch.sigmoid(cpu)
        error = (output_cpu - expected).abs().max().item()
        print(f"  shape=[1,{channels},{anchors}]")
        print(f"    input:  shape={list(gpu.shape)} strides={gpu._logical_strides} texture=#{gpu._owner.texture} offset={gpu._storage_offset}")
        print(f"    output: shape={list(output.shape)} strides={output._logical_strides} texture=#{output._owner.texture} offset={output._storage_offset}")
        print(f"    max_abs_error: {error:.6g}")
        print(f"    allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
        print(f"    new output texture: {output._owner.texture != gpu._owner.texture}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
