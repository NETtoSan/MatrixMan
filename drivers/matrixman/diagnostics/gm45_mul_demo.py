#!/usr/bin/env python3
"""Deterministic GM45 broadcast multiplication test for YOLO decode."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def main() -> int:
    gm45.set_trace(True)
    torch.manual_seed(20260829)
    print("GM45 YOLO broadcast multiplication")
    for anchors in (84, 336, 8400):
        boxes_cpu = torch.randn((1, 4, anchors), dtype=torch.float32)
        strides_cpu = torch.randn((1, anchors), dtype=torch.float32)
        boxes = gm45.to_device(boxes_cpu)
        strides = gm45.to_device(strides_cpu)
        output = boxes * strides
        output_cpu = output.cpu()
        expected = boxes_cpu * strides_cpu
        error = (output_cpu - expected).abs().max().item()
        print(f"  A={anchors}")
        print(f"    left:   shape={list(boxes.shape)} strides={boxes._logical_strides} texture=#{boxes._owner.texture} offset={boxes._storage_offset}")
        print(f"    right:  shape={list(strides.shape)} strides={strides._logical_strides} texture=#{strides._owner.texture} offset={strides._storage_offset}")
        print(f"    output: shape={list(output.shape)} strides={output._logical_strides} texture=#{output._owner.texture} offset={output._storage_offset}")
        print(f"    max_abs_error: {error:.6g}")
        print(f"    allclose: {torch.allclose(output_cpu, expected, atol=1e-5, rtol=1e-5)}")
    print("  GPU->CPU readback: only explicit output validation")
    gm45.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
