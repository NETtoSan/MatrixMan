#!/usr/bin/env python3
"""Step 10B dominant Conv2D baseline versus spatial-reuse diagnostic."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import os

from drivers import matrixman
from drivers.matrixman.backends.opengl import profiling


def _mismatch_report(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    difference = (actual - expected).abs()
    mask = difference > 1e-4
    indices = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    print(f"  {label}_error_count={int(indices.numel())}")
    if indices.numel():
        print(f"  {label}_first_mismatching_indices={indices[:10].tolist()}")
    lane_counts = [0, 0, 0, 0]
    interior_count = 0
    border_count = 0
    width = actual.shape[-1]
    height = actual.shape[-2]
    for flat_index in indices.tolist():
        lane_counts[flat_index % 4] += 1
        x = flat_index % width
        y = (flat_index // width) % height
        if 0 < x < width - 1 and 0 < y < height - 1:
            interior_count += 1
        else:
            border_count += 1
    coordinates = []
    for flat_index in indices[:10].tolist():
        x = flat_index % width
        y = (flat_index // width) % height
        channel = (flat_index // (height * width)) % actual.shape[-3]
        batch = flat_index // (actual.shape[-3] * height * width)
        coordinates.append((batch, channel, y, x))
    print(f"  {label}_first_mismatch_coordinates={coordinates}")
    print(f"  {label}_lane_mismatches={{0: {lane_counts[0]}, 1: {lane_counts[1]}, 2: {lane_counts[2]}, 3: {lane_counts[3]}}}")
    print(f"  {label}_interior_mismatches={interior_count} {label}_border_mismatches={border_count}")


def _run_tiny_check(gpu_input, weight) -> bool:
    expected = F.conv2d(gpu_input.cpu(), weight, padding=1)
    os.environ.pop("MATRIXMAN_CONV_SPATIAL_REUSE", None)
    baseline = F.conv2d(gpu_input, weight, padding=1).cpu()
    os.environ["MATRIXMAN_CONV_SPATIAL_REUSE"] = "1"
    actual = F.conv2d(gpu_input, weight, padding=1).cpu()
    passed = torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)
    print("  tiny deterministic Conv [1,4,4,4] -> [1,4,4,4]:")
    print(f"    baseline_flat={baseline.reshape(-1).tolist()}")
    print(f"    spatial_reuse_flat={actual.reshape(-1).tolist()}")
    print(f"    result={'PASS' if passed else 'FAIL'}")
    _mismatch_report(actual, baseline, "tiny_spatial_vs_baseline")
    return passed


def main() -> int:
    torch.manual_seed(10010)
    source = torch.randn((1, 64, 80, 80), dtype=torch.float32) * 0.05
    weight = torch.randn((64, 64, 3, 3), dtype=torch.float32) * 0.05
    expected = F.conv2d(source, weight, padding=1)

    matrixman.init()
    matrixman.profile_reset()
    try:
        gpu_input = matrixman.to_device(source)
        os.environ.pop("MATRIXMAN_CONV_SPATIAL_REUSE", None)
        matrixman.profile_reset()
        baseline = F.conv2d(gpu_input, weight, padding=1).cpu()
        baseline_gpu = profiling.gpu_timings.get("Conv2D", {}).get("total")

        os.environ["MATRIXMAN_CONV_SPATIAL_REUSE"] = "1"
        matrixman.profile_reset()
        actual = F.conv2d(gpu_input, weight, padding=1).cpu()
        fast_gpu = profiling.gpu_timings.get("Conv2D spatial reuse", {}).get("total")
        difference = (actual - expected).to(torch.float64)
        max_abs = float(difference.abs().max())
        rmse = float(torch.sqrt((difference * difference).mean()))
        passed = torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)
        print("Step 10B dominant Conv2D spatial-reuse diagnostic")
        print("  input: [1,64,80,80]")
        print("  weight: [64,64,3,3]")
        print("  output: [1,64,80,80]")
        print("  stride=1 padding=1 dilation=1 groups=1")
        print("  atlas: 320x320")
        print(f"  max_abs_error={max_abs:.6g} rmse={rmse:.6g}")
        print(f"  result: {'PASS' if passed else 'FAIL'}")
        baseline_difference = (baseline - expected).to(torch.float64)
        print(f"  baseline_max_abs_error={float(baseline_difference.abs().max()):.6g}")
        _mismatch_report(actual, baseline, "spatial_vs_baseline")
        tiny_source = torch.arange(1, 1 + 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4) / 17.0
        tiny_weight = torch.arange(1, 1 + 4 * 4 * 3 * 3, dtype=torch.float32).reshape(4, 4, 3, 3) / 31.0
        tiny_passed = _run_tiny_check(matrixman.to_device(tiny_source), tiny_weight)
        print(f"  baseline_gpu_time={'unavailable' if baseline_gpu is None else f'{baseline_gpu:.6f}s'}")
        print(f"  spatial_reuse_gpu_time={'unavailable' if fast_gpu is None else f'{fast_gpu:.6f}s'}")
        if baseline_gpu and fast_gpu:
            print(f"  measured_speedup={baseline_gpu / fast_gpu:.4f}x")
        else:
            print("  measured_speedup=unavailable")
        return 0 if passed and tiny_passed else 1
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
