"""Run a real MatrixMan/OpenGL Conv2D dispatch and show its physical tiles."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from drivers import matrixman
from drivers.matrixman.backend import get_backend


RTOL = 5e-4
ATOL = 5e-4


def _numpy_conv2d(x, weight, bias, stride: int, padding: int) -> np.ndarray:
    """Independent NCHW float32 reference using padded spatial windows."""
    batch, channels, height, width = x.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    if batch != 1 or weight_channels != channels:
        raise ValueError("reference requires batch=1 and matching channels")
    out_h = (height + 2 * padding - kernel_h) // stride + 1
    out_w = (width + 2 * padding - kernel_w) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv parameters produce an empty output")
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    result = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)
    for ky in range(kernel_h):
        for kx in range(kernel_w):
            window = padded[:, :, ky:ky + out_h * stride:stride, kx:kx + out_w * stride:stride]
            result += np.einsum("nchw,oc->nohw", window, weight[:, :, ky, kx], optimize=True).astype(np.float32)
    if bias is not None:
        result += bias.reshape(1, out_channels, 1, 1)
    return result


def _metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "allclose": bool(np.allclose(actual, reference, rtol=RTOL, atol=ATOL)),
    }


def _print_tiles(telemetry: dict) -> None:
    atlas_w, atlas_h = (int(value) for value in telemetry.get("atlas", (0, 0)))
    tiles = telemetry.get("tiles", [])
    print(f"logical output atlas: {atlas_w}x{atlas_h}")
    print(f"execution mode: {'tiled' if telemetry.get('tiled') else 'direct'}")
    print(f"physical tiles: {len(tiles)}")
    largest = max(((int(t["width"]), int(t["height"])) for t in tiles), default=(0, 0))
    print(f"largest physical tile: {largest[0]}x{largest[1]}")
    print(f"consolidation: {'yes' if telemetry.get('tiled') else 'no'}")
    if not tiles:
        return
    by_grid = {tuple(t.get("grid", (i, 0))): t for i, t in enumerate(tiles)}
    max_x = max(grid[0] for grid in by_grid)
    max_y = max(grid[1] for grid in by_grid)
    print("\nTile layout (actual physical geometry):")
    for grid_y in range(max_y + 1):
        print(" | ".join(
            "--" if (grid_x, grid_y) not in by_grid else
            f"{int(by_grid[(grid_x, grid_y)]['width'])}x{int(by_grid[(grid_x, grid_y)]['height'])}"
            for grid_x in range(max_x + 1)
        ))
    for index, tile in enumerate(tiles):
        origin_x, origin_y = (int(value) for value in tile["origin"])
        print(f"tile {index}: x={origin_x:<4} y={origin_y:<4} w={int(tile['width']):<4} h={int(tile['height'])}")


def _show_values(name: str, values: np.ndarray, size: int) -> None:
    print(f"\n{name} (channel 0, top-left {size}x{size}):")
    print(values[0, 0, :size, :size])


def _run_matrixman(x, weight, bias, stride: int, padding: int, tile_limit: int):
    matrixman.config.tileLimit = tile_limit
    started = time.perf_counter()
    with torch.no_grad():
        result = F.conv2d(matrixman.to_device(x), weight, bias, stride=stride, padding=padding)
    actual = result.cpu().numpy()
    return actual, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="MatrixMan OpenGL Conv2D physical tiling demo")
    parser.add_argument("--tile-size", type=int, default=16, help="temporary OpenGL physical tile limit")
    parser.add_argument("--input-size", type=int, default=32, help="square input height and width")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--preset", choices=("default", "yolo80"), default="default")
    parser.add_argument("--tile-sync", choices=("per_tile", "end"), default=None)
    parser.add_argument("--spatial-reuse", action="store_true")
    parser.add_argument("--show-values", action="store_true")
    parser.add_argument("--show-size", type=int, default=8)
    parser.add_argument("--compare-direct", action="store_true")
    args = parser.parse_args()

    if args.preset == "yolo80":
        args.input_size, args.channels, args.out_channels = 80, 64, 64
        args.kernel_size, args.stride, args.padding = 3, 1, 1
    if min(args.tile_size, args.input_size, args.channels, args.out_channels, args.kernel_size, args.stride) <= 0:
        parser.error("sizes, channels, kernel-size, and stride must be positive")
    if args.padding < 0 or args.show_size <= 0:
        parser.error("padding must be non-negative and show-size must be positive")
    if args.kernel_size not in (1, 3) or args.stride not in (1, 2) or args.padding not in (0, 1):
        parser.error("MatrixMan Conv supports kernel-size 1/3, stride 1/2, and padding 0/1")

    rng = np.random.default_rng(0)
    x_np = (rng.standard_normal((1, args.channels, args.input_size, args.input_size)) * 0.05).astype(np.float32)
    weight_np = (rng.standard_normal((args.out_channels, args.channels, args.kernel_size, args.kernel_size)) * 0.05).astype(np.float32)
    bias_np = (rng.standard_normal(args.out_channels) * 0.02).astype(np.float32)
    x, weight, bias = map(torch.from_numpy, (x_np, weight_np, bias_np))
    reference = _numpy_conv2d(x_np, weight_np, bias_np, args.stride, args.padding)

    old_values = {"tileLimit": matrixman.config.tileLimit, "tileSync": matrixman.config.tileSync, "convSpatialReuse": matrixman.config.convSpatialReuse}
    try:
        matrixman.prefer("opengl")
        matrixman.config.tileLimit = args.tile_size
        if args.tile_sync is not None:
            matrixman.config.tileSync = args.tile_sync
        matrixman.config.convSpatialReuse = args.spatial_reuse
        matrixman.init()
        backend = get_backend()
        if backend.name != "opengl":
            raise RuntimeError(f"OpenGL diagnostic requires OpenGL; selected {backend.name}")
        info = backend.device_info()
        from drivers.matrixman.backends.opengl import convolution

        print("MatrixMan OpenGL Conv tiling demo\n")
        for label, key in (("vendor", "vendor"), ("renderer", "renderer"), ("OpenGL", "opengl"), ("GLSL", "glsl")):
            print(f"{label}: {info.get(key, 'unavailable')}")
        print(f"requested GPU preference: {info.get('gpu_preference', 'unavailable')}")
        print(f"actual GPU classification: {info.get('device_policy', 'unavailable')}")
        print(f"\ninput:        {list(x.shape)}\nweights:      {list(weight.shape)}\noutput:       {list(reference.shape)}")
        print(f"\nrequested tile limit: {args.tile_size}\neffective tile limit:  {matrixman.config.tileLimit}")

        tiled, tiled_seconds = _run_matrixman(x, weight, bias, args.stride, args.padding, args.tile_size)
        telemetry = convolution.diagnostic_tile_geometry()
        print(f"\nGPU runtime: {tiled_seconds:.4f}s")
        _print_tiles(telemetry)
        tiled_metrics = _metrics(reference, tiled)
        print("\ncomparison:")
        print(f"  max_abs:  {tiled_metrics['max_abs']:.6g}\n  mean_abs: {tiled_metrics['mean_abs']:.6g}\n  rmse:     {tiled_metrics['rmse']:.6g}")
        print(f"  allclose: {'PASS' if tiled_metrics['allclose'] else 'FAIL'}")
        if args.show_values:
            size = min(args.show_size, tiled.shape[2], tiled.shape[3])
            _show_values("NumPy reference", reference, size)
            _show_values("MatrixMan result", tiled, size)
            _show_values("absolute difference", np.abs(tiled - reference), size)

        direct_metrics = None
        if args.compare_direct:
            direct, direct_seconds = _run_matrixman(x, weight, bias, args.stride, args.padding, max(args.tile_size, 4096))
            direct_metrics = _metrics(reference, direct)
            cross_metrics = _metrics(tiled, direct)
            print("\ncompare-direct:")
            print(f"  NumPy vs tiled:  {tiled_metrics['max_abs']:.6g} {'PASS' if tiled_metrics['allclose'] else 'FAIL'}")
            print(f"  NumPy vs direct: {direct_metrics['max_abs']:.6g} {'PASS' if direct_metrics['allclose'] else 'FAIL'}")
            print(f"  tiled vs direct: {cross_metrics['max_abs']:.6g} {'PASS' if cross_metrics['allclose'] else 'FAIL'}")
            print(f"  direct runtime:  {direct_seconds:.4f}s")
        return 0 if tiled_metrics["allclose"] and (direct_metrics is None or direct_metrics["allclose"]) else 2
    finally:
        matrixman.shutdown()
        matrixman.config.tileLimit = old_values["tileLimit"]
        matrixman.config.tileSync = old_values["tileSync"]
        matrixman.config.convSpatialReuse = old_values["convSpatialReuse"]


if __name__ == "__main__":
    raise SystemExit(main())
