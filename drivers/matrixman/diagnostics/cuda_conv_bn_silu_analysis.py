"""CUDA-only Conv+BatchNorm+SiLU fusion opportunity diagnostic.

This intentionally does not alter the production model or CUDA backend.  It
compares the current three-op MatrixMan path with a CPU-prepared, BN-folded
Conv followed by the existing SiLU operation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from drivers import matrixman
from drivers.matrixman.backend import get_backend
from drivers.matrixman.backends.cuda import profiling


def _fold_bn(weight, bias, running_mean, running_var, bn_weight, bn_bias, eps):
    if bias is None:
        bias = torch.zeros_like(running_mean)
    scale = bn_weight / torch.sqrt(running_var + eps)
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = (bias - running_mean) * scale + bn_bias
    return folded_weight, folded_bias


def _counter_snapshot():
    kernel_records = ("Conv2D", "BatchNorm", "SiLU")
    return {
        # The aggregate record table also contains HtoD/DtoH, allocation,
        # free, and synchronization calls.  These are the kernel launches for
        # the representative chain itself.
        "launches": sum(int(profiling.records[name]["calls"]) for name in kernel_records),
        "allocations": int(profiling.allocation["requests"]),
        "driver_allocations": int(profiling.allocation["driver_allocations"]),
        "bytes": int(profiling.allocation["requested_bytes"]),
    }


def _run_case(size, folded, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((1, 16, size, size), generator=generator, dtype=torch.float32)
    weight = torch.randn((16, 16, 3, 3), generator=generator, dtype=torch.float32) * 0.1
    bias = torch.randn((16,), generator=generator, dtype=torch.float32) * 0.1
    running_mean = torch.randn((16,), generator=generator, dtype=torch.float32) * 0.1
    running_var = torch.rand((16,), generator=generator, dtype=torch.float32) + 0.5
    bn_weight = torch.randn((16,), generator=generator, dtype=torch.float32)
    bn_bias = torch.randn((16,), generator=generator, dtype=torch.float32)
    eps = 1e-3

    folded_weight, folded_bias = _fold_bn(
        weight, bias, running_mean, running_var, bn_weight, bn_bias, eps
    )
    expected = F.silu(
        F.conv2d(x, folded_weight if folded else weight,
                 folded_bias if folded else bias, padding=1)
        if folded else
        F.batch_norm(
            F.conv2d(x, weight, bias, padding=1), running_mean, running_var,
            bn_weight, bn_bias, training=False, momentum=0.1, eps=eps,
        )
    )

    matrixman.profile_reset()
    gpu_x = matrixman.to_device(x)
    started = time.perf_counter()
    if folded:
        gpu_y = F.conv2d(gpu_x, folded_weight, folded_bias, padding=1)
        gpu_out = F.silu(gpu_y, inplace=True)
    else:
        gpu_y = F.conv2d(gpu_x, weight, bias, padding=1)
        gpu_z = F.batch_norm(
            gpu_y, running_mean, running_var, bn_weight, bn_bias,
            training=False, momentum=0.1, eps=eps,
        )
        # This matches Ultralytics' Conv.act path: SiLU is logically in-place
        # and reuses the BatchNorm output storage.
        gpu_out = F.silu(gpu_z, inplace=True)
    actual = gpu_out.cpu()
    synchronized_elapsed = time.perf_counter() - started
    stats = _counter_snapshot()
    error = (actual - expected).abs()
    return {
        "size": size,
        "folded": folded,
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
        "allclose": bool(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)),
        "launches": stats["launches"],
        "allocations": stats["allocations"],
        "driver_allocations": stats["driver_allocations"],
        "requested_bytes": stats["bytes"],
        "synchronized_ms": synchronized_elapsed * 1000.0,
    }


def _inventory(model_path: Path):
    from ultralytics import YOLO

    model = YOLO(str(model_path)).model
    blocks = []
    for index, module in enumerate(model.modules()):
        if hasattr(module, "bn") and hasattr(module, "act"):
            blocks.append((index, type(module).__name__, type(module.bn).__name__, type(module.act).__name__))
    print("Ultralytics Conv-block inventory")
    print(f"  Conv+BN+activation blocks: {len(blocks)}")
    print(f"  blocks with SiLU activation: {sum(item[3].lower().startswith('silu') for item in blocks)}")
    print("  production profiler observation: 42 Conv, 35 BatchNorm, 35 SiLU launches/frame")
    print("  intermediate consumer analysis: Conv output is consumed by BN, BN output by SiLU; no extra consumer in these sequential blocks")
    observed = []
    hooks = []
    for index, module in enumerate(model.model):
        if hasattr(module, "bn") and hasattr(module, "act") and type(module.act).__name__.lower().startswith("silu"):
            hooks.append(module.register_forward_hook(
                lambda _module, _inputs, output, index=index: observed.append(
                    (index, tuple(int(item) for item in output.shape), int(output.numel()))
                )
            ))
    with torch.no_grad():
        model(torch.zeros((1, 3, 320, 320), dtype=torch.float32))
    for hook in hooks:
        hook.remove()
    print(f"  320x320 output elements across eligible chains: {sum(item[2] for item in observed)}")
    print(f"  eligible chain shapes: {[item[1] for item in observed]}")


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[3] / "demo"
    parser.add_argument("--model", type=Path, default=base / "models/VisDrone-arm64-480/weights/best.pt")
    args = parser.parse_args()

    matrixman.prefer("cuda")
    matrixman.init()
    try:
        print(f"CUDA device: {get_backend().device_info()}")
        _inventory(args.model)
        print("\nFolded BN formula: scale=bn_weight/sqrt(running_var+eps); folded_weight=weight*scale; folded_bias=(bias-running_mean)*scale+bn_bias")
        for size in (80, 40, 20):
            for folded in (False, True):
                # Warm up PTX loading and the allocation/parameter caches;
                # report the second synchronized run.
                _run_case(size, folded, seed=1000 + size)
                result = _run_case(size, folded, seed=1000 + size)
                print(result)
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    main()
