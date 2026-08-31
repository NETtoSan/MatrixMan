#!/usr/bin/env python3
"""Deterministic GM45 arange checks."""

from __future__ import annotations

import torch

from drivers import matrixman as gm45


def check(label: str, result, expected) -> None:
    cpu = result.cpu()
    max_abs = (cpu - expected).abs().max().item() if expected.numel() else 0.0
    print(label)
    print(f"  gm45: {cpu.tolist()}")
    print(f"  cpu:  {expected.tolist()}")
    print(
        f"  shape: {tuple(result.shape)} texture #{result._owner.texture} "
        f"offset={result._storage_offset} strides={result._logical_strides}"
    )
    print(f"  max_abs_error: {max_abs:.8g}")
    print(f"  allclose: {torch.allclose(cpu, expected)}")


def check_from_callable(label: str, factory, expected) -> None:
    result = factory()
    check(label, result, expected)


def check_inplace_fill(label: str, tensor, value: float, expected) -> None:
    before_id = id(tensor)
    before_texture = tensor._owner.texture
    result = tensor.fill_(value)
    same_object = id(result) == before_id
    cpu = result.cpu()
    max_abs = (cpu - expected).abs().max().item()
    print(label)
    print(f"  before texture: #{before_texture}")
    print(f"  after texture:  #{result._owner.texture}")
    print(f"  same Python object: {same_object}")
    print(f"  values: {cpu.tolist()}")
    print(f"  max_abs_error: {max_abs:.8g}")
    print(f"  allclose: {torch.allclose(cpu, expected)}")


def main() -> None:
    gm45.set_trace(True)
    seed_tensor = gm45.to_device(torch.zeros((1,), dtype=torch.float32))
    device = seed_tensor.device

    check(
        "torch.arange(4, device=gm45, dtype=float32)",
        torch.arange(4, device=device, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
    )
    check(
        "torch.arange(8, device=gm45, dtype=float32)  # YOLO 64x64 decode grid",
        torch.arange(8, device=device, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
    )
    check(
        "torch.arange(1, 7, 2, device=gm45, dtype=float32)",
        torch.arange(1, 7, 2, device=device, dtype=torch.float32),
        torch.arange(1, 7, 2, dtype=torch.float32),
    )
    gm45_arange = torch.arange(8, device=device, dtype=torch.float32)
    check_from_callable(
        "torch.arange(8, device=gm45, dtype=float32) + 0.5",
        lambda: gm45_arange + 0.5,
        torch.arange(8, dtype=torch.float32) + 0.5,
    )
    check_from_callable(
        "0.5 + torch.arange(8, device=gm45, dtype=float32)",
        lambda: 0.5 + gm45_arange,
        0.5 + torch.arange(8, dtype=torch.float32),
    )
    nchw_cpu = torch.arange(1 * 3 * 2 * 4, dtype=torch.float32).reshape(1, 3, 2, 4)
    nchw = gm45.to_device(nchw_cpu)
    check_from_callable(
        "packed NCHW tensor + scalar",
        lambda: nchw + 1.25,
        nchw_cpu + 1.25,
    )
    shifted = torch.arange(8, device=device, dtype=torch.float32) + 0.5
    col = shifted.view(8, 1)
    row = shifted.view(1, 8)
    check_from_callable(
        "YOLO meshgrid-style expand [8,1] -> [8,8]",
        lambda: col.expand(8, 8),
        (torch.arange(8, dtype=torch.float32) + 0.5).view(8, 1).expand(8, 8),
    )
    check_from_callable(
        "YOLO meshgrid-style expand [1,8] -> [8,8]",
        lambda: row.expand(8, 8),
        (torch.arange(8, dtype=torch.float32) + 0.5).view(1, 8).expand(8, 8),
    )
    sx = row.expand(8, 8)
    sy = col.expand(8, 8)
    check_from_callable(
        "YOLO meshgrid stack((sx, sy), -1)",
        lambda: torch.stack((sx, sy), -1),
        torch.stack(
            (
                (torch.arange(8, dtype=torch.float32) + 0.5).view(1, 8).expand(8, 8),
                (torch.arange(8, dtype=torch.float32) + 0.5).view(8, 1).expand(8, 8),
            ),
            -1,
        ),
    )
    check(
        "torch.full((64, 1), 8.0, device=gm45, dtype=float32)",
        torch.full((64, 1), 8.0, device=device, dtype=torch.float32),
        torch.full((64, 1), 8.0, dtype=torch.float32),
    )
    check_inplace_fill(
        "torch.empty((2, 3), device=gm45, dtype=float32).fill_(2.5)",
        torch.empty((2, 3), device=device, dtype=torch.float32),
        2.5,
        torch.full((2, 3), 2.5, dtype=torch.float32),
    )


if __name__ == "__main__":
    main()
