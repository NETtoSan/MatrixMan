#!/usr/bin/env python3
"""Deterministic metadata-only split test for the experimental GM45 backend."""

from __future__ import annotations

import argparse

import torch

from drivers import matrixman as gm45


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true", help="print gm45 dispatch/kernel trace")
    args = parser.parse_args()

    gm45.set_trace(args.trace)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260829)
    x_cpu = torch.randn((1, 32, 16, 16), dtype=torch.float32, generator=generator)

    x = gm45.to_gm45(x_cpu)
    before_texture = x._owner.texture
    a, b = torch.split(x, 16, dim=1)

    print("GM45 split metadata-only correctness")
    print(f"input shape: {tuple(x.shape)} texture #{x._owner.texture} offset={x._storage_offset}")
    print(f"a shape: {tuple(a.shape)} texture #{a._owner.texture} offset={a._storage_offset}")
    print(f"b shape: {tuple(b.shape)} texture #{b._owner.texture} offset={b._storage_offset}")
    print(f"textures reused: {before_texture == a._owner.texture == b._owner.texture}")
    print(f"offsets differ: {x._storage_offset != b._storage_offset and a._storage_offset != b._storage_offset}")

    a_cpu = a.cpu()
    b_cpu = b.cpu()
    a_expected = x_cpu[:, 0:16]
    b_expected = x_cpu[:, 16:32]
    a_error = (a_cpu - a_expected).abs().max().item()
    b_error = (b_cpu - b_expected).abs().max().item()
    print(f"a max absolute error: {a_error:.6g}")
    print(f"b max absolute error: {b_error:.6g}")
    print(f"a torch.allclose: {torch.allclose(a_cpu, a_expected)}")
    print(f"b torch.allclose: {torch.allclose(b_cpu, b_expected)}")
    print("readback during split: no")
    print("readback for validation: yes")

    # This verifies a later GLSL kernel consumes the split view's nonzero
    # storage offset instead of reading from the start of the original texture.
    b_clone = gm45.Gm45Tensor._from_owner(b._owner, tuple(b.shape), b._storage_offset)
    torch.ops.aten.silu_.default(b_clone)
    silu_cpu = b_clone.cpu()
    silu_expected = torch.nn.functional.silu(b_expected)
    silu_error = (silu_cpu - silu_expected).abs().max().item()
    print(f"nonzero-offset SiLU max absolute error: {silu_error:.6g}")
    print(f"nonzero-offset SiLU torch.allclose: {torch.allclose(silu_cpu, silu_expected, atol=1e-5, rtol=1e-5)}")


if __name__ == "__main__":
    main()
