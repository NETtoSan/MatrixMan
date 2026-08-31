#!/usr/bin/env python3
"""Deterministic selected-backend MatrixMan scalar-division diagnostic."""

from __future__ import annotations

import torch

from drivers import matrixman
from drivers.matrixman.diagnostics.backend_helpers import (
    describe_storage,
    readback_tensor,
    set_trace_if_supported,
)


def main() -> int:
    set_trace_if_supported(matrixman, True)
    from drivers.matrixman.backend import get_backend

    backend = get_backend()
    print("selected backend:", backend.name)
    print("aten.div.Tensor schema:", torch.ops.aten.div.Tensor._schema)
    print("aten.div.Scalar schema:", torch.ops.aten.div.Scalar._schema)
    print("tensor / 2 RHS representation: Python int")
    info = backend.device_info()
    if info.get("device"):
        print("device:", info["device"])
    if info.get("compute_capability"):
        print("compute capability:", info["compute_capability"])

    def check(label, actual, expected, source):
        if not isinstance(actual, matrixman.MatrixManTensor):
            raise AssertionError(f"{label}: result is not a MatrixMan tensor")
        value = readback_tensor(actual)
        error = float((value - expected).abs().max()) if value.numel() else 0.0
        close = torch.allclose(value, expected, rtol=1e-5, atol=1e-6)
        print(
            f"{label}: input_shape={list(source.shape)} input_strides={list(source._logical_strides)} "
            f"output_shape={list(actual.shape)} output_strides={list(actual._logical_strides)} "
            f"max_abs_diff={error:.6g} allclose={close} storage={describe_storage(actual)}"
        )
        if not close:
            raise AssertionError(f"{label}: result mismatch")

    def run(label, operation, expected, source):
        try:
            check(label, operation(), expected, source)
        except (NotImplementedError, RuntimeError) as exc:
            print(f"{label}: unsupported on selected backend ({exc})")

    base_cpu = torch.arange(48, dtype=torch.float32).reshape(1, 2, 24).contiguous()
    base = matrixman.to_device(base_cpu)
    for label, divisor in (("scalar 2", 2.0), ("scalar -2", -2.0), ("scalar 0.5", 0.5)):
        run(label, lambda divisor=divisor: base / divisor, base_cpu / divisor, base)

    transposed_cpu = base_cpu.transpose(1, 2)
    transposed = base.transpose(1, 2)
    run("transposed scalar 2", lambda: transposed / 2.0, transposed_cpu / 2.0, transposed)

    expanded_base_cpu = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24).contiguous()
    expanded_base = matrixman.to_device(expanded_base_cpu)
    expanded_cpu = expanded_base_cpu.expand(1, 2, 24)
    expanded = expanded_base.expand(1, 2, 24)
    run("zero-stride scalar 2", lambda: expanded / 2.0, expanded_cpu / 2.0, expanded)

    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
