#!/usr/bin/env python3
"""Deterministic selected-backend MatrixMan elementwise multiplication diagnostic."""

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
    print("aten.mul.Tensor schema:", torch.ops.aten.mul.Tensor._schema)
    print("YOLO broadcast case: dbox=[1,4,A], strides=[1,A]")
    info = backend.device_info()
    if info.get("device"):
        print("device:", info["device"])
    if info.get("compute_capability"):
        print("compute capability:", info["compute_capability"])

    def check(label, actual, expected, inputs):
        if not isinstance(actual, matrixman.MatrixManTensor):
            raise AssertionError(f"{label}: result is not a MatrixMan tensor")
        value = readback_tensor(actual)
        error = float((value - expected).abs().max()) if value.numel() else 0.0
        close = torch.allclose(value, expected, rtol=1e-5, atol=1e-6)
        print(
            f"{label}: input_shapes={[list(item.shape) for item in inputs]} "
            f"input_strides={[list(item._logical_strides) for item in inputs]} "
            f"output_shape={list(actual.shape)} output_strides={list(actual._logical_strides)} "
            f"max_abs_diff={error:.6g} allclose={close} "
            f"new_output={all(actual._owner is not item._owner for item in inputs)} "
            f"storage={describe_storage(actual)}"
        )
        if not close:
            raise AssertionError(f"{label}: result mismatch")

    base_a_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).contiguous()
    base_b_cpu = torch.arange(10, 16, dtype=torch.float32).reshape(2, 3).contiguous()
    base_a = matrixman.to_device(base_a_cpu)
    base_b = matrixman.to_device(base_b_cpu)
    def run(label, operation, expected, inputs):
        try:
            check(label, operation(), expected, inputs)
        except (NotImplementedError, RuntimeError) as exc:
            print(f"{label}: unsupported on selected backend ({exc})")

    run("contiguous", lambda: base_a * base_b, base_a_cpu * base_b_cpu, (base_a, base_b))

    transposed_a = base_a.transpose(0, 1)
    transposed_b = base_b.transpose(0, 1)
    run("transposed", lambda: transposed_a * transposed_b, base_a_cpu.T * base_b_cpu.T, (transposed_a, transposed_b))

    expanded_cpu = torch.arange(3, dtype=torch.float32).reshape(1, 3).contiguous().expand(4, 3)
    expanded_base = matrixman.to_device(torch.arange(3, dtype=torch.float32).reshape(1, 3).contiguous())
    expanded = expanded_base.expand(4, 3)
    other_cpu = torch.ones((4, 3), dtype=torch.float32)
    other = matrixman.to_device(other_cpu)
    run("zero-stride", lambda: expanded * other, expanded_cpu * other_cpu, (expanded, other))

    anchors = 24
    dbox_cpu = torch.arange(1, 1 + 4 * anchors, dtype=torch.float32).reshape(1, 4, anchors).contiguous()
    strides_cpu = torch.full((1, anchors), 8.0, dtype=torch.float32).contiguous()
    dbox = matrixman.to_device(dbox_cpu)
    strides = matrixman.to_device(strides_cpu)
    run("YOLO dbox * strides", lambda: dbox * strides, dbox_cpu * strides_cpu, (dbox, strides))

    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
