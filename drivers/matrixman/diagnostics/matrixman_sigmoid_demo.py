#!/usr/bin/env python3
"""Deterministic selected-backend MatrixMan sigmoid diagnostic."""

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
    print("aten.sigmoid.default schema:", torch.ops.aten.sigmoid.default._schema)
    info = backend.device_info()
    if info.get("device"):
        print("device:", info["device"])
    if info.get("compute_capability"):
        print("compute capability:", info["compute_capability"])

    def run(label, source, expected):
        try:
            result = torch.sigmoid(source)
            if not isinstance(result, matrixman.MatrixManTensor):
                raise AssertionError(f"{label}: result is not a MatrixMan tensor")
            actual = readback_tensor(result)
            error = float((actual - expected).abs().max()) if actual.numel() else 0.0
            close = torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
            print(
                f"{label}: input_shape={list(source.shape)} input_strides={list(source._logical_strides)} "
                f"output_shape={list(result.shape)} output_strides={list(result._logical_strides)} "
                f"max_abs_diff={error:.6g} allclose={close} storage={describe_storage(result)}"
            )
            if not close:
                raise AssertionError(f"{label}: result mismatch")
        except (NotImplementedError, RuntimeError) as exc:
            print(f"{label}: unsupported on selected backend ({exc})")

    values = torch.tensor([[-20.0, -2.0, -0.5, 0.0, 0.5, 2.0, 20.0]], dtype=torch.float32)
    source = matrixman.to_device(values)
    run("contiguous", source, torch.sigmoid(values))

    scores = torch.arange(1, 1 + 80 * 84, dtype=torch.float32).reshape(1, 80, 84).contiguous() / 10.0
    score_tensor = matrixman.to_device(scores)
    run("YOLO scores [1,80,84]", score_tensor, torch.sigmoid(scores))

    # CUDA supports logical strides; OpenGL's historical sigmoid shader is
    # intentionally left to report this layout as unsupported if selected.
    if backend.name == "cuda":
        expanded_cpu = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24).contiguous().expand(1, 2, 24)
        expanded = matrixman.to_device(torch.arange(24, dtype=torch.float32).reshape(1, 1, 24).contiguous()).expand(1, 2, 24)
        run("zero-stride", expanded, torch.sigmoid(expanded_cpu))

    matrixman.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
