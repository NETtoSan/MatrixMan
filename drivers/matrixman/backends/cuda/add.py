#!/usr/bin/env python3
"""Standalone legacy-CUDA tensor-add diagnostic."""

import numpy as np
import torch

from drivers import matrixman
from ...backend import get_backend
from ...tensor import MatrixManTensor
from .backend import CudaBackend, CudaTensorOwner

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def run_diagnostic() -> int:
    left = np.arange(1, 1 + 1 * 3 * 2 * 4, dtype=np.float32).reshape(1, 3, 2, 4)
    right = np.arange(100, 100 + left.size, dtype=np.float32).reshape(left.shape)
    expected = left + right
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        left_pointer = cuda.to_device(left)
        right_pointer = cuda.to_device(right)
        output_pointer = cuda.allocate(expected.nbytes)
        try:
            cuda.add(left_pointer, right_pointer, output_pointer, expected.size)
            print_check("NCHW add", cuda.from_device(output_pointer, expected.shape), expected)
        finally:
            for pointer in (left_pointer, right_pointer, output_pointer):
                cuda.free(pointer)

    backend = get_backend()
    if not isinstance(backend, CudaBackend):
        raise RuntimeError("MatrixMan/CUDA: select CUDA before running the add diagnostic")

    def check(label, actual, expected, inputs):
        if not isinstance(actual, MatrixManTensor) or not isinstance(actual._owner, CudaTensorOwner):
            raise AssertionError(f"{label}: result is not CUDA-backed")
        raw = actual._owner.execution.from_device(actual._owner.pointer, tuple(actual.shape))
        value = torch.from_numpy(raw)
        error = float((value - expected).abs().max())
        print(
            f"{label}: input_shapes={[list(item.shape) for item in inputs]} "
            f"input_strides={[list(item._logical_strides) for item in inputs]} "
            f"output_shape={list(actual.shape)} output_strides={list(actual._logical_strides)} "
            f"max_abs_diff={error:.6g} allclose={torch.allclose(value, expected, rtol=1e-5, atol=1e-6)} "
            f"new_output={all(actual._owner is not item._owner for item in inputs)}"
        )
        if not torch.allclose(value, expected, rtol=1e-5, atol=1e-6):
            raise AssertionError(f"{label}: mismatch")

    base_a_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).contiguous()
    base_b_cpu = torch.full((2, 3), 2.0, dtype=torch.float32).contiguous()
    base_a = matrixman.to_device(base_a_cpu)
    base_b = matrixman.to_device(base_b_cpu)
    transposed_a = base_a.transpose(0, 1)
    transposed_b = base_b.transpose(0, 1)
    check("MatrixMan transposed add", transposed_a + transposed_b, base_a_cpu.T + base_b_cpu.T, (transposed_a, transposed_b))
    check("MatrixMan transposed alpha", torch.add(transposed_a, transposed_b, alpha=0.5), base_a_cpu.T + 0.5 * base_b_cpu.T, (transposed_a, transposed_b))
    check("MatrixMan transposed negative alpha", torch.add(transposed_a, transposed_b, alpha=-2.0), base_a_cpu.T - 2.0 * base_b_cpu.T, (transposed_a, transposed_b))

    expanded_base_cpu = torch.arange(3, dtype=torch.float32).reshape(1, 3).contiguous()
    expanded_base = matrixman.to_device(expanded_base_cpu)
    expanded = expanded_base.expand(4, 3)
    expanded_other_cpu = torch.ones((4, 3), dtype=torch.float32)
    expanded_other = matrixman.to_device(expanded_other_cpu)
    check("MatrixMan zero-stride add", expanded + expanded_other, expanded_base_cpu.expand(4, 3) + expanded_other_cpu, (expanded, expanded_other))

    sx_cpu = torch.arange(24, dtype=torch.float32) + 0.5
    sy_cpu = 100.0 + torch.arange(24, dtype=torch.float32) + 0.5
    sx = matrixman.to_device(sx_cpu)
    sy = matrixman.to_device(sy_cpu)
    anchor_points = torch.stack((sx, sy), -1).view(-1, 2).transpose(0, 1).unsqueeze(0)
    rb_cpu = torch.full((1, 2, 24), 0.25, dtype=torch.float32)
    rb = matrixman.to_device(rb_cpu)
    print("YOLO anchor_points strides:", list(anchor_points._logical_strides))
    check("YOLO anchor_points + rb", anchor_points + rb, torch.stack((sx_cpu, sy_cpu), -1).view(-1, 2).T.unsqueeze(0) + rb_cpu, (anchor_points, rb))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
