#!/usr/bin/env python3
"""Standalone legacy-CUDA NCHW Conv2D diagnostic."""

import numpy as np

try:
    from .gpumatrix import CudaExecutionBackend, print_check
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gpumatrix import CudaExecutionBackend, print_check


def reference_conv2d(input_array, weight, bias, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1):
    """Small NumPy-only reference; it is not part of the execution backend."""
    n, c, h, w = input_array.shape
    k, channels_per_group, r, s = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    out_h = (h + 2 * pad_h - dilation_h * (r - 1) - 1) // stride_h + 1
    out_w = (w + 2 * pad_w - dilation_w * (s - 1) - 1) // stride_w + 1
    output = np.zeros((n, k, out_h, out_w), dtype=np.float32)
    outputs_per_group = k // groups
    for batch in range(n):
        for channel_out in range(k):
            group = channel_out // outputs_per_group
            channel_start = group * channels_per_group
            for out_y in range(out_h):
                for out_x in range(out_w):
                    value = 0.0 if bias is None else float(bias[channel_out])
                    for channel_in in range(channels_per_group):
                        for kernel_y in range(r):
                            for kernel_x in range(s):
                                input_y = out_y * stride_h - pad_h + kernel_y * dilation_h
                                input_x = out_x * stride_w - pad_w + kernel_x * dilation_w
                                if 0 <= input_y < h and 0 <= input_x < w:
                                    value += float(
                                        input_array[
                                            batch,
                                            channel_start + channel_in,
                                            input_y,
                                            input_x,
                                        ]
                                    ) * float(weight[channel_out, channel_in, kernel_y, kernel_x])
                    output[batch, channel_out, out_y, out_x] = value
    return output


def run_diagnostic() -> int:
    input_array = np.arange(1, 1 + 2 * 4 * 5, dtype=np.float32).reshape(1, 2, 4, 5)
    weight = np.arange(1, 1 + 3 * 2 * 3 * 2, dtype=np.float32).reshape(3, 2, 3, 2) / 10
    bias = np.array([0.5, -1.0, 2.0], dtype=np.float32)
    expected = reference_conv2d(input_array, weight, bias)
    host_geometry = (expected.shape[2], expected.shape[3], expected.size)
    if host_geometry != (4, 6, 72):
        raise AssertionError(f"diagnostic geometry mismatch: expected (4, 6, 72), got {host_geometry}")
    print("Host Conv2D geometry: Hout=4, Wout=6, total_outputs=72")

    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        input_pointer = cuda.to_device(input_array)
        weight_pointer = cuda.to_device(weight)
        bias_pointer = cuda.to_device(bias)
        output_pointer = cuda.allocate(expected.nbytes)
        try:
            cuda.convolution(
                input_pointer,
                weight_pointer,
                bias_pointer,
                output_pointer,
                1, 2, 4, 5, 3, 3, 2, 4, 6,
                1, 1, 1, 1, 1, 1, 1,
            )
            actual = cuda.from_device(output_pointer, expected.shape)
            print_check("NCHW Conv2D", actual, expected)
        finally:
            for pointer in (input_pointer, weight_pointer, bias_pointer, output_pointer):
                cuda.free(pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
