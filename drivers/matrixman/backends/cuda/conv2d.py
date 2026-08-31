#!/usr/bin/env python3
"""Standalone legacy-CUDA NCHW Conv2D diagnostic."""

import numpy as np
import torch
import torch.nn.functional as F

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

        # Regression case: two output channels per input group.  This catches
        # using the group-local output-channel remainder to index [Cout, ...]
        # weights, which can look correct for output channel zero.
        input_array = np.empty((1, 8, 16, 16), dtype=np.float32)
        for channel in range(8):
            input_array[0, channel] = channel * 100.0 + np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
        weight = np.empty((16, 1, 3, 3), dtype=np.float32)
        for output_channel in range(16):
            weight[output_channel, 0] = (output_channel + 1) * np.arange(1, 10, dtype=np.float32).reshape(3, 3) / 10.0
        expected = reference_conv2d(input_array, weight, None, stride=(2, 2), padding=(1, 1), groups=8)
        print("Grouped multiplier-2 Conv2D: input=[1,8,16,16], weight=[16,1,3,3], groups=8")
        input_pointer = cuda.to_device(input_array)
        weight_pointer = cuda.to_device(weight)
        output_pointer = cuda.allocate(expected.nbytes)
        try:
            cuda.convolution(
                input_pointer, weight_pointer, type(input_pointer)(), output_pointer,
                1, 8, 16, 16, 16, 3, 3, 8, 8,
                2, 2, 1, 1, 1, 1, 8,
            )
            actual = cuda.from_device(output_pointer, expected.shape)
            for output_channel in range(16):
                group = output_channel // (16 // 8)
                source_channel = group * (8 // 8)
                error = np.abs(actual[0, output_channel] - expected[0, output_channel])
                status = "PASS" if np.allclose(actual[0, output_channel], expected[0, output_channel], rtol=5e-4, atol=5e-4) else "FAIL"
                print(
                    f"  oc={output_channel:2d} expected_group={group} "
                    f"source_input_channel={source_channel} "
                    f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} {status}"
                )
        finally:
            for pointer in (input_pointer, weight_pointer, output_pointer):
                cuda.free(pointer)

        def run_specialized_case(height, width):
            rng = np.random.RandomState(height * 100 + width)
            input_case = rng.uniform(-1.0, 1.0, (1, 64, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (64, 64, 3, 3)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (64,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case),
                torch.from_numpy(weight_case),
                torch.from_numpy(bias_case),
                stride=1,
                padding=1,
            ).numpy()
            input_dev = cuda.to_device(input_case)
            weight_dev = cuda.to_device(weight_case)
            bias_dev = cuda.to_device(bias_case)
            output_dev = cuda.allocate(expected_case.nbytes)
            try:
                cuda.convolution(
                    input_dev, weight_dev, bias_dev, output_dev,
                    1, 64, height, width, 64, 3, 3, height, width,
                    1, 1, 1, 1, 1, 1, 1,
                    specialized=True,
                )
                actual_case = cuda.from_device(output_dev, expected_case.shape)
            finally:
                for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                    cuda.free(pointer)
            error = np.abs(actual_case - expected_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 3x3 Conv2D [1,64,{height},{width}] -> "
                f"[1,64,{height},{width}]: max_abs_diff={error.max():.6g} "
                f"mean_abs_diff={error.mean():.6g} matches CPU reference: {matches}"
            )
            if height == 4 and width == 4:
                for channel in range(64):
                    channel_error = error[0, channel]
                    channel_ok = np.allclose(
                        actual_case[0, channel], expected_case[0, channel],
                        rtol=5e-4, atol=5e-4,
                    )
                    print(
                        f"  channel={channel:2d} max_abs_diff={channel_error.max():.6g} "
                        f"mean_abs_diff={channel_error.mean():.6g} "
                        f"{'PASS' if channel_ok else 'FAIL'}"
                    )
                print("  corner samples:", actual_case[0, :2, :2, :2])

        for height, width in ((4, 4), (40, 40), (80, 80)):
            print(f"testing specialized [1,64,{height},{width}]...", flush=True)
            run_specialized_case(height, width)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
