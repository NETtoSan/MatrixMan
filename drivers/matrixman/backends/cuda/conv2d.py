#!/usr/bin/env python3
"""Standalone legacy-CUDA NCHW Conv2D diagnostic."""

import time

import numpy as np
import torch
import torch.nn.functional as F

from .gpumatrix import CudaExecutionBackend, print_check


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
        for label, function in (
            ("conv2d_3x3_s1_p1_c64_plane", cuda.convolution_plane_function),
            ("conv2d_3x3_s1_p1_c64_plane_2block", cuda.convolution_plane_2block_function),
            ("conv2d_3x3_s1_p1_c64_plane_256", cuda.convolution_plane_256_function),
        ):
            print(f"{label} attributes:", cuda.function_attributes(function))
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

        def run_specialized_case(height, width, variant="plane"):
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
                    specialized_3x3_plane=variant == "plane",
                    specialized_3x3_plane_legacy=variant == "plane_legacy",
                    specialized_3x3_spatial=variant == "spatial",
                    specialized_3x3_plane_2block=variant == "plane_2block",
                    specialized_3x3_plane_256=variant == "plane_256",
                )
                actual_case = cuda.from_device(output_dev, expected_case.shape)
            finally:
                for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                    cuda.free(pointer)
            error = np.abs(actual_case - expected_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 3x3 {variant} Conv2D "
                f"[1,64,{height},{width}] -> "
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
            return actual_case

        plane_results = {}
        for height, width in ((4, 4), (20, 20), (40, 40), (80, 80)):
            print(f"testing specialized plane [1,64,{height},{width}]...", flush=True)
            plane_results[(height, width)] = run_specialized_case(height, width)

        two_block_results = {}
        for height, width in ((4, 4), (20, 20), (40, 40), (80, 80)):
            print(f"testing specialized plane_2block [1,64,{height},{width}]...", flush=True)
            two_block_results[(height, width)] = run_specialized_case(height, width, "plane_2block")
        for key in sorted(set(plane_results) & set(two_block_results)):
            difference = np.abs(plane_results[key] - two_block_results[key])
            print(
                f"Specialized plane/plane_2block comparison {key[0]}x{key[1]}: "
                f"max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )

        plane_256_results = {}
        for height, width in ((4, 4), (20, 20), (40, 40), (80, 80)):
            print(f"testing specialized plane_256 [1,64,{height},{width}]...", flush=True)
            plane_256_results[(height, width)] = run_specialized_case(height, width, "plane_256")
        for key in sorted(set(plane_results) & set(plane_256_results)):
            difference = np.abs(plane_results[key] - plane_256_results[key])
            print(
                f"Specialized plane/plane_256 comparison {key[0]}x{key[1]}: "
                f"max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )

        def benchmark_specialized_variants(height, width, variants, repetitions=20):
            rng = np.random.RandomState(9100 + height * 100 + width)
            input_case = rng.uniform(-1.0, 1.0, (1, 64, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (64, 64, 3, 3)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (64,)).astype(np.float32)
            input_dev = cuda.to_device(input_case)
            weight_dev = cuda.to_device(weight_case)
            bias_dev = cuda.to_device(bias_case)
            output_dev = cuda.allocate(input_case.nbytes)
            try:
                for variant in variants:
                    def launch():
                        cuda.convolution(
                            input_dev, weight_dev, bias_dev, output_dev,
                            1, 64, height, width, 64, 3, 3, height, width,
                            1, 1, 1, 1, 1, 1, 1,
                            specialized_3x3_plane=variant == "plane",
                            specialized_3x3_plane_2block=variant == "plane_2block",
                            specialized_3x3_plane_256=variant == "plane_256",
                        )

                    for _ in range(3):
                        launch()
                    started = time.perf_counter()
                    for _ in range(repetitions):
                        launch()
                    elapsed_ms = (time.perf_counter() - started) * 1000.0 / repetitions
                    macs = 64 * height * width * 64 * 9
                    gmacs = macs / (elapsed_ms / 1000.0) / 1e9
                    print(
                        f"Benchmark {variant} [1,64,{height},{width}]: "
                        f"avg_kernel_ms={elapsed_ms:.3f} GMAC/s={gmacs:.3f} "
                        f"repetitions={repetitions}"
                    )
            finally:
                for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                    cuda.free(pointer)

        for height, width in ((20, 20), (40, 40), (80, 80)):
            benchmark_specialized_variants(height, width, ("plane", "plane_2block", "plane_256"))


        legacy_results = {}
        for height, width in ((4, 4), (40, 40), (80, 80)):
            print(f"testing specialized plane_legacy [1,64,{height},{width}]...", flush=True)
            legacy_results[(height, width)] = run_specialized_case(height, width, "plane_legacy")

        spatial_results = {}
        for height, width in ((4, 4), (20, 20), (40, 40), (80, 80)):
            print(f"testing specialized spatial [1,64,{height},{width}]...", flush=True)
            spatial_results[(height, width)] = run_specialized_case(height, width, "spatial")
        for key in sorted(set(plane_results) & set(spatial_results)):
            difference = np.abs(plane_results[key] - spatial_results[key])
            print(
                f"Specialized plane/spatial comparison {key[0]}x{key[1]}: "
                f"max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )
        for key in sorted(set(plane_results) & set(legacy_results)):
            difference = np.abs(plane_results[key] - legacy_results[key])
            print(
                f"Specialized plane/plane_legacy comparison {key[0]}x{key[1]}: "
                f"max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )

        def run_small_channel_case(channels, height, width):
            rng = np.random.RandomState(8600 + channels * 1000 + height * 100 + width)
            input_case = rng.uniform(-1.0, 1.0, (1, channels, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (channels, channels, 3, 3)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (channels,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case), torch.from_numpy(weight_case),
                torch.from_numpy(bias_case), stride=1, padding=1,
            ).numpy()

            def execute(specialized):
                input_dev = cuda.to_device(input_case)
                weight_dev = cuda.to_device(weight_case)
                bias_dev = cuda.to_device(bias_case)
                output_dev = cuda.allocate(expected_case.nbytes)
                try:
                    cuda.convolution(
                        input_dev, weight_dev, bias_dev, output_dev,
                        1, channels, height, width, channels, 3, 3, height, width,
                        1, 1, 1, 1, 1, 1, 1,
                        specialized_3x3_small_c8=(specialized and channels == 8),
                        specialized_3x3_small_c10=(specialized and channels == 10),
                        specialized_3x3_small_c12=(specialized and channels == 12),
                        specialized_3x3_small_c24=(specialized and channels == 24),
                    )
                    return cuda.from_device(output_dev, expected_case.shape)
                finally:
                    for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                        cuda.free(pointer)

            actual_case = execute(True)
            generic_case = execute(False)
            error = np.abs(actual_case - expected_case)
            generic_error = np.abs(generic_case - expected_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 3x3 small-c{channels} [1,{channels},{height},{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"matches CPU reference: {bool(matches)}"
            )
            print(
                f"Generic/specialized small-c{channels} comparison [1,{channels},{height},{width}]: "
                f"max_abs_diff={np.abs(actual_case - generic_case).max():.6g} "
                f"generic_max_abs_diff={generic_error.max():.6g}"
            )
            if not matches or not np.allclose(actual_case, generic_case, rtol=5e-4, atol=5e-4):
                raise AssertionError(f"specialized small-c{channels} Conv2D does not match reference/generic CUDA")

        for channels, sizes in (
            (8, ((4, 4), (20, 20), (40, 40), (80, 80))),
            (10, ((4, 4), (20, 20), (40, 40), (80, 80))),
            (12, ((4, 4), (20, 20), (40, 40))),
            (24, ((4, 4), (20, 20), (40, 40))),
        ):
            for height, width in sizes:
                print(f"testing specialized small-c{channels} [1,{channels},{height},{width}]...", flush=True)
                run_small_channel_case(channels, height, width)

        def run_fixed_cin_c64_plane_case(cin, height, width):
            rng = np.random.RandomState(8100 + cin * 1000 + height * 100 + width)
            input_case = rng.uniform(-1.0, 1.0, (1, cin, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (64, cin, 3, 3)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (64,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case), torch.from_numpy(weight_case),
                torch.from_numpy(bias_case), stride=1, padding=1,
            ).numpy()
            input_dev = cuda.to_device(input_case)
            weight_dev = cuda.to_device(weight_case)
            bias_dev = cuda.to_device(bias_case)
            output_dev = cuda.allocate(expected_case.nbytes)
            try:
                cuda.convolution(
                    input_dev, weight_dev, bias_dev, output_dev,
                    1, cin, height, width, 64, 3, 3, height, width,
                    1, 1, 1, 1, 1, 1, 1,
                    specialized_3x3_c8_c64_plane=(cin == 8),
                    specialized_3x3_c24_c64_plane=(cin == 24),
                    specialized_3x3_c48_c64_plane=(cin == 48),
                )
                actual_case = cuda.from_device(output_dev, expected_case.shape)
            finally:
                for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                    cuda.free(pointer)
            error = np.abs(actual_case - expected_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 3x3 c{cin}->c64 plane [{height}x{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"matches CPU reference: {matches}"
            )
            if not matches:
                raise AssertionError(f"specialized c{cin}->c64 Conv2D does not match CPU reference")

        for cin, sizes in ((8, ((4, 4), (20, 20), (40, 40), (80, 80))),
                           (24, ((4, 4), (20, 20), (40, 40))),
                           (48, ((4, 4), (20, 20), (40, 40)))):
            for height, width in sizes:
                print(f"testing specialized c{cin}->c64 plane [{height}x{width}]...", flush=True)
                run_fixed_cin_c64_plane_case(cin, height, width)

        def run_specialized_1x1_case(height, width):
            rng = np.random.RandomState(7000 + height * 100 + width)
            input_case = rng.uniform(-1.0, 1.0, (1, 64, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (64, 64, 1, 1)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (64,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case),
                torch.from_numpy(weight_case),
                torch.from_numpy(bias_case),
            ).numpy()
            input_dev = cuda.to_device(input_case)
            weight_dev = cuda.to_device(weight_case)
            bias_dev = cuda.to_device(bias_case)
            output_dev = cuda.allocate(expected_case.nbytes)
            try:
                cuda.convolution(
                    input_dev, weight_dev, bias_dev, output_dev,
                    1, 64, height, width, 64, 1, 1, height, width,
                    1, 1, 0, 0, 1, 1, 1,
                    specialized_1x1=True,
                )
                actual_case = cuda.from_device(output_dev, expected_case.shape)
            finally:
                for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                    cuda.free(pointer)
            error = np.abs(actual_case - expected_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 1x1 c64 Conv2D [1,64,{height},{width}] -> "
                f"[1,64,{height},{width}]: max_abs_diff={error.max():.6g} "
                f"mean_abs_diff={error.mean():.6g} matches CPU reference: {matches}"
            )
            if not matches:
                raise AssertionError("specialized 1x1 Conv2D does not match CPU reference")

        for height, width in ((80, 80), (40, 40), (20, 20)):
            print(f"testing specialized 1x1 c64 [1,64,{height},{width}]...", flush=True)
            run_specialized_1x1_case(height, width)

        def run_specialized_1x1_cin24_case(height, width, output_channels):
            rng = np.random.RandomState(7100 + height * 100 + width + output_channels)
            input_case = rng.uniform(-1.0, 1.0, (1, 24, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (output_channels, 24, 1, 1)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (output_channels,)).astype(np.float32)
            input_tensor = torch.from_numpy(input_case)
            expected_case = F.conv2d(
                input_tensor, torch.from_numpy(weight_case), torch.from_numpy(bias_case)
            ).numpy()

            def execute(specialized):
                input_dev = cuda.to_device(input_case)
                weight_dev = cuda.to_device(weight_case)
                bias_dev = cuda.to_device(bias_case)
                output_dev = cuda.allocate(expected_case.nbytes)
                try:
                    cuda.convolution(
                        input_dev, weight_dev, bias_dev, output_dev,
                        1, 24, height, width, output_channels, 1, 1, height, width,
                        1, 1, 0, 0, 1, 1, 1,
                        specialized_1x1_cin24=specialized,
                    )
                    return cuda.from_device(output_dev, expected_case.shape)
                finally:
                    for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                        cuda.free(pointer)

            actual_case = execute(True)
            generic_case = execute(False)
            error = np.abs(actual_case - expected_case)
            difference = np.abs(actual_case - generic_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 1x1 cin24 [1,24,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"CPU match: {bool(matches)} "
                f"specialized-vs-generic max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )
            if not matches or not np.allclose(actual_case, generic_case, rtol=5e-4, atol=5e-4):
                raise AssertionError("specialized 1x1 cin24 Conv2D does not match CPU/generic reference")

        for height, width, output_channels in (
            (80, 80, 16), (40, 40, 24), (40, 40, 8), (4, 4, 16)
        ):
            print(
                f"testing specialized 1x1 cin24 [1,24,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]...", flush=True
            )
            run_specialized_1x1_cin24_case(height, width, output_channels)

        def run_specialized_1x1_cin48_case(height, width, output_channels):
            rng = np.random.RandomState(7200 + height * 100 + width + output_channels)
            input_case = rng.uniform(-1.0, 1.0, (1, 48, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (output_channels, 48, 1, 1)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (output_channels,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case), torch.from_numpy(weight_case),
                torch.from_numpy(bias_case)
            ).numpy()

            def execute(specialized):
                input_dev = cuda.to_device(input_case)
                weight_dev = cuda.to_device(weight_case)
                bias_dev = cuda.to_device(bias_case)
                output_dev = cuda.allocate(expected_case.nbytes)
                try:
                    cuda.convolution(
                        input_dev, weight_dev, bias_dev, output_dev,
                        1, 48, height, width, output_channels, 1, 1, height, width,
                        1, 1, 0, 0, 1, 1, 1,
                        specialized_1x1_cin48=specialized,
                    )
                    return cuda.from_device(output_dev, expected_case.shape)
                finally:
                    for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                        cuda.free(pointer)

            actual_case = execute(True)
            generic_case = execute(False)
            error = np.abs(actual_case - expected_case)
            difference = np.abs(actual_case - generic_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 1x1 cin48 [1,48,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"CPU match: {bool(matches)} "
                f"specialized-vs-generic max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )
            if not matches or not np.allclose(actual_case, generic_case, rtol=5e-4, atol=5e-4):
                raise AssertionError("specialized 1x1 cin48 Conv2D does not match CPU/generic reference")

        for height, width, output_channels in (
            (40, 40, 24), (20, 20, 48), (20, 20, 24), (4, 4, 24)
        ):
            print(
                f"testing specialized 1x1 cin48 [1,48,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]...", flush=True
            )
            run_specialized_1x1_cin48_case(height, width, output_channels)

        def run_specialized_1x1_cin36_case(height, width, output_channels):
            rng = np.random.RandomState(7300 + height * 100 + width + output_channels)
            input_case = rng.uniform(-1.0, 1.0, (1, 36, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (output_channels, 36, 1, 1)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (output_channels,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case), torch.from_numpy(weight_case),
                torch.from_numpy(bias_case)
            ).numpy()

            def execute(specialized):
                input_dev = cuda.to_device(input_case)
                weight_dev = cuda.to_device(weight_case)
                bias_dev = cuda.to_device(bias_case)
                output_dev = cuda.allocate(expected_case.nbytes)
                try:
                    cuda.convolution(
                        input_dev, weight_dev, bias_dev, output_dev,
                        1, 36, height, width, output_channels, 1, 1, height, width,
                        1, 1, 0, 0, 1, 1, 1,
                        specialized_1x1_cin36=specialized,
                    )
                    return cuda.from_device(output_dev, expected_case.shape)
                finally:
                    for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                        cuda.free(pointer)

            actual_case = execute(True)
            generic_case = execute(False)
            error = np.abs(actual_case - expected_case)
            difference = np.abs(actual_case - generic_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 1x1 cin36 [1,36,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"CPU match: {bool(matches)} "
                f"specialized-vs-generic max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )
            if not matches or not np.allclose(actual_case, generic_case, rtol=5e-4, atol=5e-4):
                raise AssertionError("specialized 1x1 cin36 Conv2D does not match CPU/generic reference")

        for height, width, output_channels in ((40, 40, 24), (4, 4, 24)):
            print(
                f"testing specialized 1x1 cin36 [1,36,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]...", flush=True
            )
            run_specialized_1x1_cin36_case(height, width, output_channels)

        def run_specialized_1x1_cin16_case(height, width, output_channels, specialized=True):
            rng = np.random.RandomState(7400 + height * 100 + width + output_channels)
            input_case = rng.uniform(-1.0, 1.0, (1, 16, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (output_channels, 16, 1, 1)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (output_channels,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case), torch.from_numpy(weight_case),
                torch.from_numpy(bias_case)
            ).numpy()

            def execute(specialized):
                input_dev = cuda.to_device(input_case)
                weight_dev = cuda.to_device(weight_case)
                bias_dev = cuda.to_device(bias_case)
                output_dev = cuda.allocate(expected_case.nbytes)
                try:
                    cuda.convolution(
                        input_dev, weight_dev, bias_dev, output_dev,
                        1, 16, height, width, output_channels, 1, 1, height, width,
                        1, 1, 0, 0, 1, 1, 1,
                        specialized_1x1_cin16=specialized,
                    )
                    return cuda.from_device(output_dev, expected_case.shape)
                finally:
                    for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                        cuda.free(pointer)

            actual_case = execute(True)
            generic_case = execute(False)
            error = np.abs(actual_case - expected_case)
            difference = np.abs(actual_case - generic_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"{'Specialized' if specialized else 'Generic'} 1x1 cin16 [1,16,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"CPU match: {bool(matches)} "
                f"specialized-vs-generic max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )
            if not matches or not np.allclose(actual_case, generic_case, rtol=5e-4, atol=5e-4):
                raise AssertionError("specialized 1x1 cin16 Conv2D does not match CPU/generic reference")

        for height, width, output_channels in ((80, 80, 16), (4, 4, 16)):
            print(
                f"testing specialized 1x1 cin16 [1,16,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]...", flush=True
            )
            run_specialized_1x1_cin16_case(height, width, output_channels)

        print("testing generic 1x1 cin16 Cout=1 fallback [1,16,4,4] -> [1,1,4,4]...", flush=True)
        run_specialized_1x1_cin16_case(4, 4, 1, specialized=False)

        # Nearby 1x1 shapes must continue to use the generic kernel.
        input_case = np.arange(1, 1 + 32 * 4 * 4, dtype=np.float32).reshape(1, 32, 4, 4) / 100
        weight_case = np.arange(1, 1 + 64 * 32, dtype=np.float32).reshape(64, 32, 1, 1) / 100
        bias_case = np.arange(64, dtype=np.float32) / 100
        expected_case = F.conv2d(
            torch.from_numpy(input_case),
            torch.from_numpy(weight_case),
            torch.from_numpy(bias_case),
        ).numpy()
        print("testing generic 1x1 fallback [1,32,4,4] -> [1,64,4,4]...", flush=True)
        input_dev = cuda.to_device(input_case)
        weight_dev = cuda.to_device(weight_case)
        bias_dev = cuda.to_device(bias_case)
        output_dev = cuda.allocate(expected_case.nbytes)
        try:
            cuda.convolution(
                input_dev, weight_dev, bias_dev, output_dev,
                1, 32, 4, 4, 64, 1, 1, 4, 4,
                1, 1, 0, 0, 1, 1, 1,
            )
            actual_case = cuda.from_device(output_dev, expected_case.shape)
        finally:
            for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                cuda.free(pointer)
        error = np.abs(actual_case - expected_case)
        matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
        print(
            f"Generic fallback max_abs_diff={error.max():.6g} "
            f"mean_abs_diff={error.mean():.6g} matches CPU reference: {matches}"
        )
        if not matches:
            raise AssertionError("generic 1x1 fallback does not match CPU reference")

        def run_specialized_1x1_cin72_case(height, width, output_channels):
            rng = np.random.RandomState(7500 + height * 100 + width + output_channels)
            input_case = rng.uniform(-1.0, 1.0, (1, 72, height, width)).astype(np.float32)
            weight_case = rng.uniform(-0.25, 0.25, (output_channels, 72, 1, 1)).astype(np.float32)
            bias_case = rng.uniform(-0.1, 0.1, (output_channels,)).astype(np.float32)
            expected_case = F.conv2d(
                torch.from_numpy(input_case), torch.from_numpy(weight_case),
                torch.from_numpy(bias_case)
            ).numpy()

            def execute(specialized):
                input_dev = cuda.to_device(input_case)
                weight_dev = cuda.to_device(weight_case)
                bias_dev = cuda.to_device(bias_case)
                output_dev = cuda.allocate(expected_case.nbytes)
                try:
                    cuda.convolution(
                        input_dev, weight_dev, bias_dev, output_dev,
                        1, 72, height, width, output_channels, 1, 1, height, width,
                        1, 1, 0, 0, 1, 1, 1,
                        specialized_1x1_cin72=specialized,
                    )
                    return cuda.from_device(output_dev, expected_case.shape)
                finally:
                    for pointer in (input_dev, weight_dev, bias_dev, output_dev):
                        cuda.free(pointer)

            actual_case = execute(True)
            generic_case = execute(False)
            error = np.abs(actual_case - expected_case)
            difference = np.abs(actual_case - generic_case)
            matches = np.allclose(actual_case, expected_case, rtol=5e-4, atol=5e-4)
            print(
                f"Specialized 1x1 cin72 [1,72,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]: "
                f"max_abs_diff={error.max():.6g} mean_abs_diff={error.mean():.6g} "
                f"CPU match: {bool(matches)} "
                f"specialized-vs-generic max_abs_diff={difference.max():.6g} "
                f"mean_abs_diff={difference.mean():.6g}"
            )
            if not matches or not np.allclose(actual_case, generic_case, rtol=5e-4, atol=5e-4):
                raise AssertionError("specialized 1x1 cin72 Conv2D does not match CPU/generic reference")

        for height, width, output_channels in ((20, 20, 48), (4, 4, 48)):
            print(
                f"testing specialized 1x1 cin72 [1,72,{height},{width}] -> "
                f"[1,{output_channels},{height},{width}]...", flush=True
            )
            run_specialized_1x1_cin72_case(height, width, output_channels)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
