#!/usr/bin/env python3
"""Standalone legacy-CUDA nearest-neighbor resize diagnostic."""

import numpy as np

from .gpumatrix import CudaExecutionBackend, print_check


def nearest_reference(values, output_height, output_width):
    input_height, input_width = values.shape[-2:]
    y = (np.arange(output_height) * input_height // output_height).astype(np.int64)
    x = (np.arange(output_width) * input_width // output_width).astype(np.int64)
    return values[:, :, y[:, None], x[None, :]]


def check_resize(cuda, values, output_height, output_width, label):
    expected = nearest_reference(values, output_height, output_width)
    input_pointer = cuda.to_device(values)
    output_pointer = cuda.allocate(expected.nbytes)
    try:
        cuda.upsample_nearest2d(
            input_pointer,
            output_pointer,
            expected.size,
            values.shape[1],
            values.shape[2],
            values.shape[3],
            output_height,
            output_width,
        )
        actual = cuda.from_device(output_pointer, expected.shape)
        try:
            print_check(label, actual, expected)
            return True
        except RuntimeError as error:
            print(f"{label} failed; continuing diagnostics: {error}")
            return False
    finally:
        cuda.free(input_pointer)
        cuda.free(output_pointer)


def run_diagnostic() -> int:
    exact_input = np.arange(1, 1 + 1 * 2 * 2 * 3, dtype=np.float32).reshape(1, 2, 2, 3)
    non_integer_input = np.arange(1, 1 + 1 * 1 * 3 * 4, dtype=np.float32).reshape(1, 1, 3, 4)
    with CudaExecutionBackend() as cuda:
        print("CUDA device:", cuda.info["name"])
        print("compute capability:", cuda.info["compute_capability"])
        print("device memory (MiB):", cuda.info["memory_mib"])
        passed = check_resize(cuda, exact_input, 4, 6, "NCHW nearest 2x resize")
        passed = check_resize(cuda, non_integer_input, 5, 7, "NCHW nearest non-integer resize") and passed

        # Regression case for NCHW plane addressing: every channel has a
        # distinct range so a missing Hin*Win plane stride is obvious.
        values = np.empty((1, 24, 4, 4), dtype=np.float32)
        for channel in range(24):
            values[0, channel] = channel * 1000.0 + np.arange(16, dtype=np.float32).reshape(4, 4)
        expected = nearest_reference(values, 8, 8)
        print("NCHW multi-channel nearest 2x: input=[1,24,4,4], output=[1,24,8,8]")
        input_pointer = cuda.to_device(values)
        output_pointer = cuda.allocate(expected.nbytes)
        try:
            cuda.upsample_nearest2d(
                input_pointer, output_pointer, expected.size, 24, 4, 4, 8, 8
            )
            actual = cuda.from_device(output_pointer, expected.shape)
            print("  address probes (NCHW contiguous input):")
            for n, channel, output_y, output_x in ((0, 0, 0, 0), (0, 1, 0, 0), (0, 1, 2, 0)):
                input_y = output_y * values.shape[2] // 8
                input_x = output_x * values.shape[3] // 8
                expected_index = (
                    ((n * values.shape[1] + channel) * values.shape[2] + input_y) * values.shape[3]
                    + input_x
                )
                actual_value = float(actual[n, channel, output_y, output_x])
                matches = np.flatnonzero(values.reshape(-1) == actual_value)
                actual_index = int(matches[0]) if matches.size else None
                print(
                    f"    n={n} c={channel} oy={output_y} ox={output_x}: "
                    f"expected_input_index={expected_index} "
                    f"expected_value={float(values.reshape(-1)[expected_index]):.6g} "
                    f"actual_value={actual_value:.6g} "
                    f"actual_matching_input_index={actual_index}"
                )
            for channel in range(24):
                error = np.abs(actual[0, channel] - expected[0, channel])
                status = "PASS" if np.allclose(actual[0, channel], expected[0, channel], rtol=1e-5, atol=1e-5) else "FAIL"
                passed = passed and status == "PASS"
                print(
                    f"  channel={channel:2d} max_abs_diff={error.max():.6g} "
                    f"mean_abs_diff={error.mean():.6g} {status}"
                )
            print("  samples:")
            for channel in (0, 1, 2, 23):
                print(f"    channel {channel} CUDA: {actual[0, channel].reshape(-1)[:8]}")
                print(f"    channel {channel} CPU:  {expected[0, channel].reshape(-1)[:8]}")
        finally:
            cuda.free(input_pointer)
            cuda.free(output_pointer)
    if not passed:
        raise RuntimeError("one or more CUDA upsample diagnostic cases failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_diagnostic())
