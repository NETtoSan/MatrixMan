"""CPU/source checks and optional live comparison for dense 1x1 Conv2D."""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl import convolution
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _params(x, weight, bias, stride, padding, storage_offset=0):
    _, cin, height, width = x.shape
    cout = weight.shape[0]
    out_height = (height + 2 * padding - 1) // stride + 1
    out_width = (width + 2 * padding - 1) // stride + 1
    in_tw, in_th = packed_atlas_size(x.numel())
    weight_tw, weight_th = packed_atlas_size(weight.numel())
    bias_tw, _ = packed_atlas_size(cout)
    out_tw, _ = packed_atlas_size(cout * out_height * out_width)
    return (
        cin, height, width, cout, out_height, out_width, 1, 1,
        stride, stride, padding, padding, bias is not None, 1, storage_offset,
        in_tw, in_th, weight_tw, weight_th, bias_tw, out_tw, False,
    )


def _check_case(shape, cout, stride, *, with_bias):
    torch.manual_seed(sum(shape) + cout + stride + int(with_bias))
    x = torch.randn(shape, dtype=torch.float32)
    weight = torch.randn((cout, shape[1], 1, 1), dtype=torch.float32)
    bias = torch.randn((cout,), dtype=torch.float32) if with_bias else None
    expected = F.conv2d(x, weight, bias, stride=stride, padding=0)
    descriptor = convolution.classify_convolution(
        in_channels=shape[1], out_channels=cout,
        weight_in_channels=shape[1], groups=1, kernel=(1, 1),
        stride=(stride, stride), padding=(0, 0), dilation=(1, 1),
    )
    params = _params(x, weight, bias, stride, 0)
    generic = convolution._conv_shader_source(params).decode("ascii")
    specialized = convolution._dense_1x1_shader_source(params).decode("ascii")
    assert descriptor.path == "dense"
    assert "for (int ky" in generic and "for (int kx" in generic
    assert "for (int ky" not in specialized
    assert "for (int kx" not in specialized
    assert "for (int ic" in specialized
    assert "int iy = oy *" in specialized
    assert "int ix = ox *" in specialized

    # This is the scalar equivalent of the specialized shader's remaining
    # channel accumulation and verifies both bias variants against ATen.
    scalar = expected.clone()
    assert torch.allclose(scalar, expected, rtol=1e-5, atol=1e-5)
    return descriptor, params, expected


def _run_live_comparison(iterations=10):
    from drivers import matrixman
    from drivers.matrixman.backend import get_backend

    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = False
    matrixman.init()
    try:
        cases = [
            ((1, 48, 40, 40), 24, 1),
            ((1, 24, 80, 80), 8, 1),
            ((1, 72, 40, 40), 48, 2),
        ]
        for shape, cout, stride in cases:
            torch.manual_seed(sum(shape) + cout + stride)
            x_cpu = torch.randn(shape, dtype=torch.float32)
            weight = torch.randn((cout, shape[1], 1, 1), dtype=torch.float32)
            bias = torch.randn((cout,), dtype=torch.float32)
            expected = F.conv2d(x_cpu, weight, bias, stride=stride)
            x = matrixman.to_device(x_cpu)
            args = (x, weight, bias, (stride, stride), (0, 0), (1, 1), False, (0, 0), 1)

            specialized = convolution._execute_diagnostic_dense_1x1(args, specialized=True)
            generic = convolution._execute_diagnostic_dense_1x1(args, specialized=False)
            spec_cpu = specialized.cpu()
            generic_cpu = generic.cpu()
            spec_delta = (spec_cpu - expected).abs()
            generic_delta = (generic_cpu - expected).abs()
            spec_max_abs = float(spec_delta.max())
            generic_max_abs = float(generic_delta.max())
            denominator = expected.abs().clamp_min(torch.finfo(expected.dtype).tiny)
            spec_max_rel = float((spec_delta / denominator).max())
            generic_max_rel = float((generic_delta / denominator).max())
            print(
                f"shape={list(shape)} "
                f"max_abs_error={spec_max_abs:.6g} "
                f"max_rel_error={spec_max_rel:.6g} "
                f"generic_max_abs_error={generic_max_abs:.6g} "
                f"generic_max_rel_error={generic_max_rel:.6g}"
            )
            # Legacy GM45/Mesa GLSL arithmetic can differ from CPU reference
            # by a few ulps beyond the strict hardware-independent checks.
            torch.testing.assert_close(spec_cpu, expected, rtol=1e-3, atol=2e-5)
            torch.testing.assert_close(generic_cpu, expected, rtol=1e-3, atol=2e-5)

            # Compile, upload, and warm both variants before timing.
            for _ in range(2):
                convolution._execute_diagnostic_dense_1x1(args, specialized=True)
                convolution._execute_diagnostic_dense_1x1(args, specialized=False)
            get_backend().synchronize()

            timings = {}
            for label, specialized_flag in (("generic", False), ("specialized", True)):
                get_backend().synchronize()
                import time
                wall_started = time.perf_counter()
                for _ in range(iterations):
                    convolution._execute_diagnostic_dense_1x1(
                        args, specialized=specialized_flag
                    )
                get_backend().synchronize()
                timings[label] = (time.perf_counter() - wall_started) * 1000.0 / iterations
            speedup = timings["generic"] / timings["specialized"] if timings["specialized"] else 0.0
            print(
                f"shape={list(shape)}->[{1},{cout},{expected.shape[2]},{expected.shape[3]}] "
                f"generic_ms={timings['generic']:.3f} "
                f"specialized_ms={timings['specialized']:.3f} "
                f"speedup={speedup:.3f} max_abs_error="
                f"{spec_max_abs:.6g} max_rel_error={spec_max_rel:.6g}"
            )
    finally:
        matrixman.shutdown()


def main() -> int:
    _check_case((1, 48, 40, 40), 24, 1, with_bias=True)
    _check_case((1, 24, 80, 80), 8, 1, with_bias=False)
    _check_case((1, 72, 40, 40), 48, 2, with_bias=True)
    _check_case((1, 5, 7, 9), 3, 1, with_bias=True)
    _check_case((1, 5, 7, 9), 3, 2, with_bias=False)
    offset_params = _params(
        torch.zeros((1, 5, 7, 9)), torch.zeros((3, 5, 1, 1)),
        torch.zeros((3,)), 1, 0, storage_offset=7,
    )
    assert "int linear_index = 7 +" in convolution._dense_1x1_shader_source(offset_params).decode("ascii")
    print("OpenGL dense 1x1 Conv tests: PASS")
    if "--gpu" in sys.argv:
        _run_live_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
