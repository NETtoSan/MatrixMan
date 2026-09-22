"""CPU and shader-source checks for the OpenGL depthwise Conv2D path.

The numerical checks are hardware-independent.  The generated GLSL is checked
against the generic grouped source so this remains useful on machines without
an OpenGL 2.1 context; the same helpers are used by the runtime shader path.
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from drivers.matrixman.backends.opengl import convolution
from drivers.matrixman.backends.opengl.storage import packed_atlas_size


def _params(x, weight, bias, stride, padding):
    _, cin, height, width = x.shape
    cout = weight.shape[0]
    out_height = (height + 2 * padding - 3) // stride + 1
    out_width = (width + 2 * padding - 3) // stride + 1
    in_tw, in_th = packed_atlas_size(x.numel())
    weight_tw, weight_th = packed_atlas_size(weight.numel())
    bias_tw, _ = packed_atlas_size(bias.numel())
    out_tw, _ = packed_atlas_size(cout * out_height * out_width)
    return (
        cin, height, width, cout, out_height, out_width, 3, 3,
        stride, stride, padding, padding, True, cin, 0,
        in_tw, in_th, weight_tw, weight_th, bias_tw, out_tw, False,
    )


def _explicit_depthwise_reference(x, weight, bias, stride, padding):
    """Reference the specialized oc -> ic mapping one output channel at a time."""
    cin = x.shape[1]
    multiplier = weight.shape[0] // cin
    channels = []
    for oc in range(weight.shape[0]):
        input_channel = oc // multiplier
        channels.append(F.conv2d(
            x[:, input_channel:input_channel + 1],
            weight[oc:oc + 1],
            bias[oc:oc + 1],
            stride=stride,
            padding=padding,
            groups=1,
        ))
    return torch.cat(channels, dim=1)


def _check_case(shape, cout, stride):
    torch.manual_seed(sum(shape) + cout + stride)
    x = torch.randn(shape, dtype=torch.float32)
    weight = torch.randn((cout, 1, 3, 3), dtype=torch.float32)
    bias = torch.randn((cout,), dtype=torch.float32)
    cin = shape[1]
    descriptor = convolution.classify_convolution(
        in_channels=cin, out_channels=cout, weight_in_channels=1,
        groups=cin, kernel=(3, 3), stride=(stride, stride),
        padding=(1, 1), dilation=(1, 1),
    )
    assert descriptor.path == "depthwise"
    grouped_reference = F.conv2d(
        x, weight, bias, stride=stride, padding=1, groups=cin
    )
    specialized_reference = _explicit_depthwise_reference(
        x, weight, bias, stride, 1
    )
    torch.testing.assert_close(specialized_reference, grouped_reference)

    params = _params(x, weight, bias, stride, 1)
    generic = convolution._conv_shader_source(params).decode("ascii")
    specialized = convolution._depthwise_shader_source(params).decode("ascii")
    tiled_specialized = convolution._conv_tile_shader_source(
        params, 0, 0, descriptor
    ).decode("ascii")
    assert "for (int ic" in generic
    assert "for (int ic" not in specialized
    assert "for (int ic" not in tiled_specialized
    assert f"input_channel = oc / {descriptor.depthwise_multiplier}" in specialized
    assert "read_weight(oc, ky, kx)" in specialized
    assert "input_channel = oc /" in tiled_specialized
    return specialized_reference


def _run_live_gpu_comparison() -> None:
    """Compare both shader families on one live OpenGL context.

    Invoke this diagnostic with ``--gpu`` on a host with the MatrixMan
    OpenGL backend available.  Normal test execution remains hardware-free.
    """
    from drivers import matrixman

    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = False
    matrixman.init()
    try:
        torch.manual_seed(991)
        x_cpu = torch.randn((1, 8, 17, 19), dtype=torch.float32)
        weight = torch.randn((16, 1, 3, 3), dtype=torch.float32)
        bias = torch.randn((16,), dtype=torch.float32)
        x = matrixman.to_device(x_cpu)
        args = (x, weight, bias, (2, 2), (1, 1), (1, 1), False, (0, 0), 8)
        from drivers.matrixman.backends.opengl import convolution

        specialized = convolution._execute_diagnostic_path(args, "depthwise").cpu()
        generic = convolution._execute_diagnostic_path(args, "grouped").cpu()
        expected = F.conv2d(x_cpu, weight, bias, stride=2, padding=1, groups=8)
        torch.testing.assert_close(specialized, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(generic, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(specialized, generic, rtol=1e-4, atol=1e-5)
        print("  live generic-vs-depthwise GPU comparison: PASS")
    finally:
        matrixman.shutdown()


def main() -> int:
    _check_case((1, 8, 17, 19), 8, 1)
    _check_case((1, 8, 17, 19), 16, 2)
    _check_case((1, 24, 15, 13), 48, 1)

    # Model-style shapes exercise the descriptor and packed-atlas geometry;
    # numerical GPU execution is covered by the same runtime path on a live
    # OpenGL diagnostic host.
    for shape, cout, stride in (
        ((1, 8, 320, 320), 16, 2),
        ((1, 24, 80, 80), 48, 2),
    ):
        descriptor = convolution.classify_convolution(
            in_channels=shape[1], out_channels=cout, weight_in_channels=1,
            groups=shape[1], kernel=(3, 3), stride=(stride, stride),
            padding=(1, 1), dilation=(1, 1),
        )
        assert descriptor.path == "depthwise"
        assert descriptor.depthwise_multiplier == cout // shape[1]

    # The cache identity must keep the specialized and generic variants apart.
    depthwise = convolution.classify_convolution(
        in_channels=8, out_channels=16, weight_in_channels=1, groups=8,
        kernel=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1),
    )
    grouped = convolution.classify_convolution(
        in_channels=8, out_channels=16, weight_in_channels=2, groups=4,
        kernel=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1),
    )
    assert convolution._conv_cache_key((1, 2), depthwise, "conv2d") != convolution._conv_cache_key(
        (1, 2), grouped, "conv2d"
    )
    if "--gpu" in sys.argv:
        _run_live_gpu_comparison()
    print("OpenGL depthwise Conv tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
