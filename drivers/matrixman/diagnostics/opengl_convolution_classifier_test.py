"""Hardware-independent tests for OpenGL convolution-family classification."""

from __future__ import annotations

from drivers.matrixman.backends.opengl import convolution


def _classify(cin, cout, groups, *, weight_cin=None, kernel=(3, 3),
              stride=(1, 1), padding=(1, 1), dilation=(1, 1),
              transposed=False, output_padding=(0, 0)):
    return convolution.classify_convolution(
        in_channels=cin,
        out_channels=cout,
        weight_in_channels=cin // groups if weight_cin is None else weight_cin,
        groups=groups,
        kernel=kernel,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
    )


def _assert_rejected(**kwargs):
    try:
        _classify(**kwargs)
    except RuntimeError:
        return
    raise AssertionError(f"invalid convolution was accepted: {kwargs}")


def main() -> int:
    for cin, cout, groups, multiplier in (
        (8, 8, 8, 1),
        (8, 16, 8, 2),
        (24, 48, 24, 2),
    ):
        descriptor = _classify(cin, cout, groups)
        assert descriptor.path == "depthwise"
        assert descriptor.in_channels == cin
        assert descriptor.out_channels == cout
        assert descriptor.groups == groups
        assert descriptor.in_channels_per_group == 1
        assert descriptor.out_channels_per_group == multiplier
        assert descriptor.depthwise_multiplier == multiplier

    for cin, cout, groups in ((16, 24, 8), (24, 8, 8)):
        descriptor = _classify(cin, cout, groups)
        assert descriptor.path == "grouped"
        assert descriptor.in_channels_per_group == cin // groups
        assert descriptor.out_channels_per_group == cout // groups
        assert descriptor.depthwise_multiplier is None

    dense = _classify(16, 32, 1, kernel=(1, 1), padding=(0, 0))
    assert dense.path == "dense"
    assert dense.in_channels_per_group == 16
    assert dense.out_channels_per_group == 32
    assert convolution._convolution_trace_name(dense) == "Conv2D"

    _assert_rejected(cin=10, cout=16, groups=4)
    _assert_rejected(cin=16, cout=10, groups=4)
    _assert_rejected(cin=8, cout=10, groups=8)
    _assert_rejected(cin=8, cout=16, groups=8, weight_cin=2)
    _assert_rejected(cin=8, cout=16, groups=8, transposed=True)
    _assert_rejected(cin=8, cout=16, groups=8, dilation=(2, 2))
    assert convolution._convolution_trace_name(_classify(8, 16, 8)) == "DWConv"
    assert convolution._convolution_trace_name(_classify(16, 24, 8)) == "GroupedConv"

    dense_key = convolution._conv_cache_key((1, 2, 3), dense, "conv2d")
    depthwise = _classify(8, 16, 8)
    depthwise_key = convolution._conv_cache_key((1, 2, 3), depthwise, "conv2d")
    assert dense_key != depthwise_key

    print("OpenGL convolution classifier tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
