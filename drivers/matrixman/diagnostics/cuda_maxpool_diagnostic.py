"""Focused CUDA max_pool2d_with_indices checks for the YOLO SPPF subset."""

from __future__ import annotations

import torch

from drivers import matrixman


def check(name, values, expected):
    error = (values - expected).abs()
    result = {
        "case": name,
        "shape": tuple(values.shape),
        "max_abs_error": float(error.max()) if error.numel() else 0.0,
        "mean_abs_error": float(error.mean()) if error.numel() else 0.0,
        "allclose": bool(torch.allclose(values, expected, atol=0.0, rtol=0.0)),
    }
    print(result)
    assert result["allclose"]


def run():
    matrixman.prefer("cuda")
    matrixman.init()
    try:
        cases = [
            ("random_128x20x20", torch.randn(1, 128, 20, 20)),
            ("random_small", torch.randn(1, 3, 7, 9)),
            ("all_negative", -torch.rand(1, 4, 6, 8)),
            ("padding_edge", torch.tensor([[[[-9.0, -8.0], [-7.0, -6.0]]]])),
        ]
        for name, source in cases:
            gpu_input = matrixman.to_device(source)
            values, indices = torch.nn.functional.max_pool2d(
                gpu_input, kernel_size=5, stride=1, padding=2,
                dilation=1, ceil_mode=False, return_indices=True,
            )
            check(name, values.cpu(), torch.nn.functional.max_pool2d(
                source, kernel_size=5, stride=1, padding=2,
                dilation=1, ceil_mode=False,
            ))
            assert tuple(indices.shape) == (0,)

        source = torch.randn(1, 2, 6, 6)
        base = matrixman.to_device(torch.cat((torch.zeros(1), source.reshape(-1))))
        offset_input = type(base)._from_owner(
            base._owner, source.shape, storage_offset=1,
            logical_strides=(72, 36, 6, 1),
        )
        values, _ = torch.nn.functional.max_pool2d(
            offset_input, kernel_size=5, stride=1, padding=2,
            dilation=1, ceil_mode=False, return_indices=True,
        )
        check("nonzero_storage_offset", values.cpu(), torch.nn.functional.max_pool2d(
            source, kernel_size=5, stride=1, padding=2,
            dilation=1, ceil_mode=False,
        ))
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    run()
