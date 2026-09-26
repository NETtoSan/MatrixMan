"""Focused test for the public physical OpenGL texture inspection API."""

from __future__ import annotations

import numpy as np
import torch

from drivers import matrixman
from drivers.matrixman.tensor import readback_tensor


def main() -> int:
    matrixman.config.backend = "opengl"
    matrixman.config.preparedExecution = True
    values = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=True).eval()
    with torch.no_grad():
        expected = conv(values)
    matrixman.init()
    try:
        source = matrixman.to_device(values)
        result = conv(source)
        raw_source = matrixman.raw_texture(source)
        raw_result = matrixman.raw_texture(result)
        if raw_source.values.dtype != np.float32:
            raise AssertionError("raw texture dtype was not float32")
        if tuple(raw_source.values.shape) != (raw_source.height, raw_source.width, 4):
            raise AssertionError("raw source shape did not preserve physical RGBA geometry")
        flat = raw_source.values.reshape(-1)
        if not np.allclose(flat[: values.numel()], values.numpy().reshape(-1), rtol=0.0, atol=0.0):
            raise AssertionError("raw source did not preserve packed ordering")
        if tuple(raw_result.values.shape) != (raw_result.height, raw_result.width, 4):
            raise AssertionError("raw result shape did not preserve physical geometry")
        if raw_result.logical_shape != tuple(result.shape):
            raise AssertionError("raw result logical metadata was incorrect")
        actual = readback_tensor(result, audit_reason="raw texture regression")
        if not torch.allclose(actual, expected, rtol=2e-4, atol=2e-5):
            raise AssertionError("pending convolution was not materialized before raw readback")
    finally:
        matrixman.shutdown()
    try:
        matrixman.raw_texture(torch.zeros(1))
    except TypeError as error:
        if "MatrixManTensor" not in str(error):
            raise AssertionError(f"non-MatrixMan error was not informative: {error}")
    else:
        raise AssertionError("non-MatrixMan raw texture input was accepted")
    print("OpenGL raw texture test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
