"""Smoke test for the single persistent Conv diagnostics window."""

from __future__ import annotations

import numpy as np
import torch

from drivers import matrixman


def main() -> int:
    torch.manual_seed(11)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.SiLU(inplace=True),
        torch.nn.Conv2d(4, 2, 1),
    ).eval()
    value = torch.randn(1, 3, 8, 8)
    matrixman.config.backend = "opengl"
    matrixman.config.profile = True
    matrixman.config.profileDetail = False

    # The disabled path must not create the window or perform the diagnostic
    # raw-texture readback.  The ordinary final .cpu() readback is unrelated.
    matrixman.config.convDiag = False
    matrixman.init()
    try:
        with torch.no_grad():
            model(matrixman.to_device(value)).cpu()
        from drivers.matrixman.backends.opengl import conv_diagnostics

        disabled_snapshot = conv_diagnostics.snapshot()
        if disabled_snapshot["window_created"]:
            raise AssertionError(f"convDiag=False created a window: {disabled_snapshot}")
        if disabled_snapshot["raw_readback_count"] != 0:
            raise AssertionError(f"convDiag=False performed a raw readback: {disabled_snapshot}")
    finally:
        matrixman.shutdown()

    matrixman.config.convDiag = True
    matrixman.init()
    try:
        with torch.no_grad():
            result = model(matrixman.to_device(value)).cpu()
        from drivers.matrixman.backends.opengl import conv_diagnostics
        from drivers.matrixman.backends.opengl import resources

        before_shutdown = conv_diagnostics.snapshot()
        latest = conv_diagnostics._latest
        if before_shutdown["window_create_count"] != 1:
            raise AssertionError(f"expected one diagnostics window, got {before_shutdown}")
        if before_shutdown["capture_count"] < 2:
            raise AssertionError(f"expected at least two Conv captures, got {before_shutdown}")
        if before_shutdown["raw_readback_count"] != before_shutdown["capture_count"]:
            raise AssertionError(f"expected one raw readback per capture, got {before_shutdown}")
        if tuple(result.shape) != (1, 2, 8, 8):
            raise AssertionError(f"unexpected output shape: {tuple(result.shape)}")
        if latest is None:
            raise AssertionError("diagnostics did not retain the latest Conv capture")

        raw = latest["raw_texture_values"]
        logical = np.asarray(latest["output_values"], dtype=np.float32)
        expected_raw, expected_layout = resources.pack_linear_rgba(logical)
        expected_shape = (expected_layout.texture_height, expected_layout.texture_width, 4)
        if tuple(raw.shape) != expected_shape:
            raise AssertionError(
                f"raw panel shape {raw.shape} did not match physical texture {expected_shape}"
            )
        if tuple(raw.shape) == tuple(logical.shape):
            raise AssertionError("raw panel unexpectedly has the logical NCHW shape")
        if not np.allclose(raw, expected_raw, rtol=0.0, atol=5e-4):
            raise AssertionError("raw RGBA texel/lane ordering changed during diagnostics")
        raw_preview = conv_diagnostics._raw_texture_image(raw)
        if tuple(raw_preview.shape[:2]) != tuple(raw.shape[:2]):
            raise AssertionError("raw visualization changed physical texel geometry")
    finally:
        matrixman.shutdown()

    after_shutdown = conv_diagnostics.snapshot()
    if after_shutdown["window_created"]:
        raise AssertionError("diagnostics window remained after shutdown")
    print("OpenGL Conv diagnostics window test: PASS")
    print(
        f"  windows_created={before_shutdown['window_create_count']} "
        f"captures={before_shutdown['capture_count']} "
        f"raw_readbacks={before_shutdown['raw_readback_count']} "
        f"raw_shape={before_shutdown['raw_texture_shape']} "
        f"cleaned_up={not after_shutdown['window_created']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
