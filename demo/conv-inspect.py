"""Minimal CPU-versus-MatrixMan Conv2D visual inspection demo."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drivers import matrixman

RTOL = 1e-4
ATOL = 2e-6


def main() -> None:
    torch.manual_seed(0)

    x_cpu = torch.randn(1, 3, 16, 16)
    conv = torch.nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    )
    conv.eval()

    with torch.no_grad():
        y_cpu = conv(x_cpu)

    try:
        matrixman.prepare(conv, backend="opengl")
        x_mm = matrixman.to_device(x_cpu)
        with torch.no_grad():
            y_mm = conv(x_mm)
        raw = matrixman.raw_texture(y_mm)

        print(f"raw texture id: {raw.texture_id}")
        print(f"raw texture width: {raw.width}")
        print(f"raw texture height: {raw.height}")
        print(f"raw.values.shape: {raw.values.shape}")
        print(f"raw.values.dtype: {raw.values.dtype}")
        print(f"raw storage offset: {raw.storage_offset}")

        # This is the explicit MatrixMan -> CPU validation boundary.
        y_matrixman = y_mm.cpu()
        error = (y_cpu - y_matrixman).abs()

        print(f"input shape: {list(x_cpu.shape)}")
        print(f"output shape: {list(y_matrixman.shape)}")
        print(f"max abs error: {error.max().item():.8g}")
        print(f"mean abs error: {error.mean().item():.8g}")
        passed = torch.allclose(y_cpu, y_matrixman, rtol=RTOL, atol=ATOL)
        print(f"rtol: {RTOL}")
        print(f"atol: {ATOL}")
        print(f"torch.allclose: {passed}")

        images = (
            x_cpu[0, 0],
            y_cpu[0, 0],
            y_matrixman[0, 0],
            error[0, 0],
        )
        titles = (
            "Input channel 0",
            "CPU Conv output channel 0",
            "MatrixMan Conv output channel 0",
            "Absolute error",
        )
        rgba = raw.values
        rgb = rgba[..., :3].copy()
        for lane_index in range(3):
            lane = rgb[..., lane_index]
            min_value = lane.min()
            max_value = lane.max()
            if max_value > min_value:
                rgb[..., lane_index] = (lane - min_value) / (max_value - min_value)
            else:
                rgb[..., lane_index] = 0.0

        figure, axes = plt.subplots(1, 5, figsize=(20, 4))
        for axis, image, title in zip(axes[:4], images, titles):
            plotted = axis.imshow(image, cmap="viridis")
            axis.set_title(title)
            axis.axis("off")
            figure.colorbar(plotted, ax=axis, fraction=0.046, pad=0.04)
        axes[4].imshow(rgb)
        axes[4].set_title("Raw OpenGL texture RGB")
        axes[4].axis("off")
        plt.tight_layout()
        plt.show()
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    main()
