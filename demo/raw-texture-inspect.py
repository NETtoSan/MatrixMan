import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import torch

from drivers import matrixman


def _print_metadata(name, raw):
    print(f"{name} logical shape: {list(raw.logical_shape)}")
    print(f"{name} texture id: {raw.texture_id}")
    print(f"{name} texture size: {raw.width}x{raw.height}")
    print(f"{name} storage offset: {raw.storage_offset}")
    print(f"{name} raw.values.shape: {raw.values.shape}")
    print(f"{name} dtype: {raw.values.dtype}")


def main() -> None:
    torch.manual_seed(0)
    x_cpu = torch.randn(1, 3, 16, 16)
    conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1, bias=True)
    conv.eval()
    matrixman.config.backend = "opengl"
    matrixman.prepare(conv, backend="opengl")
    x_mm = matrixman.to_device(x_cpu)
    try:
        with torch.no_grad():
            y_mm = conv(x_mm)

        raw_in = matrixman.raw_texture(x_mm)
        raw_out = matrixman.raw_texture(y_mm)
        _print_metadata("input", raw_in)
        _print_metadata("output", raw_out)

        cpu_flat = x_cpu.contiguous().reshape(-1).numpy()
        packed_flat = raw_in.values.reshape(-1)
        input_error = abs(packed_flat[:cpu_flat.size] - cpu_flat)
        print(f"raw input packing allclose: {bool(torch.allclose(torch.from_numpy(cpu_flat), torch.from_numpy(packed_flat[:cpu_flat.size]), rtol=1e-6, atol=1e-6))}")
        print(f"raw input packing max abs error: {float(input_error.max())}")

        figure, axes = plt.subplots(2, 4, figsize=(12, 6))
        for lane, axis in enumerate(axes[0]):
            image = axis.imshow(raw_in.values[..., lane])
            axis.set_title(f"Input {'RGBA'[lane]}")
            figure.colorbar(image, ax=axis)
        for lane, axis in enumerate(axes[1]):
            image = axis.imshow(raw_out.values[..., lane])
            axis.set_title(f"Output {'RGBA'[lane]}")
            figure.colorbar(image, ax=axis)
        plt.tight_layout()
        plt.show()
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    main()
