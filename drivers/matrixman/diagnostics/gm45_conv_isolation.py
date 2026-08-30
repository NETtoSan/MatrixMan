#!/usr/bin/env python3
"""Diagnose the exact large Detect-head Conv2D divergence without changing GM45."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from drivers import matrixman as gm45
from drivers.matrixman.backends.opengl import storage


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "demo/models/VisDrone-arm64-480/weights/best.pt"
VIDEO = ROOT / "demo/videos/video0.mp4"


def stats(cpu, gpu):
    d = (gpu - cpu).to(torch.float64)
    return float(d.abs().max()), float(d.abs().mean()), float(torch.sqrt((d * d).mean()))


def report(name, cpu, gpu):
    a, b, c = stats(cpu, gpu)
    print(f"{name}: shape={list(cpu.shape)} max={a:.6g} mean={b:.6g} rmse={c:.6g} allclose={torch.allclose(cpu, gpu, atol=1e-5, rtol=1e-5)}")


def exact_case():
    yolo = YOLO(str(MODEL))
    model = yolo.model.eval()
    detect = model.model[15]
    activation = {}

    def capture(_module, _inputs, output):
        activation["value"] = output.detach().clone()

    hook = detect.cv2[0][0].register_forward_hook(capture)
    cap = cv2.VideoCapture(str(VIDEO))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read the diagnostic frame")
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cpu_input = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0).unsqueeze(0)
    with torch.no_grad():
        model(cpu_input)
    hook.remove()

    exact_input = activation["value"]
    weight = detect.cv2[0][1].conv.weight.detach().clone()
    print("Exact Detect.cv2[0][1].conv isolation")
    print(f"  input shape={list(exact_input.shape)} strides={list(exact_input.stride())}")
    print(f"  weight shape={list(weight.shape)} strides={list(weight.stride())} bias=None")
    print("  convolution: stride=(1,1) padding=(1,1) dilation=(1,1) groups=1")

    upload_start = time.perf_counter()
    gpu_input = gm45.to_gm45(exact_input)
    print(f"  input upload: {time.perf_counter() - upload_start:.3f}s texture=#{gpu_input._owner.texture}")
    with torch.no_grad():
        cpu_out = F.conv2d(exact_input, weight, None, stride=1, padding=1)
        gpu_out = F.conv2d(gpu_input, weight, None, stride=1, padding=1)
    readback_start = time.perf_counter()
    gpu_cpu = gpu_out.cpu()
    print(f"  output readback: {time.perf_counter() - readback_start:.3f}s texture=#{gpu_out._owner.texture}")
    report("  overall", cpu_out, gpu_cpu)
    print(f"  layouts: input tex={gpu_input._owner.layout.texture_width}x{gpu_input._owner.layout.texture_height}; output tex={gpu_out._owner.layout.texture_width}x{gpu_out._owner.layout.texture_height}")
    print(f"  logical/packed: input={exact_input.numel()}/{exact_input.numel() // 4} texels; weight={weight.numel()}/{(weight.numel() + 3) // 4} texels; output={cpu_out.numel()}/{cpu_out.numel() // 4} texels")
    print(f"  computed atlas: input={storage.packed_atlas_size(exact_input.numel())}; weight={storage.packed_atlas_size(weight.numel())}; output={storage.packed_atlas_size(cpu_out.numel())}")

    errors = (gpu_cpu - cpu_out).abs()
    channel_rows = []
    for channel in range(cpu_out.shape[1]):
        e = errors[0, channel]
        channel_rows.append((float(e.max()), float(e.mean()), float(torch.sqrt((e * e).mean())), channel))
    print("  10 best channels (max, mean, rmse, channel):")
    print("   ", sorted(channel_rows)[:10])
    print("  10 worst channels (max, mean, rmse, channel):")
    print("   ", sorted(channel_rows, reverse=True)[:10])

    regions = {
        "center": errors[..., 1:-1, 1:-1],
        "top border": errors[..., 0, 1:-1],
        "bottom border": errors[..., -1, 1:-1],
        "left border": errors[..., 1:-1, 0],
        "right border": errors[..., 1:-1, -1],
        "top-left corner": errors[..., 0, 0],
        "top-right corner": errors[..., 0, -1],
        "bottom-left corner": errors[..., -1, 0],
        "bottom-right corner": errors[..., -1, -1],
    }
    print("  spatial regions (max, mean, rmse):")
    for name, value in regions.items():
        print(f"    {name}: {float(value.max()):.6g}, {float(value.mean()):.6g}, {float(torch.sqrt((value * value).mean())):.6g}")

    print("  4x4 spatial tile mean absolute errors:")
    tile_h, tile_w = cpu_out.shape[-2] // 4, cpu_out.shape[-1] // 4
    for row in range(4):
        values = []
        for col in range(4):
            tile = errors[..., row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w]
            values.append(float(tile.mean()))
        print("   ", " ".join(f"{v:.5g}" for v in values))

    print("  output channel groups of 4 mean absolute error:")
    for start in range(0, 64, 4):
        print(f"    channels {start:02d}:{start + 4:02d}: {float(errors[:, start:start + 4].mean()):.6g}")
    return exact_input, weight


def synthetic_case(in_channels, out_channels, height, width):
    torch.manual_seed(1000 + in_channels + out_channels + height)
    cpu_input = torch.randn((1, in_channels, height, width), dtype=torch.float32)
    weight = torch.randn((out_channels, in_channels, 3, 3), dtype=torch.float32)
    gpu_input = gm45.to_gm45(cpu_input)
    cpu_out = F.conv2d(cpu_input, weight, None, stride=1, padding=1)
    gpu_out = F.conv2d(gpu_input, weight, None, stride=1, padding=1).cpu()
    a, b, c = stats(cpu_out, gpu_out)
    print(f"synthetic input=[1,{in_channels},{height},{width}] weight=[{out_channels},{in_channels},3,3] output={list(cpu_out.shape)} max={a:.6g} mean={b:.6g} rmse={c:.6g} allclose={torch.allclose(cpu_out, gpu_out, atol=1e-5, rtol=1e-5)}")


def main():
    gm45.set_trace(False)
    exact_case()
    print("Synthetic regressions")
    for in_channels, out_channels in ((8, 8), (16, 16), (32, 32), (64, 64)):
        synthetic_case(in_channels, out_channels, 32, 32)
    for size in (64, 128, 160):
        synthetic_case(64, 64, size, size)
    gm45.shutdown()


if __name__ == "__main__":
    main()
