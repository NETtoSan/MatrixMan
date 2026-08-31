#!/usr/bin/env python3
"""Minimal Ultralytics inference example using MatrixMan."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

# Make ``python3 demo/example/yolo_example.py ...`` work from any directory.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _preprocess(image, imgsz: int) -> torch.Tensor:
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).to(torch.float32) / 255.0
    return image.permute(2, 0, 1).contiguous().unsqueeze(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal MatrixMan YOLO example")
    parser.add_argument("model", type=Path, help="Ultralytics YOLO detection checkpoint")
    parser.add_argument("image", type=Path, help="input image path")
    parser.add_argument("--imgsz", type=int, default=320, help="square inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = parser.parse_args()

    if args.imgsz <= 0 or args.imgsz > 640 or args.imgsz % 32:
        parser.error("--imgsz must be a positive multiple of 32 no larger than 640")
    if not args.model.is_file():
        parser.error(f"could not find model checkpoint: {args.model}")
    from ultralytics import YOLO
    from drivers import matrixman

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"could not read image: {args.image}")

    print("Loading:", args.model)
    model = YOLO(str(args.model)).model.eval()

    input_cpu = _preprocess(image, args.imgsz)
    print("CPU input:", list(input_cpu.shape), input_cpu.dtype)

    # CPU preprocessing/upload ends here. The model forward uses MatrixMan.
    input_mm = matrixman.to_device(input_cpu)
    print("MatrixMan input:", type(input_mm).__name__, input_mm.device)
    with torch.no_grad():
        output_mm = model(input_mm)

    # This is the only explicit readback in the example.
    output_cpu = _first_tensor(output_mm)
    if output_cpu is None:
        raise RuntimeError(f"model returned no tensor: {type(output_mm).__name__}")
    output_cpu = output_cpu.cpu()
    if output_cpu.ndim != 3 or output_cpu.shape[0] != 1 or output_cpu.shape[1] < 5:
        raise RuntimeError(f"unexpected model output shape: {list(output_cpu.shape)}")

    scores = output_cpu[:, 4:, :]
    best_scores = scores.max(dim=1).values[0]
    detections = int((best_scores >= args.conf).sum())
    print("MatrixMan output shape:", list(output_cpu.shape))
    print(f"detections above {args.conf:.2f}: {detections}")
    print(f"highest confidence: {float(best_scores.max()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
