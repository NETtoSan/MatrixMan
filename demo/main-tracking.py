#!/usr/bin/env python3
"""Simple public-facing MatrixMan YOLO demonstration."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.yolo_helpers import detections, first_tensor, preprocess_frame
from drivers import matrixman
from drivers.matrixman.backend import get_backend


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MatrixMan VisDrone tracking demo")
    parser.add_argument("--model", type=Path, default=base / "models/VisDrone-arm64-480/weights/best.pt")
    parser.add_argument("--video", type=Path, default=base / "videos/video0.mp4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 means until quit/end")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model_path = args.model.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if args.imgsz <= 0 or args.imgsz > 640 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32 and no greater than 640")

    matrixman.init()
    try:
        info = get_backend().device_info()
        print("MatrixMan VisDrone demo")
        print(f"backend: MatrixMan / {info['backend']} / {info.get('renderer', 'unknown')}")
        yolo = YOLO(str(model_path))
        net = yolo.model.eval()
        names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        frame_count = 0
        try:
            if not args.no_display:
                cv2.namedWindow("MatrixMan VisDrone", cv2.WINDOW_AUTOSIZE)
            while args.frames == 0 or frame_count < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                display_frame = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
                cpu_input = preprocess_frame(frame, args.imgsz)
                gpu_input = matrixman.to_gm45(cpu_input)
                with torch.no_grad():
                    prediction = first_tensor(net(gpu_input))
                if not matrixman.is_gm45_tensor(prediction):
                    raise RuntimeError("model output did not remain a Gm45Tensor")
                prediction = prediction.cpu()
                result, _ = detections(prediction, args.imgsz, args.imgsz, names, args.conf, args.iou)
                for (x1, y1, x2, y2), score, _cls, label in result:
                    box = tuple(int(v) for v in (x1, y1, x2, y2))
                    cv2.rectangle(display_frame, box[:2], box[2:], (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{label} {score:.2f}", (box[0], max(12, box[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                elapsed = time.perf_counter() - frame_started
                print(f"frame {frame_count}: detections={len(result)} total={elapsed:.3f}s FPS={1 / max(elapsed, 1e-9):.2f}")
                if not args.no_display:
                    cv2.imshow("MatrixMan VisDrone", display_frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                frame_count += 1
        finally:
            cap.release()
            if not args.no_display:
                cv2.destroyAllWindows()
        print(f"completed frames: {frame_count}")
        return 0
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
