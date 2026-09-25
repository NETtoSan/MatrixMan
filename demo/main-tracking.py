#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
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


# ============================================================
# MATRIXMAN CONFIGURATION / HELP
# ============================================================
#
# The package is the source of truth for all MatrixMan settings. To inspect
# every public option from this demo, set MATRIXMAN_SHOW_CONFIG_HELP to True.
# Leave the example assignments below commented unless you intentionally want
# to override environment variables and MatrixMan defaults.
MATRIXMAN_SHOW_CONFIG_HELP = False

matrixman.trace = True


# matrixman.config.describe()
# matrixman.config.describe("activation_pool")
# matrixman.config.options("sync_policy")
#
# Example tuning:
# matrixman.config.sync_policy = "safe"
# matrixman.config.activation_pool = "deferred"
# matrixman.config.scratch_pool = "deferred"
# matrixman.config.tile_limit = 256
#
# Known useful GM45 combination (example only; not enabled):
# matrixman.config.sync_policy = "safe"
# matrixman.config.activation_pool = "deferred"
# matrixman.config.scratch_pool = "deferred"
# matrixman.config.tile_limit = 256
# matrixman.profile = "summary"
# matrixman.trace = False


def show_matrixman_configuration_help() -> None:
    """Print package-generated configuration help when explicitly requested."""
    if MATRIXMAN_SHOW_CONFIG_HELP:
        matrixman.config.describe()

def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MatrixMan VisDrone tracking demo")
    parser.add_argument("--model", type=Path, default=base / "models/VisDrone-matrixman-optimized/weights/best.pt")
    parser.add_argument("--video", type=Path, default=base / "videos/video0.mp4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 means until quit/end")
    parser.add_argument("--no-display", action="store_true")
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare", action="store_true", help="eagerly prepare the model before inference")
    preparation.add_argument("--no-auto-prepare", action="store_true", help="disable lazy model preparation")
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

    show_matrixman_configuration_help()
    matrixman.init()
    try:
        info = get_backend().device_info()
        print("MatrixMan VisDrone demo")
        device = info.get("renderer") or info.get("device") or info.get("name") or "unknown"
        print(f"backend: MatrixMan / {info['backend']} / {device}")
        yolo = YOLO(str(model_path))
        model = yolo.model.eval()
        if args.prepare:
            model = matrixman.prepare(model, backend=info["backend"].lower())
        elif not args.no_auto_prepare:
            # Normal inference: MatrixMan prepares supported patterns once.
            model = matrixman.auto_prepare(model)
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
                input_tensor = matrixman.to_device(preprocess_frame(frame, args.imgsz))

                event_timing = os.environ.get("MATRIXMAN_CUDA_EVENT_TIMING", "").lower() in {"1", "true", "yes", "on"}
                execution = get_backend().execution if event_timing else None
                event_started = execution.gpu_timing_start() if execution is not None else False

                with torch.no_grad():
                    prediction = first_tensor(model(input_tensor))
                event_stopped = execution.gpu_timing_stop() if event_started else False

                readback_started = time.perf_counter() if event_stopped else None
                # MatrixMan tensor -> CPU tensor: the single normal readback boundary.
                prediction = prediction.cpu()
                if event_stopped:
                    gpu_ms = execution.gpu_timing_elapsed_ms()
                    readback_ms = (time.perf_counter() - readback_started) * 1000.0
                    print(
                        f"  CUDA events: gpu_execution={gpu_ms:.3f} ms "
                        f"readback_boundary={readback_ms:.3f} ms"
                    )
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
        if matrixman.profiling:
            matrixman.profile_report(frame_count=frame_count)
        print(f"completed frames: {frame_count}")
        return 0
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # Ctrl+C is a normal way to stop the demo; avoid dumping the
        # implementation traceback while preserving main() cleanup.
        print("\nInterrupted by user.")
        raise SystemExit(130)
