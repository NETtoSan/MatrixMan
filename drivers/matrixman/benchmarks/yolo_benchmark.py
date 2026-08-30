#!/usr/bin/env python3
"""Backend-neutral YOLO performance runner.

The runner owns warm-up epochs, measured-frame timing, profiling collection,
and JSON output. Backend-specific controls remain environment variables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.yolo_helpers import detections, first_tensor, preprocess_frame, reduced_detections
from drivers import matrixman
from drivers.matrixman.backend import get_backend


def _parse_env(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env requires NAME=VALUE, got {value!r}")
        name, setting = value.split("=", 1)
        if not name:
            raise ValueError("--env variable name must not be empty")
        result[name] = setting
    return result


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parents[3] / "demo"
    parser = argparse.ArgumentParser(description="MatrixMan backend-neutral YOLO benchmark")
    parser.add_argument("--backend", default="opengl", help="backend name reserved in the result schema; OpenGL is currently implemented")
    parser.add_argument("--model", type=Path, default=base / "models/VisDrone-arm64-480/weights/best.pt")
    parser.add_argument("--video", type=Path, default=base / "videos/video0.mp4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--variant", default="default", help="label this run for baseline/optimization comparisons")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="backend control; may be repeated")
    parser.add_argument("--json", type=Path, default=Path("textlogs/yolo_benchmark.json"))
    return parser.parse_args()


def _profile_snapshot() -> dict:
    try:
        from drivers.matrixman.backends.opengl import profiling
        profiling.collect_gpu_timing()
        return {
            "cpu_operations": {name: dict(record) for name, record in profiling.ops.items()},
            "counters": dict(profiling.counters),
            "conv": dict(profiling.conv),
            "gpu_timing": {name: dict(record) for name, record in profiling.gpu_timings.items()},
            "gpu_timer": {
                "capability": profiling._gpu_timer_capable,
                "api": profiling._gpu_timer_api,
                "reason": profiling._gpu_timer_reason,
                "query_count": len(profiling._gpu_timer_all),
                "unresolved_queries": len(profiling._gpu_timer_pending),
                "dropped_queries": profiling._gpu_timer_dropped,
            },
            "gpu_samples": list(profiling.gpu_timing_samples),
        }
    except (ImportError, AttributeError):
        return {"backend_profile": "unavailable"}


def main() -> int:
    args = parse_args()
    if args.backend.lower() != "opengl":
        raise RuntimeError("OpenCL benchmark logging is reserved in the schema but OpenCL is not implemented")
    if args.warmup < 0 or args.frames < 0:
        raise ValueError("--warmup and --frames must be non-negative")
    if args.frames == 0:
        raise ValueError("--frames must be positive for a measured benchmark")
    settings = _parse_env(args.env)
    for name, value in settings.items():
        os.environ[name] = value
    from ultralytics import YOLO

    model_path = args.model.expanduser().resolve()
    video_path = args.video.expanduser().resolve()
    if not model_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(model_path if not model_path.is_file() else video_path)
    matrixman.init()
    try:
        info = get_backend().device_info()
        yolo = YOLO(str(model_path))
        net = yolo.model.eval()
        names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video: {video_path}")
        matrixman.profile_reset()
        warmup_done = 0
        measured = []
        try:
            while warmup_done < args.warmup or len(measured) < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                is_warmup = warmup_done < args.warmup
                frame_started = time.perf_counter()
                cpu_input = preprocess_frame(frame, args.imgsz)
                upload_started = time.perf_counter()
                gpu_input = matrixman.to_gm45(cpu_input)
                upload_time = time.perf_counter() - upload_started
                inference_started = time.perf_counter()
                with torch.no_grad():
                    output = first_tensor(net(gpu_input))
                inference_time = time.perf_counter() - inference_started
                if not matrixman.is_gm45_tensor(output):
                    raise RuntimeError("model output did not remain a MatrixMan tensor")
                readback_started = time.perf_counter()
                reduction_enabled = os.environ.get("MATRIXMAN_GPU_POSTPROCESS", "").strip().lower() not in {"", "0", "false", "no", "off"}
                if reduction_enabled:
                    reduced_output = matrixman.gpu_postprocess_detection(output)
                    prediction = reduced_output.cpu()
                    readback_bytes = reduced_output._owner.layout.texture_width * reduced_output._owner.layout.texture_height * 16
                else:
                    prediction = output.cpu()
                    readback_bytes = output._owner.layout.texture_width * output._owner.layout.texture_height * 16
                readback_time = time.perf_counter() - readback_started
                postprocess_started = time.perf_counter()
                decode = reduced_detections if reduction_enabled else detections
                result, candidates = decode(prediction, args.imgsz, args.imgsz, names, args.conf, args.iou)
                postprocess_time = time.perf_counter() - postprocess_started
                if is_warmup:
                    warmup_done += 1
                    if warmup_done == args.warmup:
                        matrixman.profile_reset()
                    continue
                measured.append({
                    "index": len(measured),
                    "inference_seconds": inference_time,
                    "upload_seconds": upload_time,
                    "readback_seconds": readback_time,
                    "postprocess_seconds": postprocess_time,
                    "total_seconds": time.perf_counter() - frame_started,
                    "readback_bytes": readback_bytes,
                    "candidates_before_nms": candidates,
                    "detections": len(result),
                })
        finally:
            cap.release()
        profile = _profile_snapshot()
        matrixman.profile_report()
        totals = [frame["total_seconds"] for frame in measured]
        result = {
            "schema": "matrixman.yolo-benchmark.v1",
            "backend": {"requested": args.backend, **info},
            "variant": args.variant,
            "configuration": {
                "model": str(model_path), "video": str(video_path), "imgsz": args.imgsz,
                "conf": args.conf, "iou": args.iou, "warmup_frames": args.warmup,
                "measured_frames_requested": args.frames, "environment": settings,
            },
            "warmup_frames_completed": warmup_done,
            "measured_frames": measured,
            "summary": {
                "measured_frames": len(measured),
                "mean_total_seconds": sum(totals) / len(totals) if totals else None,
                "mean_fps": len(totals) / sum(totals) if totals and sum(totals) else None,
            },
            "profile": profile,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"benchmark JSON: {args.json}")
        return 0 if len(measured) == args.frames else 2
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
