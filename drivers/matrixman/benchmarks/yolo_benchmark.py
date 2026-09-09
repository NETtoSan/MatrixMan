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
from drivers.matrixman.benchmarks.cpu_audit import frame_stages, memory_snapshot, stage


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
    parser.add_argument("--cpu-audit", action="store_true", help="record coarse process/thread CPU time, readbacks, threads, and RSS")
    return parser.parse_args()


def _persistent_parameter_resource_count() -> int | None:
    try:
        from drivers.matrixman.backends.opengl.resources import persistent_parameter_resources
        return int(persistent_parameter_resources())
    except (ImportError, AttributeError, RuntimeError):
        return None


def _activation_pool_snapshot() -> dict | None:
    try:
        from drivers.matrixman.backends.opengl.resources import activation_pool_stats
        return activation_pool_stats()
    except (ImportError, AttributeError, RuntimeError):
        return None


def _profile_snapshot() -> dict:
    try:
        from drivers.matrixman.backends.opengl import profiling
        profiling.collect_gpu_timing()
        persistent_resources = _persistent_parameter_resource_count()
        return {
            "elapsed_seconds": time.perf_counter() - profiling.started,
            "thread_cpu_clock": "time.thread_time" if profiling.thread_cpu_supported() else "time.process_time fallback",
            "detailed_cpu_stage_profiling": profiling.detailed_enabled(),
            "cpu_operations": {name: dict(record) for name, record in profiling.ops.items()},
            "cpu_stage_timings": {
                name: dict(record) for name, record in profiling.stage_timings.items()
            },
            "gl_call_timings": {
                name: dict(record) for name, record in profiling.gl_call_timings.items()
            },
            "conv_stage_timings": {
                name: dict(record) for name, record in profiling.conv_stage_timings.items()
            },
            "state_cache_skips": profiling.diagnostic_snapshot().get("state_cache_skips", {}),
            "counters": dict(profiling.counters),
            "parameter_groups": profiling.parameter_group_snapshot(),
            "persistent_parameter_resources": persistent_resources,
            "peak_persistent_parameter_resources": int(
                profiling.counters["persistent_parameter_resources_peak"]
            ) if persistent_resources is not None else None,
            "activation_pool": _activation_pool_snapshot(),
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


_RESIDENCY_COUNTERS = (
    "parameter_uploads", "parameter_upload_bytes", "repeated_parameter_uploads",
    "parameter_cache_hits", "parameter_cache_misses", "parameter_cache_invalidations",
    "texture_allocations", "scratch_texture_allocations", "scratch_texture_reuses",
    "scratch_texture_releases", "scratch_texture_evictions",
    "activation_pool_allocations", "activation_pool_reuses",
    "activation_pool_releases", "activation_pool_evictions",
    "readback_calls", "readback_bytes",
)


def _residency_snapshot() -> dict[str, int]:
    try:
        from drivers.matrixman.backends.opengl import profiling
        return {name: int(profiling.counters[name]) for name in _RESIDENCY_COUNTERS}
    except (ImportError, AttributeError):
        return {}


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {name: int(after.get(name, 0) - before.get(name, 0)) for name in _RESIDENCY_COUNTERS}


def _diagnostic_snapshot() -> dict:
    try:
        from drivers.matrixman.backends.opengl import profiling
        return profiling.diagnostic_snapshot()
    except (ImportError, AttributeError):
        return {"stages": {}, "gl_calls": {}, "state_cache_skips": {}}


def _diagnostic_delta(before: dict, after: dict) -> dict:
    result = {}
    for section in ("stages", "gl_calls"):
        result[section] = {}
        for name, values in after.get(section, {}).items():
            previous = before.get(section, {}).get(name, {})
            result[section][name] = {
                field: values[field] - previous.get(field, 0)
                for field in values
                if isinstance(values[field], (int, float))
            }
    result["state_cache_skips"] = {
        name: int(value) - int(before.get("state_cache_skips", {}).get(name, 0))
        for name, value in after.get("state_cache_skips", {}).items()
    }
    return result


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
    matrixman.config.reloadFromEnvironment()
    if args.cpu_audit:
        matrixman.config.auditCpuLeaks = True
        matrixman.config.profile = True
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
        benchmark_process_cpu_started = time.process_time()
        benchmark_thread_cpu_started = getattr(time, "thread_time", time.process_time)()
        audit_start = memory_snapshot() if args.cpu_audit else None
        memory_samples = []
        warmup_done = 0
        stream_frame = 0
        measured = []
        warmup_residency = {}
        warmup_parameter_groups = []
        warmup_diagnostics = {"stages": {}, "gl_calls": {}}
        try:
            while warmup_done < args.warmup or len(measured) < args.frames:
                ok, frame = cap.read()
                if not ok:
                    break
                is_warmup = warmup_done < args.warmup
                frame_started = time.perf_counter()
                frame_process_cpu_started = time.process_time()
                frame_thread_cpu_started = getattr(time, "thread_time", time.process_time)()
                residency_before = _residency_snapshot()
                from drivers.matrixman import audit
                upload_started = time.perf_counter()
                diagnostic_before = _diagnostic_snapshot()
                with audit.frame(stream_frame), frame_stages(
                    args.cpu_audit or bool(matrixman.config.profileDetail)
                ) as stages:
                    with stage("preprocessing"):
                        cpu_input = preprocess_frame(frame, args.imgsz)
                    with stage("host_to_gpu_upload"):
                        gpu_input = matrixman.to_device(cpu_input)
                    upload_time = (
                        stages.values["host_to_gpu_upload"]["wall_seconds"]
                        if stages is not None else time.perf_counter() - upload_started
                    )
                    inference_started = time.perf_counter()
                    with stage("matrixman_dispatch"):
                        with torch.no_grad():
                            output = first_tensor(net(gpu_input))
                    inference_time = time.perf_counter() - inference_started
                    if not matrixman.is_matrixman_tensor(output):
                        raise RuntimeError("model output did not remain a MatrixMan tensor")
                    reduction_enabled = matrixman.config.gpuPostprocess
                    readback_started = time.perf_counter()
                    with stage("explicit_readback"):
                        with audit.readback_context("final_output"):
                            if reduction_enabled:
                                reduced_output = matrixman.gpu_postprocess_detection(output)
                                prediction = reduced_output.cpu()
                                readback_bytes = reduced_output._owner.layout.texture_width * reduced_output._owner.layout.texture_height * 16
                            else:
                                prediction = output.cpu()
                                readback_bytes = output._owner.layout.texture_width * output._owner.layout.texture_height * 16
                    readback_time = (
                        stages.values["explicit_readback"]["wall_seconds"]
                        if stages is not None else time.perf_counter() - readback_started
                    )
                    postprocess_started = time.perf_counter()
                    with stage("cpu_postprocess"):
                        decode = reduced_detections if reduction_enabled else detections
                        result, candidates = decode(prediction, args.imgsz, args.imgsz, names, args.conf, args.iou)
                    postprocess_time = (
                        stages.values["cpu_postprocess"]["wall_seconds"]
                        if stages is not None else time.perf_counter() - postprocess_started
                    )
                if is_warmup:
                    warmup_done += 1
                    if warmup_done == args.warmup:
                        warmup_residency = _counter_delta({}, _residency_snapshot())
                        from drivers.matrixman.backends.opengl import profiling
                        warmup_parameter_groups = profiling.parameter_group_snapshot()
                        warmup_diagnostics = _diagnostic_snapshot()
                        matrixman.profile_reset()
                        if args.cpu_audit:
                            audit.reset_readbacks()
                    stream_frame += 1
                    continue
                measured.append({
                    "index": len(measured),
                    "inference_seconds": inference_time,
                    "upload_seconds": upload_time,
                    "readback_seconds": readback_time,
                    "postprocess_seconds": postprocess_time,
                    "total_seconds": time.perf_counter() - frame_started,
                    "process_cpu_seconds": time.process_time() - frame_process_cpu_started,
                    "thread_cpu_seconds": getattr(time, "thread_time", time.process_time)() - frame_thread_cpu_started,
                    "readback_bytes": readback_bytes,
                    "candidates_before_nms": candidates,
                    "detections": len(result),
                    "residency": _counter_delta(residency_before, _residency_snapshot()),
                    "stages": ({name: dict(value) for name, value in stages.values.items()}
                               if stages is not None else {}),
                    "diagnostics": _diagnostic_delta(
                        diagnostic_before, _diagnostic_snapshot()
                    ),
                })
                if args.cpu_audit:
                    memory_samples.append({"frame": len(measured) - 1, **memory_snapshot()})
                stream_frame += 1
        finally:
            cap.release()
        profile = _profile_snapshot()
        audit_end = memory_snapshot() if args.cpu_audit else None
        from drivers.matrixman import audit
        readbacks = audit.readback_report() if args.cpu_audit else None
        matrixman.profile_report(frame_count=len(measured))
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
                "mean_stage_seconds": {
                    name: {
                        metric: sum(frame["stages"].get(name, {}).get(metric, 0.0) for frame in measured) / len(measured)
                        for metric in ("wall_seconds", "thread_cpu_seconds")
                    }
                    for name in sorted({name for frame in measured for name in frame["stages"]})
                },
                "mean_cpu_stage_timings": {
                    section: {
                        name: {
                            field: sum(
                                frame.get("diagnostics", {}).get(section, {}).get(name, {}).get(field, 0.0)
                                for frame in measured
                            ) / len(measured)
                            for field in ("calls", "wall_seconds", "thread_cpu_seconds", "redundant_calls")
                            if any(field in frame.get("diagnostics", {}).get(section, {}).get(name, {}) for frame in measured)
                        }
                        for name in sorted({
                            name
                            for frame in measured
                            for name in frame.get("diagnostics", {}).get(section, {})
                        })
                    }
                    for section in ("stages", "gl_calls")
                },
                "warmup_cpu_stage_timings": warmup_diagnostics["stages"],
                "warmup_gl_call_timings": warmup_diagnostics["gl_calls"],
                "mean_process_cpu_seconds": sum(
                    frame["process_cpu_seconds"] for frame in measured
                ) / len(measured) if measured else None,
                "mean_thread_cpu_seconds": sum(
                    frame["thread_cpu_seconds"] for frame in measured
                ) / len(measured) if measured else None,
                "residency": {
                    "warmup": warmup_residency,
                    "warmup_parameter_groups": warmup_parameter_groups,
                    "first_measured_frame": measured[0].get("residency", {}) if measured else {},
                    "measured_parameter_groups": profile.get("parameter_groups", []),
                    "steady_state_measured_frames": {
                        name: sum(frame.get("residency", {}).get(name, 0) for frame in measured[1:]) / max(1, len(measured) - 1)
                        for name in _RESIDENCY_COUNTERS
                    },
                },
            },
            "profile": profile,
        }
        if args.cpu_audit:
            result["cpu_audit"] = {
                "process_cpu_seconds": time.process_time() - benchmark_process_cpu_started,
                "thread_cpu_seconds": getattr(time, "thread_time", time.process_time)() - benchmark_thread_cpu_started,
                "logical_cpus": os.cpu_count(),
                "start_memory": audit_start,
                "end_memory": audit_end,
                "memory_samples": memory_samples,
                "readbacks": readbacks,
                "unsupported_operations": matrixman.unsupported_report(),
            }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"benchmark JSON: {args.json}")
        return 0 if len(measured) == args.frames else 2
    finally:
        matrixman.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
