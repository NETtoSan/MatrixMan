#!/usr/bin/env python3
"""VisDrone video demo using the MatrixMan GM45 PyTorch backend.

The model forward is intentionally called directly with a Gm45Tensor.  Only
the final model output is read back to CPU; OpenCV and postprocessing stay on
the CPU as they did in the original tracking demo.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivers import matrixman


def _stage(label: str, started: float, detail: str = "") -> float:
    now = time.perf_counter()
    suffix = f"; {detail}" if detail else ""
    print(f"  [{now - started:9.3f}s] {label}{suffix}")
    return now


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


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(_tensor_leaves(item))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for item in value.values():
            leaves.extend(_tensor_leaves(item))
        return leaves
    return []


def _activation_stats(value: torch.Tensor) -> dict:
    value = value.detach().to(torch.float32)
    return {
        "shape": list(value.shape),
        "min": float(value.min()) if value.numel() else 0.0,
        "max": float(value.max()) if value.numel() else 0.0,
        "mean": float(value.mean()) if value.numel() else 0.0,
        "mean_abs": float(value.abs().mean()) if value.numel() else 0.0,
        "zeros": int((value == 0).sum()),
        "numel": value.numel(),
        "nan": int(torch.isnan(value).sum()),
        "inf": int(torch.isinf(value).sum()),
    }


def _diagnostic_checkpoints(model, selected: set[int], store: dict, readback: bool) -> list:
    hooks = []
    for index, module in enumerate(model.model):
        if index not in selected:
            continue

        def hook(_module, _inputs, output, index=index):
            leaves = _tensor_leaves(output)
            records = []
            for leaf_index, leaf in enumerate(leaves):
                if readback:
                    if not matrixman.is_gm45_tensor(leaf):
                        raise RuntimeError(f"diagnostic GPU checkpoint {index} was not a Gm45Tensor")
                    leaf = leaf.cpu()
                records.append((_activation_stats(leaf), leaf.detach().clone()))
            store[index] = records

        hooks.append(module.register_forward_hook(hook))
    return hooks


def _run_divergence_diagnostic(cpu_net, gpu_net, cpu_input, gpu_input, selected, names, started):
    cpu_records = {}
    gpu_records = {}
    cpu_hooks = _diagnostic_checkpoints(cpu_net, selected, cpu_records, readback=False)
    gpu_hooks = _diagnostic_checkpoints(gpu_net, selected, gpu_records, readback=True)
    try:
        with torch.no_grad():
            cpu_net(cpu_input)
            gpu_net(gpu_input)
    finally:
        for hook in cpu_hooks + gpu_hooks:
            hook.remove()

    print("\nMatrixMan activation divergence diagnostic")
    print(f"  checkpoints: {sorted(selected)}")
    first_bad = None
    previous_good = None
    for index in sorted(selected):
        module = gpu_net.model[index]
        cpu_leaves = cpu_records.get(index, [])
        gpu_leaves = gpu_records.get(index, [])
        print(f"  module {index} ({type(module).__name__})")
        if len(cpu_leaves) != len(gpu_leaves):
            print(f"    tensor-count mismatch: CPU={len(cpu_leaves)} GM45={len(gpu_leaves)}")
            first_bad = index if first_bad is None else first_bad
            continue
        module_bad = False
        for leaf_index, ((cpu, cpu_value), (gpu, gpu_value)) in enumerate(zip(cpu_leaves, gpu_leaves)):
            count = max(cpu["numel"], 1)
            difference = (gpu_value - cpu_value).to(torch.float64)
            max_error = float(difference.abs().max()) if difference.numel() else 0.0
            mean_error = float(difference.abs().mean()) if difference.numel() else 0.0
            rmse = float(torch.sqrt((difference * difference).mean())) if difference.numel() else 0.0
            zero_fraction_cpu = cpu["zeros"] / count
            zero_fraction_gpu = gpu["zeros"] / count
            if cpu["shape"] != gpu["shape"] or max_error > 1e-3 or mean_error > 1e-4 or abs(zero_fraction_cpu - zero_fraction_gpu) > 0.01:
                module_bad = True
            print(
                f"    tensor {leaf_index}: shape CPU={cpu['shape']} GM45={gpu['shape']} "
                f"error max/mean/rmse=({max_error:.6g},{mean_error:.6g},{rmse:.6g}) "
                f"min/max/mean CPU=({cpu['min']:.6g},{cpu['max']:.6g},{cpu['mean']:.6g}) "
                f"GM45=({gpu['min']:.6g},{gpu['max']:.6g},{gpu['mean']:.6g}) "
                f"mean_abs CPU/GM45=({cpu['mean_abs']:.6g},{gpu['mean_abs']:.6g}) "
                f"zeros CPU/GM45=({zero_fraction_cpu:.2%},{zero_fraction_gpu:.2%}) "
                f"nan/inf CPU=({cpu['nan']},{cpu['inf']}) GM45=({gpu['nan']},{gpu['inf']})"
            )
        if module_bad and first_bad is None:
            first_bad = index
        if not module_bad:
            previous_good = index
    _stage("divergence diagnostic complete", started,
           f"first_bad={first_bad}; last_checked_good={previous_good}")
    return first_bad


def _capture_diagnostic(store: dict, key: str, value, readback: bool) -> None:
    leaves = _tensor_leaves(value)
    records = []
    for leaf in leaves:
        if readback:
            if not matrixman.is_gm45_tensor(leaf):
                raise RuntimeError(f"Detect diagnostic checkpoint {key} was not a Gm45Tensor")
            leaf = leaf.cpu()
        else:
            leaf = leaf.detach().clone()
        records.append((_activation_stats(leaf), leaf.detach().clone()))
    store[key] = records


def _install_detect_diagnostic(detect, store: dict, readback: bool):
    hooks = []
    for branch_name in ("cv2", "cv3"):
        branches = getattr(detect, branch_name)
        for scale, branch in enumerate(branches):
            for name, module in branch.named_modules():
                if not name:
                    continue
                key = f"{branch_name}[{scale}].{name}"

                def hook(_module, _inputs, output, key=key):
                    _capture_diagnostic(store, key, output, readback)

                hooks.append(module.register_forward_hook(hook))

    original_forward_head = detect.forward_head

    def forward_head(self, x, box_head=None, cls_head=None):
        if box_head is None or cls_head is None:
            return original_forward_head(x, box_head, cls_head)
        bs = x[0].shape[0]
        box_parts = []
        score_parts = []
        for scale in range(self.nl):
            _capture_diagnostic(store, f"input[{scale}]", x[scale], readback)
            box_raw = box_head[scale](x[scale])
            _capture_diagnostic(store, f"box_branch[{scale}] output", box_raw, readback)
            box_view = box_raw.view(bs, 4 * self.reg_max, -1)
            _capture_diagnostic(store, f"box_branch[{scale}] view", box_view, readback)
            box_parts.append(box_view)
            score_raw = cls_head[scale](x[scale])
            _capture_diagnostic(store, f"class_branch[{scale}] output", score_raw, readback)
            score_view = score_raw.view(bs, self.nc, -1)
            _capture_diagnostic(store, f"class_branch[{scale}] view", score_view, readback)
            score_parts.append(score_view)
        boxes = torch.cat(box_parts, dim=-1)
        scores = torch.cat(score_parts, dim=-1)
        _capture_diagnostic(store, "concatenated raw boxes", boxes, readback)
        _capture_diagnostic(store, "concatenated raw scores", scores, readback)
        return dict(boxes=boxes, scores=scores, feats=x)

    detect.forward_head = types.MethodType(forward_head, detect)
    original_inference = detect._inference

    def inference(self, x):
        dbox = self._get_decode_boxes(x)
        _capture_diagnostic(store, "decoded boxes", dbox, readback)
        scores = x["scores"].sigmoid()
        _capture_diagnostic(store, "sigmoid class scores", scores, readback)
        result = torch.cat((dbox, scores), 1)
        _capture_diagnostic(store, "final detection output", result, readback)
        return result

    detect._inference = types.MethodType(inference, detect)

    def dfl_hook(_module, _inputs, output):
        _capture_diagnostic(store, "DFL output", output, readback)

    hooks.append(detect.dfl.register_forward_hook(dfl_hook))
    return hooks, original_forward_head, original_inference


def _run_detect_diagnostic(cpu_net, gpu_net, cpu_input, gpu_input, started):
    cpu_store = {}
    gpu_store = {}
    cpu_detect = cpu_net.model[15]
    gpu_detect = gpu_net.model[15]
    cpu_hooks, cpu_forward_head, cpu_inference = _install_detect_diagnostic(cpu_detect, cpu_store, False)
    gpu_hooks, gpu_forward_head, gpu_inference = _install_detect_diagnostic(gpu_detect, gpu_store, True)
    try:
        with torch.no_grad():
            cpu_net(cpu_input)
            gpu_net(gpu_input)
    finally:
        for hook in cpu_hooks + gpu_hooks:
            hook.remove()
        cpu_detect.forward_head = cpu_forward_head
        cpu_detect._inference = cpu_inference
        gpu_detect.forward_head = gpu_forward_head
        gpu_detect._inference = gpu_inference

    print("\nDetect internal CPU vs MatrixMan diagnostic")
    first_bad = None
    previous_good = None
    for key, cpu_records in cpu_store.items():
        gpu_records = gpu_store.get(key)
        if gpu_records is None or len(cpu_records) != len(gpu_records):
            print(f"  {key}: checkpoint mismatch CPU={len(cpu_records)} GM45={len(gpu_records or [])}")
            first_bad = key
            break
        key_bad = False
        for leaf_index, ((cpu, cpu_value), (gpu, gpu_value)) in enumerate(zip(cpu_records, gpu_records)):
            difference = (gpu_value - cpu_value).to(torch.float64)
            max_error = float(difference.abs().max()) if difference.numel() else 0.0
            mean_error = float(difference.abs().mean()) if difference.numel() else 0.0
            rmse = float(torch.sqrt((difference * difference).mean())) if difference.numel() else 0.0
            count = max(cpu["numel"], 1)
            cpu_zero = cpu["zeros"] / count
            gpu_zero = gpu["zeros"] / count
            if cpu["shape"] != gpu["shape"] or max_error > 1e-3 or mean_error > 1e-4 or abs(cpu_zero - gpu_zero) > 0.01:
                key_bad = True
            print(
                f"  {key} tensor={leaf_index} shape CPU={cpu['shape']} GM45={gpu['shape']} "
                f"error max/mean/rmse=({max_error:.6g},{mean_error:.6g},{rmse:.6g}) "
                f"CPU min/max/mean=({cpu['min']:.6g},{cpu['max']:.6g},{cpu['mean']:.6g}) "
                f"GM45 min/max/mean=({gpu['min']:.6g},{gpu['max']:.6g},{gpu['mean']:.6g}) "
                f"zeros CPU/GM45=({cpu_zero:.2%},{gpu_zero:.2%})"
            )
        if key_bad and first_bad is None:
            first_bad = key
        if not key_bad:
            previous_good = key
    _stage("Detect diagnostic complete", started, f"first_bad={first_bad}; last_checked_good={previous_good}")
    return first_bad


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    keep = []
    order = scores.argsort(descending=True)
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes[rest, 0], boxes[current, 0])
        yy1 = torch.maximum(boxes[rest, 1], boxes[current, 1])
        xx2 = torch.minimum(boxes[rest, 2], boxes[current, 2])
        yy2 = torch.minimum(boxes[rest, 3], boxes[current, 3])
        intersection = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        area_current = (boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = intersection / (area_current + area_rest - intersection).clamp_min(1e-12)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


def _detections(prediction, width: int, height: int, names: dict, conf_threshold: float, iou_threshold: float):
    """Decode the common Ultralytics [1, 4+classes, anchors] output on CPU."""
    started = time.perf_counter()
    print(f"  [{started:.6f}] postprocess start; confidence threshold={conf_threshold}; NMS IoU threshold={iou_threshold}")
    raw_shape = tuple(prediction.shape)
    if prediction.ndim == 3:
        prediction = prediction[0]
    _stage("reshape", started, f"{list(raw_shape)} -> {list(prediction.shape)}")
    if prediction.ndim != 2 or prediction.shape[0] < 6:
        return [], 0
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.transpose(0, 1)
    _stage("reshape/transpose", started, f"logical candidates={int(prediction.shape[0])}")
    raw_candidates = int(prediction.shape[0])
    boxes = prediction[:, :4]
    scores = prediction[:, 4:]
    confidence = scores.max(dim=1).values
    _stage("confidence extraction", started, f"{raw_candidates} candidates")
    classes = scores.argmax(dim=1)
    _stage("class selection", started, f"{scores.shape[1]} classes")
    keep = confidence >= conf_threshold
    kept = int(keep.sum().item())
    _stage("confidence threshold filtering", started,
           f"raw={raw_candidates}; remaining={kept}; threshold={conf_threshold}")
    if not keep.any():
        _stage("NMS", started, "skipped: no candidates after threshold")
        _stage("tracking", started, "skipped: no candidates")
        return [], 0
    boxes = boxes[keep]
    confidence = confidence[keep]
    classes = classes[keep]
    cx, cy, w, h = boxes.unbind(1)
    x1 = (cx - w / 2).clamp(0, width - 1)
    y1 = (cy - h / 2).clamp(0, height - 1)
    x2 = (cx + w / 2).clamp(0, width - 1)
    y2 = (cy + h / 2).clamp(0, height - 1)
    _stage("box conversion", started, f"{kept} boxes")
    nms_indices = []
    for cls in classes.unique(sorted=True).tolist():
        class_indices = torch.nonzero(classes == cls, as_tuple=False).flatten()
        nms_indices.append(class_indices[_nms(boxes[class_indices], confidence[class_indices], iou_threshold)])
    selected = torch.cat(nms_indices) if nms_indices else torch.empty(0, dtype=torch.long)
    selected = selected[confidence[selected].argsort(descending=True)]
    _stage("NMS", started, f"input={kept}; output={len(selected)}; IoU threshold={iou_threshold}")
    _stage("tracking", started, "not configured; CPU tracking stage skipped")
    result = []
    loop_started = time.perf_counter()
    for box, score, cls in zip(torch.stack((x1, y1, x2, y2), 1)[selected], confidence[selected], classes[selected]):
        result.append((box.tolist(), float(score), int(cls), names.get(int(cls), str(int(cls)))))
    _stage("candidate packaging loop", started,
           f"iterated over {len(result)} candidates in {time.perf_counter() - loop_started:.3f}s")
    return result, kept


def _validation_metrics(cpu_output: torch.Tensor, gm45_output: torch.Tensor, names: dict, started: float) -> None:
    compare_started = time.perf_counter()
    if tuple(cpu_output.shape) != tuple(gm45_output.shape):
        raise RuntimeError(f"CPU/GM45 output shape mismatch: {list(cpu_output.shape)} vs {list(gm45_output.shape)}")
    if cpu_output.ndim != 3 or tuple(cpu_output.shape[:2]) != (1, 14) or cpu_output.shape[-1] <= 0:
        raise RuntimeError(f"unexpected validation output shape: {list(cpu_output.shape)}")

    def stats(cpu_part: torch.Tensor, gpu_part: torch.Tensor) -> tuple[float, float, float]:
        diff = (gpu_part - cpu_part).to(torch.float64)
        absolute = diff.abs()
        return (float(absolute.max()), float(absolute.mean()), float(torch.sqrt((diff * diff).mean())))

    overall = stats(cpu_output, gm45_output)
    boxes = stats(cpu_output[:, 0:4], gm45_output[:, 0:4])
    scores = stats(cpu_output[:, 4:14], gm45_output[:, 4:14])
    meaningful = cpu_output.abs() > 1e-8
    relative = ((gm45_output - cpu_output).abs() / cpu_output.abs().clamp_min(1e-8))[meaningful]
    cpu_scores = cpu_output[0, 4:14]
    gm45_scores = gm45_output[0, 4:14]

    print("\nCPU vs MatrixMan raw-output validation")
    print(f"  CPU output shape: {list(cpu_output.shape)}")
    print(f"  GM45 output shape: {list(gm45_output.shape)}")
    print(f"  overall max_abs_error={overall[0]:.6g} mean_abs_error={overall[1]:.6g} rmse={overall[2]:.6g}")
    print(f"  maximum relative error (|CPU|>1e-8): {float(relative.max()):.6g}")
    print(f"  NaN values: CPU={int(torch.isnan(cpu_output).sum())} GM45={int(torch.isnan(gm45_output).sum())}")
    print(f"  Inf values: CPU={int(torch.isinf(cpu_output).sum())} GM45={int(torch.isinf(gm45_output).sum())}")
    print(f"  BOX CHANNELS [0:4]: max_abs_error={boxes[0]:.6g} mean_abs_error={boxes[1]:.6g} rmse={boxes[2]:.6g}")
    print(f"  CLASS SCORES [4:14]: max_abs_error={scores[0]:.6g} mean_abs_error={scores[1]:.6g} rmse={scores[2]:.6g}")
    print(f"    CPU max={float(cpu_scores.max()):.6g} GM45 max={float(gm45_scores.max()):.6g}")
    print(f"    CPU mean={float(cpu_scores.mean()):.6g} GM45 mean={float(gm45_scores.mean()):.6g}")

    cpu_anchor_scores, cpu_classes = cpu_scores.max(dim=0)
    gm45_anchor_scores, gm45_classes = gm45_scores.max(dim=0)
    for threshold in (0.01, 0.05, 0.10, 0.25, 0.50):
        print(f"  threshold {threshold:.2f}: anchors above CPU={int((cpu_anchor_scores > threshold).sum())} GM45={int((gm45_anchor_scores > threshold).sum())}")

    def print_top(label: str, output: torch.Tensor, anchor_scores: torch.Tensor, classes: torch.Tensor) -> None:
        print(f"  top 10 {label} predictions:")
        values, indices = torch.topk(anchor_scores, k=10)
        for value, anchor in zip(values.tolist(), indices.tolist()):
            cls = int(classes[anchor])
            box = output[0, 0:4, anchor].tolist()
            print(f"    anchor={anchor} class={cls} ({names.get(cls, str(cls))}) confidence={value:.6g} box={box}")

    print_top("CPU", cpu_output, cpu_anchor_scores, cpu_classes)
    print_top("GM45", gm45_output, gm45_anchor_scores, gm45_classes)
    _stage("numerical comparison complete", started,
           f"duration={time.perf_counter() - compare_started:.3f}s")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MatrixMan GM45 VisDrone tracking demo")
    parser.add_argument("--model", type=Path, default=base / "models/VisDrone-arm64-480/weights/best.pt")
    parser.add_argument("--video", type=Path, default=base / "videos/video0.mp4")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--frames", type=int, default=0, help="stop after N frames; 0 means until quit/end")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--validate-cpu", action="store_true", help="compare one frame's raw CPU and GM45 outputs")
    parser.add_argument("--diagnose-divergence", action="store_true", help="compare selected CPU/GM45 module activations for one frame")
    parser.add_argument("--diagnostic-checkpoints", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
                        help="top-level YOLO module indices to inspect")
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
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.imgsz > 640:
        raise ValueError("--imgsz must be <= 640")
    if args.imgsz % 32:
        raise ValueError(f"--imgsz must be a positive multiple of 32; got {args.imgsz}")

    print("MatrixMan VisDrone demo")
    print(f"  model: {model_path}")
    print(f"  video: {video_path}")
    print(f"  inference resolution: {args.imgsz}x{args.imgsz}")
    print("  backend: MatrixMan / Intel GM45 / OpenGL 2.1 / GLSL 1.20")

    matrixman.set_trace(matrixman.debug_enabled())
    matrixman.init()
    matrixman.profile_reset()
    yolo = YOLO(str(model_path))
    net = yolo.model.eval()
    cpu_net = YOLO(str(model_path)).model.eval() if (args.validate_cpu or args.diagnose_divergence) else None
    names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))
    print(f"  loaded PyTorch model: {type(net).__name__}, classes={len(names)}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    frame_count = 0
    started = time.perf_counter()
    window_name = "MatrixMan VisDrone"
    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while args.frames == 0 or frame_count < args.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_started = time.perf_counter()
            preprocess_started = time.perf_counter()
            display_frame = cv2.resize(frame, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            cpu_input = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0).unsqueeze(0)
            preprocess_time = time.perf_counter() - preprocess_started
            print(f"frame {frame_count + 1}: CPU tensor {list(cpu_input.shape)}")
            cpu_output = None
            if args.diagnose_divergence:
                upload_started = time.perf_counter()
                gpu_input = matrixman.to_gm45(cpu_input)
                _stage("CPU->GM45 upload complete", started, f"duration={time.perf_counter() - upload_started:.3f}s")
                _run_detect_diagnostic(cpu_net, net, cpu_input, gpu_input, started)
                frame_count += 1
                break
            cpu_forward_started = time.perf_counter()
            if args.validate_cpu:
                with torch.no_grad():
                    cpu_raw = cpu_net(cpu_input)
                cpu_output = _first_tensor(cpu_raw)
                if not isinstance(cpu_output, torch.Tensor) or matrixman.is_gm45_tensor(cpu_output):
                    raise RuntimeError("CPU validation forward did not produce a normal CPU tensor")
                _stage("CPU forward complete", started,
                       f"duration={time.perf_counter() - cpu_forward_started:.3f}s shape={list(cpu_output.shape)}")
            upload_started = time.perf_counter()
            gpu_input = matrixman.to_gm45(cpu_input)
            upload_time = time.perf_counter() - upload_started
            _stage("CPU->GM45 upload complete", started, f"duration={time.perf_counter() - upload_started:.3f}s")
            print(f"  MatrixMan input: {gpu_input}")
            with torch.no_grad():
                gpu_forward_started = time.perf_counter()
                raw = net(gpu_input)
            gpu_forward_time = time.perf_counter() - gpu_forward_started
            _stage("GM45 forward complete", started, f"duration={time.perf_counter() - gpu_forward_started:.3f}s")
            gpu_output = _first_tensor(raw)
            if not matrixman.is_gm45_tensor(gpu_output):
                raise RuntimeError("model output did not remain a Gm45Tensor")
            print(f"  GM45 forward complete; explicit readback: output texture #{gpu_output._owner.texture}")
            readback_started = time.perf_counter()
            prediction = gpu_output.cpu()
            readback_time = time.perf_counter() - readback_started
            _stage("readback complete", started, f"shape={list(prediction.shape)}; duration={time.perf_counter() - readback_started:.3f}s")
            if args.validate_cpu:
                _validation_metrics(cpu_output, prediction, names, started)
                frame_count += 1
                break
            postprocess_started = time.perf_counter()
            detections, candidates_before_nms = _detections(prediction, args.imgsz, args.imgsz, names, args.conf, args.iou)
            postprocess_time = time.perf_counter() - postprocess_started
            _stage("postprocessing complete", started, f"duration={time.perf_counter() - postprocess_started:.3f}s")
            drawing_started = time.perf_counter()
            for (x1, y1, x2, y2), score, cls, label in detections:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label} {score:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            _stage("drawing complete", started, f"{len(detections)} detections; duration={time.perf_counter() - drawing_started:.3f}s")
            elapsed = time.perf_counter() - frame_started
            overlay_scale = max(0.3, min(0.6, args.imgsz / 640.0 * 0.6))
            cv2.putText(display_frame, f"GM45 {args.imgsz} | det {len(detections)} | FPS {1 / max(elapsed, 1e-9):.2f}",
                        (6, max(16, int(20 * overlay_scale / 0.6))), cv2.FONT_HERSHEY_SIMPLEX,
                        overlay_scale, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"fwd {gpu_forward_time:.1f}s  rb {readback_time:.1f}s  total {elapsed:.1f}s",
                        (6, max(30, int(42 * overlay_scale / 0.6))), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.28, overlay_scale - 0.08), (0, 255, 255), 1, cv2.LINE_AA)
            print(f"frame {frame_count}: input=[1,3,{args.imgsz},{args.imgsz}] output={list(prediction.shape)} anchors={prediction.shape[-1]} candidates_before_nms={candidates_before_nms} detections={len(detections)} preprocess={preprocess_time:.3f}s upload={upload_time:.3f}s GM45 forward={gpu_forward_time:.3f}s readback={readback_time:.3f}s postprocess={postprocess_time:.3f}s drawing={time.perf_counter() - drawing_started:.3f}s total={elapsed:.3f}s FPS={1 / max(elapsed, 1e-9):.2f}")
            print(f"  CPU postprocessing/drawing complete; detections={len(detections)}")
            frame_count += 1
            if not args.no_display:
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        matrixman.profile_report()
        matrixman.shutdown()
    print(f"completed frames: {frame_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
