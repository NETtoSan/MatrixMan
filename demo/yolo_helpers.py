"""Small, backend-independent helpers shared by the YOLO demo and benchmark."""

from __future__ import annotations

import os
import time

import cv2
import torch

from drivers.matrixman.benchmarks.cpu_audit import stage


_torchvision_nms = None
_torchvision_nms_checked = False


def _get_torchvision_nms():
    """Return native CPU NMS when available, probing it at most once."""
    global _torchvision_nms, _torchvision_nms_checked
    if _torchvision_nms_checked:
        return _torchvision_nms
    _torchvision_nms_checked = True
    if os.environ.get("MATRIXMAN_DISABLE_TORCHVISION_NMS", "").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        return None
    try:
        from torchvision.ops import nms
        nms(torch.empty((0, 4), dtype=torch.float32), torch.empty(0, dtype=torch.float32), 0.5)
    except (ImportError, OSError, RuntimeError):
        return None
    _torchvision_nms = nms
    return _torchvision_nms


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = first_tensor(item)
            if found is not None:
                return found
    return None


def preprocess_frame(frame, image_size: int) -> torch.Tensor:
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0).unsqueeze(0)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Run per-class NMS with a native CPU fast path and local fallback."""
    if boxes.device.type == "cpu" and boxes.dtype in (torch.float32, torch.float64):
        native_nms = _get_torchvision_nms()
        if native_nms is not None:
            try:
                return native_nms(boxes.contiguous(), scores.contiguous(), iou_threshold)
            except RuntimeError:
                # Disable only the optional operator after an operator/runtime
                # availability failure; retain the repository implementation.
                global _torchvision_nms
                _torchvision_nms = None
    keep = []
    order = scores.argsort(descending=True)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1) * (y2 - y1)
    while order.numel():
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(x1[rest], x1[current])
        yy1 = torch.maximum(y1[rest], y1[current])
        xx2 = torch.minimum(x2[rest], x2[current])
        yy2 = torch.minimum(y2[rest], y2[current])
        intersection = (xx2 - xx1).clamp_min(0) * (yy2 - yy1).clamp_min(0)
        iou = intersection / (areas[current] + areas[rest] - intersection).clamp_min(1e-12)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long)


def detections(prediction, width: int, height: int, names: dict,
               conf_threshold: float, iou_threshold: float, *, verbose: bool = False):
    """Decode [1, 4+classes, anchors], convert boxes, and run CPU NMS."""
    started = time.perf_counter()
    with stage("detection_decode_shape"):
        if prediction.ndim == 3:
            prediction = prediction[0]
        if prediction.ndim != 2 or prediction.shape[0] < 6:
            return [], 0
        if prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.transpose(0, 1)
    with stage("confidence_filter"):
        boxes = prediction[:, :4]
        scores = prediction[:, 4:]
        confidence, classes = scores.max(dim=1).values, scores.argmax(dim=1)
        keep = confidence >= conf_threshold
        kept = int(keep.sum().item())
    if not keep.any():
        return [], 0
    boxes, confidence, classes = boxes[keep], confidence[keep], classes[keep]
    with stage("box_conversion"):
        cx, cy, w, h = boxes.unbind(1)
        converted = torch.stack(((cx - w / 2).clamp(0, width - 1),
                                 (cy - h / 2).clamp(0, height - 1),
                                 (cx + w / 2).clamp(0, width - 1),
                                 (cy + h / 2).clamp(0, height - 1)), 1)
    with stage("nms"):
        selected_by_class = []
        for cls in classes.unique(sorted=True).tolist():
            indices = torch.nonzero(classes == cls, as_tuple=False).flatten()
            selected_by_class.append(indices[_nms(converted[indices], confidence[indices], iou_threshold)])
        selected = torch.cat(selected_by_class) if selected_by_class else torch.empty(0, dtype=torch.long)
        selected = selected[confidence[selected].argsort(descending=True)]
    with stage("result_packaging"):
        result = [
            (box.tolist(), float(score), int(cls), names.get(int(cls), str(int(cls))))
            for box, score, cls in zip(converted[selected], confidence[selected], classes[selected])
        ]
    if verbose:
        print(f"  postprocess: candidates={int(prediction.shape[0])} thresholded={kept} "
              f"detections={len(result)} duration={time.perf_counter() - started:.3f}s")
    return result, kept


def reduced_detections(prediction, width: int, height: int, names: dict,
                       conf_threshold: float, iou_threshold: float):
    """Decode Step 8A's GPU-reduced [1, 6, anchors] result."""
    if prediction.ndim != 3 or tuple(prediction.shape[:2]) != (1, 6):
        raise RuntimeError(f"unexpected reduced detection output shape: {list(prediction.shape)}")
    reduced = prediction[0].transpose(0, 1)
    boxes, confidence, classes = reduced[:, :4], reduced[:, 4], reduced[:, 5].to(torch.long)
    keep = confidence >= conf_threshold
    kept = int(keep.sum().item())
    if not keep.any():
        return [], 0
    boxes, confidence, classes = boxes[keep], confidence[keep], classes[keep]
    cx, cy, w, h = boxes.unbind(1)
    converted = torch.stack(((cx - w / 2).clamp(0, width - 1),
                             (cy - h / 2).clamp(0, height - 1),
                             (cx + w / 2).clamp(0, width - 1),
                             (cy + h / 2).clamp(0, height - 1)), 1)
    selected_by_class = []
    for cls in classes.unique(sorted=True).tolist():
        indices = torch.nonzero(classes == cls, as_tuple=False).flatten()
        selected_by_class.append(indices[_nms(converted[indices], confidence[indices], iou_threshold)])
    selected = torch.cat(selected_by_class) if selected_by_class else torch.empty(0, dtype=torch.long)
    selected = selected[confidence[selected].argsort(descending=True)]
    return [
        (box.tolist(), float(score), int(cls), names.get(int(cls), str(int(cls))))
        for box, score, cls in zip(converted[selected], confidence[selected], classes[selected])
    ], kept
