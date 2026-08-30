"""Small, backend-independent helpers shared by the YOLO demo and benchmark."""

from __future__ import annotations

import time

import cv2
import torch


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


def detections(prediction, width: int, height: int, names: dict,
               conf_threshold: float, iou_threshold: float, *, verbose: bool = False):
    """Decode [1, 4+classes, anchors], convert boxes, and run CPU NMS."""
    started = time.perf_counter()
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.ndim != 2 or prediction.shape[0] < 6:
        return [], 0
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.transpose(0, 1)
    boxes = prediction[:, :4]
    scores = prediction[:, 4:]
    confidence, classes = scores.max(dim=1).values, scores.argmax(dim=1)
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
