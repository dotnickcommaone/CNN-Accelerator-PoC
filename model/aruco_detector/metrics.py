from __future__ import annotations

import torch
from torch import Tensor
from torchvision.ops import box_iou


def yolo_to_xyxy(boxes: Tensor) -> Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    cx, cy, width, height = boxes.unbind(dim=1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=1,
    ).clamp(0, 1)


def detection_counts(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for prediction, target in zip(predictions, targets):
        predicted = prediction["boxes"].detach().cpu()
        actual = yolo_to_xyxy(target["boxes_yolo"].detach().cpu())
        if predicted.numel() == 0:
            false_negative += len(actual)
            continue
        if actual.numel() == 0:
            false_positive += len(predicted)
            continue
        ious = box_iou(predicted, actual)
        matched_actual: set[int] = set()
        for pred_index in prediction["scores"].detach().cpu().argsort(descending=True):
            best_iou, actual_index = ious[pred_index].max(dim=0)
            actual_id = int(actual_index)
            if float(best_iou) >= iou_threshold and actual_id not in matched_actual:
                true_positive += 1
                matched_actual.add(actual_id)
            else:
                false_positive += 1
        false_negative += len(actual) - len(matched_actual)
    return true_positive, false_positive, false_negative


def ranked_detections(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[float, bool]], int]:
    """Return score/TP pairs for dataset-level AP and the GT object count."""
    ranked: list[tuple[float, bool]] = []
    ground_truth_count = 0
    for prediction, target in zip(predictions, targets):
        predicted = prediction["boxes"].detach().cpu()
        scores = prediction["scores"].detach().cpu()
        actual = yolo_to_xyxy(target["boxes_yolo"].detach().cpu())
        ground_truth_count += len(actual)
        matched_actual: set[int] = set()
        order = scores.argsort(descending=True)
        ious = (
            box_iou(predicted, actual)
            if predicted.numel() and actual.numel()
            else torch.zeros((len(predicted), len(actual)))
        )
        for pred_index in order:
            is_true_positive = False
            if len(actual):
                best_iou, actual_index = ious[pred_index].max(dim=0)
                actual_id = int(actual_index)
                if float(best_iou) >= iou_threshold and actual_id not in matched_actual:
                    is_true_positive = True
                    matched_actual.add(actual_id)
            ranked.append((float(scores[pred_index]), is_true_positive))
    return ranked, ground_truth_count


def average_precision(
    detections: list[tuple[float, bool]], ground_truth_count: int
) -> float:
    """Compute all-point interpolated average precision."""
    if ground_truth_count == 0 or not detections:
        return 0.0
    detections = sorted(detections, key=lambda item: item[0], reverse=True)
    true_positive = torch.tensor(
        [1.0 if is_tp else 0.0 for _, is_tp in detections]
    ).cumsum(0)
    false_positive = torch.tensor(
        [0.0 if is_tp else 1.0 for _, is_tp in detections]
    ).cumsum(0)
    recall = true_positive / ground_truth_count
    precision = true_positive / (true_positive + false_positive).clamp_min(1e-9)
    recall = torch.cat((torch.tensor([0.0]), recall, torch.tensor([1.0])))
    precision = torch.cat((torch.tensor([1.0]), precision, torch.tensor([0.0])))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = torch.maximum(precision[index], precision[index + 1])
    changed = torch.where(recall[1:] != recall[:-1])[0]
    return float(
        torch.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1])
    )
