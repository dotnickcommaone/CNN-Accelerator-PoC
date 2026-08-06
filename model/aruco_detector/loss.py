from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


def build_grid_targets(
    targets: list[dict[str, Tensor]], grid_h: int, grid_w: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    object_target = torch.zeros((len(targets), grid_h, grid_w), device=device)
    box_target = torch.zeros((len(targets), 4, grid_h, grid_w), device=device)
    positive = torch.zeros_like(object_target, dtype=torch.bool)

    for batch_index, target in enumerate(targets):
        for cx, cy, width, height in target["boxes_yolo"].to(device):
            cell_x = min(int(cx * grid_w), grid_w - 1)
            cell_y = min(int(cy * grid_h), grid_h - 1)
            object_target[batch_index, cell_y, cell_x] = 1.0
            positive[batch_index, cell_y, cell_x] = True
            box_target[batch_index, :, cell_y, cell_x] = torch.stack(
                (
                    cx * grid_w - cell_x,
                    cy * grid_h - cell_y,
                    width,
                    height,
                )
            )
    return object_target, box_target, positive


class DetectionLoss(nn.Module):
    def __init__(self, positive_weight: float = 8.0, box_weight: float = 5.0) -> None:
        super().__init__()
        self.positive_weight = positive_weight
        self.box_weight = box_weight

    def forward(
        self, predictions: Tensor, targets: list[dict[str, Tensor]]
    ) -> tuple[Tensor, dict[str, float]]:
        _, _, grid_h, grid_w = predictions.shape
        object_target, box_target, positive = build_grid_targets(
            targets, grid_h, grid_w, predictions.device
        )
        weights = torch.ones_like(object_target)
        weights[positive] = self.positive_weight
        object_loss = functional.binary_cross_entropy_with_logits(
            predictions[:, 0], object_target, weight=weights
        )
        if positive.any():
            predicted_boxes = predictions[:, 1:5].sigmoid().permute(0, 2, 3, 1)
            target_boxes = box_target.permute(0, 2, 3, 1)
            box_loss = functional.smooth_l1_loss(
                predicted_boxes[positive], target_boxes[positive]
            )
        else:
            box_loss = predictions[:, 1:5].sum() * 0.0
        total = object_loss + self.box_weight * box_loss
        return total, {
            "loss": float(total.detach()),
            "object_loss": float(object_loss.detach()),
            "box_loss": float(box_loss.detach()),
        }
