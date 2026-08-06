from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.ops import nms


class ArucoMobileNetV2(nn.Module):
    """One-class grid detector with a MobileNetV2-0.35 backbone."""

    def __init__(self, width_mult: float = 0.35, pretrained: bool = False) -> None:
        super().__init__()
        if pretrained and width_mult != 1.0:
            raise ValueError(
                "torchvision does not provide pretrained MobileNetV2-0.35 "
                "weights; use pretrained=false or width_mult=1.0"
            )
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        base = mobilenet_v2(weights=weights, width_mult=width_mult)
        self.backbone = base.features
        channels = base.last_channel
        hidden = max(32, int(128 * width_mult))
        self.detect = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, 5, kernel_size=1),
        )
        self._initialize_head()

    def _initialize_head(self) -> None:
        for layer in self.detect.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Start with a low object probability to stabilize sparse detection.
        final = self.detect[-1]
        assert isinstance(final, nn.Conv2d)
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)
        final.bias.data[0] = -4.0

    def forward(self, images: Tensor) -> Tensor:
        return self.detect(self.backbone(images))


def decode_predictions(
    output: Tensor,
    score_threshold: float = 0.35,
    nms_iou_threshold: float = 0.45,
) -> list[dict[str, Tensor]]:
    """Decode BCHW predictions into normalized xyxy boxes and scores."""
    if output.ndim != 4 or output.shape[1] != 5:
        raise ValueError(f"Expected output [B,5,H,W], received {tuple(output.shape)}")

    batch, _, grid_h, grid_w = output.shape
    objectness = output[:, 0].sigmoid()
    geometry = output[:, 1:5].sigmoid()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=output.device),
        torch.arange(grid_w, device=output.device),
        indexing="ij",
    )

    cx = (grid_x[None] + geometry[:, 0]) / grid_w
    cy = (grid_y[None] + geometry[:, 1]) / grid_h
    width = geometry[:, 2]
    height = geometry[:, 3]
    boxes = torch.stack(
        (
            (cx - width / 2).clamp(0, 1),
            (cy - height / 2).clamp(0, 1),
            (cx + width / 2).clamp(0, 1),
            (cy + height / 2).clamp(0, 1),
        ),
        dim=-1,
    )

    decoded: list[dict[str, Tensor]] = []
    for index in range(batch):
        keep = objectness[index] >= score_threshold
        image_boxes = boxes[index][keep]
        image_scores = objectness[index][keep]
        if image_boxes.numel() == 0:
            decoded.append(
                {
                    "boxes": output.new_zeros((0, 4)),
                    "scores": output.new_zeros((0,)),
                }
            )
            continue
        selected = nms(image_boxes, image_scores, nms_iou_threshold)
        decoded.append(
            {"boxes": image_boxes[selected], "scores": image_scores[selected]}
        )
    return decoded
