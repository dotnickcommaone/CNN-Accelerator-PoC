from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def read_yolo_labels(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 5), dtype=np.float32)
    labels = np.loadtxt(path, dtype=np.float32, ndmin=2)
    if labels.shape[1] != 5:
        raise ValueError(f"Expected 5 YOLO columns in {path}, found {labels.shape[1]}")
    if np.any(labels[:, 0] != 0):
        raise ValueError(f"Only class 0 (aruco_marker) is supported: {path}")
    if np.any((labels[:, 1:] < 0) | (labels[:, 1:] > 1)):
        raise ValueError(f"YOLO coordinates must be normalized to [0,1]: {path}")
    return labels


class ArucoYoloDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        input_size: int,
        augment: bool = False,
        max_boxes: int = 16,
    ) -> None:
        self.root = Path(root)
        manifest_path = Path(manifest)
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        self.images = [
            self.root / line.strip()
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.images:
            raise ValueError(f"Dataset manifest is empty: {manifest_path}")
        missing_images = [path for path in self.images if not path.is_file()]
        if missing_images:
            preview = ", ".join(str(path) for path in missing_images[:3])
            raise FileNotFoundError(f"Manifest references missing images: {preview}")
        self.input_size = input_size
        self.augment = augment
        self.max_boxes = max_boxes

    def __len__(self) -> int:
        return len(self.images)

    @staticmethod
    def _label_path(image_path: Path) -> Path:
        parts = list(image_path.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        image_path = self.images[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        labels = read_yolo_labels(self._label_path(image_path))

        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 1)
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]
        if self.augment:
            gain = random.uniform(0.75, 1.25)
            bias = random.uniform(-20, 20)
            image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(
                np.uint8
            )

        image = cv2.resize(
            image, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)

        labels = labels[: self.max_boxes]
        target = {
            "boxes_yolo": torch.from_numpy(labels[:, 1:5].copy()),
            "image_id": torch.tensor(index, dtype=torch.int64),
        }
        return tensor, target


def detection_collate(
    batch: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, list[dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)
