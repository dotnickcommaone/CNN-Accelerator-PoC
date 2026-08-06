from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def marker_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def random_background(height: int, width: int) -> np.ndarray:
    base = np.random.randint(30, 225, size=(1, 1, 3), dtype=np.uint8)
    noise = np.random.normal(0, 24, size=(height, width, 3))
    image = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for _ in range(random.randint(2, 8)):
        color = tuple(int(value) for value in np.random.randint(0, 256, size=3))
        point1 = (random.randrange(width), random.randrange(height))
        point2 = (random.randrange(width), random.randrange(height))
        cv2.rectangle(image, point1, point2, color, random.choice((1, 2, -1)))
    return image


def render_sample(image_size: int, negative: bool) -> tuple[np.ndarray, list[float] | None]:
    image = random_background(image_size, image_size)
    if negative:
        return image, None

    marker_size = random.randint(image_size // 7, image_size // 2)
    marker_id = random.randrange(50)
    marker = cv2.aruco.generateImageMarker(marker_dictionary(), marker_id, marker_size)
    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    margin = max(5, marker_size // 4)
    canvas_size = marker_size + 2 * margin
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    canvas[margin : margin + marker_size, margin : margin + marker_size] = marker

    center_x = random.randint(canvas_size // 2, image_size - canvas_size // 2)
    center_y = random.randint(canvas_size // 2, image_size - canvas_size // 2)
    half = canvas_size / 2
    source = np.float32(
        [[0, 0], [canvas_size - 1, 0], [canvas_size - 1, canvas_size - 1], [0, canvas_size - 1]]
    )
    jitter = marker_size * 0.22
    destination = np.float32(
        [
            [center_x - half + random.uniform(-jitter, jitter), center_y - half + random.uniform(-jitter, jitter)],
            [center_x + half + random.uniform(-jitter, jitter), center_y - half + random.uniform(-jitter, jitter)],
            [center_x + half + random.uniform(-jitter, jitter), center_y + half + random.uniform(-jitter, jitter)],
            [center_x - half + random.uniform(-jitter, jitter), center_y + half + random.uniform(-jitter, jitter)],
        ]
    )
    destination[:, 0] = np.clip(destination[:, 0], 0, image_size - 1)
    destination[:, 1] = np.clip(destination[:, 1], 0, image_size - 1)
    transform = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(canvas, transform, (image_size, image_size))
    mask = cv2.warpPerspective(
        np.full((canvas_size, canvas_size), 255, dtype=np.uint8),
        transform,
        (image_size, image_size),
    )
    image[mask > 0] = warped[mask > 0]
    if random.random() < 0.35:
        kernel = random.choice((3, 5))
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)

    x_min, y_min = destination.min(axis=0)
    x_max, y_max = destination.max(axis=0)
    cx = ((x_min + x_max) / 2) / image_size
    cy = ((y_min + y_max) / 2) / image_size
    width = (x_max - x_min) / image_size
    height = (y_max - y_min) / image_size
    return image, [0, cx, cy, width, height]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ArUco detection data")
    parser.add_argument("--output", default="dataset/aruco")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--negative-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 10:
        raise ValueError("--count must be at least 10 to create train/val/test splits")
    if not 0 <= args.negative_fraction < 1:
        raise ValueError("--negative-fraction must be in [0,1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    root = Path(args.output)
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    relative_paths: list[str] = []

    for index in range(args.count):
        image, label = render_sample(
            args.image_size, random.random() < args.negative_fraction
        )
        stem = f"synthetic_{index:06d}"
        image_path = image_dir / f"{stem}.jpg"
        label_path = label_dir / f"{stem}.txt"
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        text = "" if label is None else " ".join(f"{value:.7f}" for value in label) + "\n"
        label_path.write_text(text, encoding="utf-8")
        relative_paths.append(image_path.relative_to(root).as_posix())

    random.shuffle(relative_paths)
    train_end = int(0.8 * len(relative_paths))
    val_end = int(0.9 * len(relative_paths))
    splits = {
        "train": relative_paths[:train_end],
        "val": relative_paths[train_end:val_end],
        "test": relative_paths[val_end:],
    }
    for split, paths in splits.items():
        (root / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    print(
        f"Created {args.count} samples in {root}: "
        + ", ".join(f"{name}={len(paths)}" for name, paths in splits.items())
    )


if __name__ == "__main__":
    main()
