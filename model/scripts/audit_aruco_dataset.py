from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        return image_path.with_suffix(".txt")
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def load_manifest(root: Path, name: str) -> list[Path]:
    path = root / name
    if not path.is_file():
        return []
    return [root / line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ArUco YOLO dataset and export raw tables")
    parser.add_argument("--root", default="dataset/aruco")
    parser.add_argument("--output", default="artifacts/dataset_audit")
    parser.add_argument("--hash-images", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_names = ("train", "val", "test")
    split_paths = {name: load_manifest(root, f"{name}.txt") for name in split_names}
    path_to_splits: dict[Path, list[str]] = {}
    for split, paths in split_paths.items():
        for path in paths:
            path_to_splits.setdefault(path.resolve(), []).append(split)

    all_images = sorted((root / "images").rglob("*")) if (root / "images").is_dir() else []
    all_images = [path for path in all_images if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    image_rows: list[dict] = []
    box_rows: list[dict] = []
    errors: list[str] = []
    hashes: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    for image_path in all_images:
        relative = image_path.relative_to(root).as_posix()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append(f"unreadable_image:{relative}")
            continue
        height, width = image.shape[:2]
        label_path = label_path_for(image_path)
        labels = np.empty((0, 5), dtype=np.float32)
        if not label_path.exists():
            errors.append(f"missing_label:{relative}")
        elif label_path.stat().st_size:
            try:
                labels = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
                if labels.shape[1] != 5:
                    raise ValueError(f"columns={labels.shape[1]}")
            except Exception as error:
                errors.append(f"invalid_label:{relative}:{error}")
                labels = np.empty((0, 5), dtype=np.float32)
        digest = ""
        if args.hash_images:
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            hashes[digest] += 1
        splits = path_to_splits.get(image_path.resolve(), [])
        image_rows.append(
            {
                "image": relative,
                "session": image_path.parent.name if image_path.parent.name != "images" else "ungrouped",
                "splits": "|".join(splits),
                "width_px": width,
                "height_px": height,
                "boxes": len(labels),
                "negative": int(len(labels) == 0),
                "sha256": digest,
            }
        )
        for box_index, row in enumerate(labels):
            class_id, cx, cy, box_width, box_height = map(float, row)
            if class_id != 0 or min(cx, cy, box_width, box_height) < 0 or max(cx, cy, box_width, box_height) > 1:
                errors.append(f"out_of_range:{relative}:box={box_index}")
            area = box_width * box_height
            widths.append(box_width)
            heights.append(box_height)
            areas.append(area)
            box_rows.append(
                {
                    "image": relative,
                    "box_index": box_index,
                    "class_id": int(class_id),
                    "center_x": cx,
                    "center_y": cy,
                    "width_norm": box_width,
                    "height_norm": box_height,
                    "area_norm": area,
                    "width_px": box_width * width,
                    "height_px": box_height * height,
                }
            )

    overlaps = {path: splits for path, splits in path_to_splits.items() if len(splits) > 1}
    manifest_missing = [str(path.relative_to(root)) for paths in split_paths.values() for path in paths if not path.is_file()]
    duplicate_hashes = sum(1 for count in hashes.values() if count > 1)
    summary = {
        "root": str(root),
        "images": len(image_rows),
        "boxes": len(box_rows),
        "negative_images": sum(row["negative"] for row in image_rows),
        "splits": {name: len(paths) for name, paths in split_paths.items()},
        "split_overlap_images": len(overlaps),
        "manifest_missing_images": len(manifest_missing),
        "errors": len(errors),
        "duplicate_hash_groups": duplicate_hashes if args.hash_images else None,
        "box_area_norm": {
            "median": percentile(areas, 50),
            "p05": percentile(areas, 5),
            "p95": percentile(areas, 95),
        },
        "box_width_norm": {"median": percentile(widths, 50), "p05": percentile(widths, 5), "p95": percentile(widths, 95)},
        "box_height_norm": {"median": percentile(heights, 50), "p05": percentile(heights, 5), "p95": percentile(heights, 95)},
    }

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "images.csv", image_rows)
    write_csv(output / "boxes.csv", box_rows)
    split_rows = [
        {
            "split": name,
            "images": len(paths),
            "fraction": len(paths) / max(sum(len(value) for value in split_paths.values()), 1),
        }
        for name, paths in split_paths.items()
    ]
    write_csv(output / "split_summary.csv", split_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "errors.txt").write_text("\n".join(errors + [f"split_overlap:{path}:{splits}" for path, splits in overlaps.items()] + [f"manifest_missing:{path}" for path in manifest_missing]), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors or overlaps or manifest_missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
