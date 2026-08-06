from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Split real ArUco data by complete recording session")
    parser.add_argument("--root", default="dataset/aruco")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if abs(args.train + args.val + args.test - 1.0) > 1e-9 or min(args.train, args.val, args.test) <= 0:
        raise ValueError("--train + --val + --test must equal 1 and all must be positive")

    root = Path(args.root)
    images_root = root / "images"
    sessions = [directory for directory in sorted(images_root.iterdir()) if directory.is_dir()]
    if len(sessions) < 3:
        raise RuntimeError("Need at least three session directories under images/ to prevent temporal leakage")
    manifests = [root / f"{name}.txt" for name in ("train", "val", "test")]
    if any(path.exists() for path in manifests) and not args.force:
        raise FileExistsError("Split manifests already exist; pass --force to replace them")

    random.Random(args.seed).shuffle(sessions)
    total = len(sessions)
    train_end = max(1, round(total * args.train))
    val_end = min(total - 1, train_end + max(1, round(total * args.val)))
    groups = {
        "train": sessions[:train_end],
        "val": sessions[train_end:val_end],
        "test": sessions[val_end:],
    }
    if any(not value for value in groups.values()):
        raise RuntimeError("Session ratios produced an empty split; collect more sessions")
    report = {"seed": args.seed, "ratios": {"train": args.train, "val": args.val, "test": args.test}, "splits": {}}
    for split, directories in groups.items():
        images = sorted(
            path.relative_to(root).as_posix()
            for directory in directories
            for path in directory.rglob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        (root / f"{split}.txt").write_text("\n".join(images) + "\n", encoding="utf-8")
        report["splits"][split] = {"sessions": [path.name for path in directories], "images": len(images)}
    (root / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
