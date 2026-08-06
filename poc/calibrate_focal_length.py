from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poc.live_webcam_demo import ClassicalDetector, parse_source


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate focal length in pixels from a marker at a known distance"
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-size-m", type=float, required=True)
    parser.add_argument("--distance-m", type=float, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if args.marker_size_m <= 0 or args.distance_m <= 0 or args.samples <= 0:
        raise ValueError("Size, distance and samples must be positive")

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    detector = ClassicalDetector()
    focal_samples: list[float] = []
    try:
        while len(focal_samples) < args.samples:
            ok, frame = capture.read()
            if not ok:
                break
            targets = [
                detection
                for detection in detector.detect(frame)
                if detection.marker_id == args.marker_id
            ]
            if targets:
                target = max(targets, key=lambda item: item.mean_side_px)
                focal_samples.append(
                    target.mean_side_px * args.distance_m / args.marker_size_m
                )
                cv2.putText(
                    frame,
                    f"samples {len(focal_samples)}/{args.samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0),
                    2,
                )
            if not args.headless:
                cv2.imshow("Focal calibration - Q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    if not focal_samples:
        raise RuntimeError("No target marker was detected")
    focal_length = median(focal_samples)
    print(f"focal_length_px={focal_length:.3f} samples={len(focal_samples)}")
    print(f"Use: --focal-length-px {focal_length:.3f}")


if __name__ == "__main__":
    main()
