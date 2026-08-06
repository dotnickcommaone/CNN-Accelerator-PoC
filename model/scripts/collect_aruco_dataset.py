from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def open_source(value: str) -> cv2.VideoCapture:
    source: int | str = int(value) if value.isdigit() else value
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {value}")
    return capture


def detect_labels(image: np.ndarray) -> tuple[list[list[float]], np.ndarray]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    height, width = image.shape[:2]
    labels: list[list[float]] = []
    annotated = image.copy()
    if ids is None:
        return labels, annotated
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    for marker_corners in corners:
        points = marker_corners.reshape(-1, 2)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        labels.append(
            [
                0,
                float((x_min + x_max) / (2 * width)),
                float((y_min + y_max) / (2 * height)),
                float((x_max - x_min) / width),
                float((y_max - y_min) / height),
            ]
        )
    return labels, annotated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real camera frames and auto-label visible ArUco markers"
    )
    parser.add_argument("--source", default="0", help="Camera index, video, or stream URL")
    parser.add_argument("--output", default="dataset/aruco")
    parser.add_argument("--session", default=None)
    parser.add_argument("--every", type=int, default=10, help="Save every Nth frame")
    parser.add_argument("--max-saved", type=int, default=500)
    parser.add_argument("--include-negatives", action="store_true")
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if args.every <= 0 or args.max_saved <= 0:
        raise ValueError("--every and --max-saved must be positive")

    session = args.session or time.strftime("session_%Y%m%d_%H%M%S")
    root = Path(args.output)
    image_dir = root / "images" / session
    label_dir = root / "labels" / session
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    capture = open_source(args.source)
    frame_index = saved = 0
    session_paths: list[str] = []

    try:
        while saved < args.max_saved:
            ok, frame = capture.read()
            if not ok:
                break
            labels, annotated = detect_labels(frame)
            if frame_index % args.every == 0 and (labels or args.include_negatives):
                stem = f"frame_{frame_index:08d}"
                image_path = image_dir / f"{stem}.jpg"
                label_path = label_dir / f"{stem}.txt"
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Could not write image: {image_path}")
                label_path.write_text(
                    "".join(
                        " ".join(f"{value:.7f}" for value in label) + "\n"
                        for label in labels
                    ),
                    encoding="utf-8",
                )
                session_paths.append(image_path.relative_to(root).as_posix())
                saved += 1
            frame_index += 1
            if args.display:
                cv2.imshow("ArUco collection - Q to stop", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    manifest = root / f"{session}.txt"
    manifest.write_text("\n".join(session_paths) + ("\n" if session_paths else ""), encoding="utf-8")
    print(f"Saved {saved} frames; session manifest: {manifest}")


if __name__ == "__main__":
    main()
