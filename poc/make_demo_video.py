from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an offline marker-approach video")
    parser.add_argument("--output", default="artifacts/webcam_poc/aruco_approach.avi")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if not 0 <= args.marker_id < 50:
        raise ValueError("--marker-id must be in [0,49] for DICT_4X4_50")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create video: {output}")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    try:
        for index in range(args.frames):
            progress = index / max(args.frames - 1, 1)
            frame = np.full((args.height, args.width, 3), 210, dtype=np.uint8)
            cv2.line(frame, (0, 350), (args.width, 350), (100, 100, 100), 3)
            cv2.putText(
                frame,
                "Synthetic indoor corridor",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 80, 80),
                2,
            )
            side = int(45 + progress * 210)
            marker = cv2.aruco.generateImageMarker(dictionary, args.marker_id, side)
            marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            center_x = int(args.width * (0.58 - 0.08 * (1 - progress)))
            center_y = int(args.height * 0.52)
            x1 = center_x - side // 2
            y1 = center_y - side // 2
            frame[y1 : y1 + side, x1 : x1 + side] = marker
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created {args.frames} frames: {output}")


if __name__ == "__main__":
    main()
