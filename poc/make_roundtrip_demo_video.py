from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def paste_marker(
    frame: np.ndarray,
    dictionary: cv2.aruco.Dictionary,
    marker_id: int,
    side: int,
    center: tuple[int, int],
) -> None:
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    x1 = center[0] - side // 2
    y1 = center[1] - side // 2
    frame[y1 : y1 + side, x1 : x1 + side] = marker_bgr


def phase_for_frame(index: int) -> tuple[str, int | None, int]:
    if index < 10:
        return "Confirm start marker", 0, 220
    if index < 25:
        return "Outbound: target not visible", None, 0
    if index < 70:
        progress = (index - 25) / 44
        return "Approach target marker", 1, int(45 + 190 * progress)
    if index < 90:
        return "Stop at target", 1, 235
    if index < 120:
        return "Turn 180 degrees", None, 0
    if index < 165:
        progress = (index - 120) / 44
        return "Return to start marker", 0, int(45 + 190 * progress)
    return "Mission complete at start", 0, 235


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a synthetic start-target-start ArUco mission video"
    )
    parser.add_argument("--output", default="artifacts/webcam_poc/roundtrip_input.avi")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.start_id == args.target_id:
        raise ValueError("--start-id and --target-id must be different")
    if not 0 <= args.start_id < 50 or not 0 <= args.target_id < 50:
        raise ValueError("Marker IDs must be in [0,49] for DICT_4X4_50")

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
            phase, canonical_id, side = phase_for_frame(index)
            marker_id = None
            if canonical_id == 0:
                marker_id = args.start_id
            elif canonical_id == 1:
                marker_id = args.target_id

            frame = np.full((args.height, args.width, 3), 210, dtype=np.uint8)
            cv2.line(frame, (0, 360), (args.width, 360), (100, 100, 100), 3)
            cv2.putText(
                frame,
                "Synthetic indoor round-trip mission",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (65, 65, 65),
                2,
            )
            cv2.putText(
                frame,
                f"Input phase: {phase}",
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (80, 80, 80),
                2,
            )
            if marker_id is not None:
                center_x = args.width // 2 + (18 if marker_id == args.target_id else -18)
                paste_marker(
                    frame,
                    dictionary,
                    marker_id,
                    side,
                    (center_x, args.height // 2 + 25),
                )
            writer.write(frame)
    finally:
        writer.release()
    print(f"Created {args.frames} frames: {output}")


if __name__ == "__main__":
    main()
