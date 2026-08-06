from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from live_webcam_demo import parse_source


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, pattern)
        return corners.astype(np.float32) if found else None
    found, corners = cv2.findChessboardCorners(gray, pattern)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size_m
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate USB camera with a checkerboard")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Camera index or video path")
    source.add_argument("--images", help="Glob, e.g. artifacts/calibration/*.jpg")
    parser.add_argument("--board-cols", type=int, default=9)
    parser.add_argument("--board-rows", type=int, default=6)
    parser.add_argument("--square-size-m", type=float, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--output", default="artifacts/calibration/camera.npz")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    if min(args.board_cols, args.board_rows, args.samples, args.sample_every) <= 0 or args.square_size_m <= 0:
        raise ValueError("Board, sample and square-size values must be positive")

    pattern = (args.board_cols, args.board_rows)
    template = object_points(*pattern, args.square_size_m)
    object_sets: list[np.ndarray] = []
    image_sets: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    def accept(frame: np.ndarray) -> bool:
        nonlocal image_size
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        corners = find_corners(gray, pattern)
        if corners is None:
            return False
        object_sets.append(template.copy())
        image_sets.append(corners)
        if not args.headless:
            cv2.drawChessboardCorners(frame, pattern, corners, True)
        return True

    if args.images:
        for path in sorted(Path().glob(args.images)):
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is not None:
                accept(frame)
    else:
        capture = cv2.VideoCapture(parse_source(args.source))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open source: {args.source}")
        frame_index = 0
        try:
            while len(image_sets) < args.samples:
                ok, frame = capture.read()
                if not ok:
                    break
                accepted = frame_index % args.sample_every == 0 and accept(frame)
                if not args.headless:
                    cv2.putText(frame, f"samples {len(image_sets)}/{args.samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0) if accepted else (0, 0, 220), 2)
                    cv2.imshow("Camera calibration - Q to quit", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                frame_index += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()

    if image_size is None or len(image_sets) < 5:
        raise RuntimeError(f"Need at least 5 valid checkerboard views; collected {len(image_sets)}")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_sets, image_sets, image_size, None, None)
    per_view_errors = []
    for object_set, image_set, rvec, tvec in zip(object_sets, image_sets, rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_set, rvec, tvec, camera_matrix, dist_coeffs)
        per_view_errors.append(float(cv2.norm(image_set, projected, cv2.NORM_L2) / len(projected)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, image_size=np.asarray(image_size, dtype=np.int32), rms=np.asarray(rms), per_view_errors=np.asarray(per_view_errors), board_size=np.asarray(pattern), square_size_m=np.asarray(args.square_size_m))
    report = {
        "output": str(output), "views": len(image_sets), "image_size": list(image_size),
        "board_inner_corners": list(pattern), "square_size_m": args.square_size_m,
        "rms_reprojection_error": float(rms), "mean_per_view_error_px": float(np.mean(per_view_errors)),
        "camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
