from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model"))
sys.path.insert(0, str(REPO_ROOT))

from aruco_detector.config import load_config, select_device
from aruco_detector.network import ArucoMobileNetV2, decode_predictions
from infer_aruco import decode_marker_roi, preprocess
from poc.mission_controller import MarkerObservation, MissionCommand, RoundTripController
from poc.robot_controller import RobotCommand, StopController


@dataclass
class MarkerDetection:
    marker_id: int
    corners: np.ndarray
    box_xyxy: tuple[int, int, int, int]
    score: float
    source: str

    @property
    def mean_side_px(self) -> float:
        points = self.corners.reshape(4, 2)
        return float(
            np.mean(
                [
                    np.linalg.norm(points[(index + 1) % 4] - points[index])
                    for index in range(4)
                ]
            )
        )


@dataclass
class MeasuredMarker:
    detection: MarkerDetection | None
    distance_m: float | None = None
    distance_method: str = "none"
    translation_m: np.ndarray | None = None


class ClassicalDetector:
    def __init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters()
        )

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return []
        detections = []
        for marker_id, marker_corners in zip(ids.flatten(), corners):
            points = marker_corners.reshape(4, 2)
            x_min, y_min = points.min(axis=0).astype(int)
            x_max, y_max = points.max(axis=0).astype(int)
            detections.append(
                MarkerDetection(
                    int(marker_id),
                    points,
                    (int(x_min), int(y_min), int(x_max), int(y_max)),
                    1.0,
                    "opencv",
                )
            )
        return detections


class CnnRoiDetector:
    def __init__(self, config_path: str, checkpoint_path: str) -> None:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            candidates = sorted((REPO_ROOT / "artifacts").glob("**/best.pt"))
            available = "\n".join(f"  - {path.relative_to(REPO_ROOT)}" for path in candidates)
            raise FileNotFoundError(
                f"Checkpoint does not exist: {checkpoint_file}\n"
                "Train the model, select an existing checkpoint, or use --mode classical.\n"
                f"Available checkpoints:\n{available or '  (none)'}"
            )
        config = load_config(config_path)
        self.device = select_device(config["device"])
        self.input_size = config["model"]["input_size"]
        self.score_threshold = config["model"]["score_threshold"]
        self.nms_threshold = config["model"]["nms_iou_threshold"]
        self.model = ArucoMobileNetV2(config["model"]["width_mult"]).to(self.device)
        checkpoint = torch.load(
            checkpoint_file, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        with torch.inference_mode():
            self.model(
                torch.zeros(1, 3, self.input_size, self.input_size, device=self.device)
            )

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        with torch.inference_mode():
            raw = self.model(preprocess(frame, self.input_size).to(self.device))
            predictions = decode_predictions(
                raw, self.score_threshold, self.nms_threshold
            )[0]
        height, width = frame.shape[:2]
        detections: list[MarkerDetection] = []
        for box_tensor, score_tensor in zip(
            predictions["boxes"], predictions["scores"]
        ):
            box = box_tensor.detach().cpu().numpy()
            ids, corners_groups = decode_marker_roi(frame, box)
            for marker_id, corners in zip(ids, corners_groups):
                points = np.asarray(corners, dtype=np.float32)
                x1, y1, x2, y2 = box
                detections.append(
                    MarkerDetection(
                        marker_id,
                        points,
                        (
                            int(x1 * width),
                            int(y1 * height),
                            int(x2 * width),
                            int(y2 * height),
                        ),
                        float(score_tensor),
                        "cnn+opencv",
                    )
                )
        return detections


class HybridDetector:
    def __init__(self, cnn: CnnRoiDetector) -> None:
        self.cnn = cnn
        self.classical = ClassicalDetector()

    def detect(self, frame: np.ndarray) -> list[MarkerDetection]:
        detections = self.cnn.detect(frame)
        return detections if detections else self.classical.detect(frame)


def estimate_distance(
    detection: MarkerDetection, marker_size_m: float, focal_length_px: float | None
) -> float | None:
    if focal_length_px is None or detection.mean_side_px <= 0:
        return None
    return marker_size_m * focal_length_px / detection.mean_side_px


def load_camera_calibration(path: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if path is None:
        return None
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Camera calibration does not exist: {calibration_path}")
    data = np.load(calibration_path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)


def estimate_marker_pose(
    detection: MarkerDetection,
    marker_size_m: float,
    calibration: tuple[np.ndarray, np.ndarray],
) -> tuple[float, np.ndarray] | None:
    half = marker_size_m / 2.0
    object_points = np.asarray(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
        dtype=np.float32,
    )
    camera_matrix, dist_coeffs = calibration
    success, _rvec, tvec = cv2.solvePnP(
        object_points,
        detection.corners.reshape(4, 2).astype(np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    translation = tvec.reshape(3)
    if float(translation[2]) <= 0:
        return None
    return float(np.linalg.norm(translation)), translation


def measure_marker(
    detections: list[MarkerDetection],
    marker_id: int,
    marker_size_m: float,
    focal_length_px: float | None,
    calibration: tuple[np.ndarray, np.ndarray] | None,
) -> MeasuredMarker:
    matches = [detection for detection in detections if detection.marker_id == marker_id]
    detection = max(matches, key=lambda item: item.mean_side_px, default=None)
    if detection is None:
        return MeasuredMarker(None)
    if calibration is not None:
        pose = estimate_marker_pose(detection, marker_size_m, calibration)
        if pose is not None:
            distance, translation = pose
            return MeasuredMarker(detection, distance, "solvepnp", translation)
    if focal_length_px is not None:
        distance = estimate_distance(detection, marker_size_m, focal_length_px)
        return MeasuredMarker(detection, distance, "pinhole")
    return MeasuredMarker(detection, distance_method="pixel_proxy")


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def sha256_file(path: str | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def draw_overlay(
    frame: np.ndarray,
    detections: list[MarkerDetection],
    command: RobotCommand | MissionCommand,
    target_id: int,
    fps: float,
    distance_m: float | None,
    mode: str,
    mission: str = "stop",
    start_id: int | None = None,
) -> None:
    for detection in detections:
        if detection.marker_id == target_id:
            color = (0, 220, 255)
        elif start_id is not None and detection.marker_id == start_id:
            color = (255, 180, 0)
        else:
            color = (0, 200, 0)
        points = detection.corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], True, color, 2)
        x1, y1, x2, y2 = detection.box_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame,
            f"ID {detection.marker_id} {detection.source} {detection.score:.2f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 86), (20, 20, 20), -1)
    distance_text = (
        f"{distance_m:.2f} m" if distance_m is not None else "pixel proxy"
    )
    marker_text = f"start ID {start_id} | target ID {target_id}" if start_id is not None else f"target ID {target_id}"
    angular_speed = float(getattr(command, "angular_speed", 0.0))
    lines = [
        f"{mode.upper()} | {mission.upper()} | FPS {fps:.1f} | {marker_text} | {distance_text}",
        f"{command.state.value} | v {command.speed:.2f} | w {angular_speed:.2f} | {command.reason}",
        "Q/ESC: quit | R: reset mission | S: save snapshot",
    ]
    colors = [
        (230, 230, 230),
        (0, 80, 255) if command.speed == 0 else (0, 220, 255),
        (180, 180, 180),
    ]
    for index, (line, color) in enumerate(zip(lines, colors)):
        cv2.putText(
            frame,
            line,
            (10, 22 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )


def make_detector(args: argparse.Namespace):
    if args.mode == "classical":
        return ClassicalDetector()
    if not args.checkpoint:
        raise ValueError(f"--checkpoint is required in {args.mode} mode")
    cnn = CnnRoiDetector(args.config, args.checkpoint)
    return cnn if args.mode == "cnn" else HybridDetector(cnn)


def main() -> None:
    parser = argparse.ArgumentParser(description="USB webcam ArUco robot mission PoC")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument(
        "--mode", choices=("classical", "cnn", "hybrid"), default="classical"
    )
    parser.add_argument("--config", default="model/configs/mobilenetv2_035.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--mission",
        choices=("stop", "roundtrip"),
        default="stop",
        help="One-way stop or start-target-start round trip",
    )
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--marker-size-m", type=float, default=0.10)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--camera-calibration", default=None, help="NPZ from poc/calibrate_camera.py")
    parser.add_argument("--stop-distance-m", type=float, default=0.45)
    parser.add_argument("--slow-distance-m", type=float, default=0.90)
    parser.add_argument("--stop-side-px", type=float, default=180.0)
    parser.add_argument("--slow-side-px", type=float, default=100.0)
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--start-confirm-frames", type=int, default=3)
    parser.add_argument("--target-dwell-frames", type=int, default=15)
    parser.add_argument("--turn-frames", type=int, default=30)
    parser.add_argument("--turn-angular-speed", type=float, default=0.50)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--exit-on-complete",
        action="store_true",
        help="Exit automatically after STOPPED or HOME_COMPLETE",
    )
    parser.add_argument("--complete-hold-frames", type=int, default=15)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-video", default=None)
    parser.add_argument("--csv", default="artifacts/webcam_poc/metrics.csv")
    parser.add_argument("--snapshot-dir", default="artifacts/webcam_poc/snapshots")
    args = parser.parse_args()
    if args.complete_hold_frames < 1:
        parser.error("--complete-hold-frames must be positive")

    try:
        detector = make_detector(args)
        camera_calibration = load_camera_calibration(args.camera_calibration)
        if args.mission == "roundtrip":
            controller = RoundTripController(
                start_id=args.start_id,
                target_id=args.target_id,
                stop_distance_m=args.stop_distance_m,
                slow_distance_m=args.slow_distance_m,
                stop_side_px=args.stop_side_px,
                slow_side_px=args.slow_side_px,
                confirm_frames=args.confirm_frames,
                start_confirm_frames=args.start_confirm_frames,
                target_dwell_frames=args.target_dwell_frames,
                turn_frames=args.turn_frames,
                turn_angular_speed=args.turn_angular_speed,
            )
        else:
            controller = StopController(
                args.target_id,
                stop_distance_m=args.stop_distance_m,
                slow_distance_m=args.slow_distance_m,
                stop_side_px=args.stop_side_px,
                slow_side_px=args.slow_side_px,
                confirm_frames=args.confirm_frames,
            )
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))
    parsed_source = parse_source(args.source)
    capture = cv2.VideoCapture(parsed_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    if isinstance(parsed_source, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.camera_fps)

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = csv_path.with_suffix(".meta.json")
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command_arguments": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "device": str(getattr(detector, "device", "cpu")),
        },
        "capture_actual": {
            "width": capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": capture.get(cv2.CAP_PROP_FPS),
        },
        "sha256": {
            "checkpoint": sha256_file(args.checkpoint),
            "input_file": sha256_file(args.source),
            "camera_calibration": sha256_file(args.camera_calibration),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    snapshot_dir = Path(args.snapshot_dir)
    writer = None
    frame_index = 0
    fps_ema = 0.0
    vision_total_ms = 0.0
    total_total_ms = 0.0
    target_frame_count = 0
    start_frame_count = 0
    first_stop_frame: int | None = None
    first_target_frame: int | None = None
    home_complete_frame: int | None = None
    completion_detected_frame: int | None = None

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
            fields = [
                "timestamp",
                "frame",
                "mode",
                "mission",
                "vision_ms",
                "total_ms",
                "fps",
                "marker_ids",
                "start_id",
                "start_visible",
                "start_source",
                "start_distance_m",
                "start_distance_method",
                "start_pose_x_m",
                "start_pose_y_m",
                "start_pose_z_m",
                "start_side_px",
                "target_id",
                "target_visible",
                "target_source",
                "distance_m",
                "distance_method",
                "pose_x_m",
                "pose_y_m",
                "pose_z_m",
                "side_px",
                "state",
                "speed",
                "linear_speed",
                "angular_speed",
                "active_marker_id",
                "target_arrived",
                "mission_complete",
            ]
            log = csv.DictWriter(csv_stream, fieldnames=fields)
            log.writeheader()
            while True:
                frame_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break
                vision_start = time.perf_counter()
                detections = detector.detect(frame)
                vision_ms = (time.perf_counter() - vision_start) * 1000
                target_measurement = measure_marker(
                    detections,
                    args.target_id,
                    args.marker_size_m,
                    args.focal_length_px,
                    camera_calibration,
                )
                start_measurement = (
                    measure_marker(
                        detections,
                        args.start_id,
                        args.marker_size_m,
                        args.focal_length_px,
                        camera_calibration,
                    )
                    if args.mission == "roundtrip"
                    else MeasuredMarker(None)
                )
                target = target_measurement.detection
                start = start_measurement.detection
                if args.mission == "roundtrip":
                    command = controller.update(
                        MarkerObservation(
                            start is not None,
                            start_measurement.distance_m,
                            start.mean_side_px if start else None,
                        ),
                        MarkerObservation(
                            target is not None,
                            target_measurement.distance_m,
                            target.mean_side_px if target else None,
                        ),
                    )
                    display_distance = (
                        start_measurement.distance_m
                        if command.active_marker_id == args.start_id
                        else target_measurement.distance_m
                    )
                else:
                    command = controller.update(
                        target is not None,
                        distance_m=target_measurement.distance_m,
                        side_px=target.mean_side_px if target else None,
                    )
                    display_distance = target_measurement.distance_m
                total_ms = (time.perf_counter() - frame_start) * 1000
                vision_total_ms += vision_ms
                total_total_ms += total_ms
                target_frame_count += int(target is not None)
                start_frame_count += int(start is not None)
                if command.state.value == "STOPPED" and first_stop_frame is None:
                    first_stop_frame = frame_index
                if command.state.value == "AT_TARGET" and first_target_frame is None:
                    first_target_frame = frame_index
                if command.state.value == "HOME_COMPLETE" and home_complete_frame is None:
                    home_complete_frame = frame_index
                if (
                    command.state.value in {"STOPPED", "HOME_COMPLETE"}
                    and completion_detected_frame is None
                ):
                    completion_detected_frame = frame_index
                instantaneous_fps = 1000 / max(total_ms, 1e-9)
                fps_ema = (
                    instantaneous_fps
                    if frame_index == 0
                    else 0.9 * fps_ema + 0.1 * instantaneous_fps
                )
                draw_overlay(
                    frame,
                    detections,
                    command,
                    args.target_id,
                    fps_ema,
                    display_distance,
                    args.mode,
                    args.mission,
                    args.start_id if args.mission == "roundtrip" else None,
                )
                if args.output_video:
                    if writer is None:
                        output = Path(args.output_video)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(output),
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            capture.get(cv2.CAP_PROP_FPS) or 20.0,
                            (frame.shape[1], frame.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create output video: {output}")
                    writer.write(frame)
                log.writerow(
                    {
                        "timestamp": time.time(),
                        "frame": frame_index,
                        "mode": args.mode,
                        "mission": args.mission,
                        "vision_ms": f"{vision_ms:.3f}",
                        "total_ms": f"{total_ms:.3f}",
                        "fps": f"{fps_ema:.3f}",
                        "marker_ids": "|".join(
                            str(detection.marker_id) for detection in detections
                        ),
                        "start_id": args.start_id if args.mission == "roundtrip" else "",
                        "start_visible": int(start is not None),
                        "start_source": "" if start is None else start.source,
                        "start_distance_m": "" if start_measurement.distance_m is None else f"{start_measurement.distance_m:.4f}",
                        "start_distance_method": start_measurement.distance_method,
                        "start_pose_x_m": "" if start_measurement.translation_m is None else f"{start_measurement.translation_m[0]:.5f}",
                        "start_pose_y_m": "" if start_measurement.translation_m is None else f"{start_measurement.translation_m[1]:.5f}",
                        "start_pose_z_m": "" if start_measurement.translation_m is None else f"{start_measurement.translation_m[2]:.5f}",
                        "start_side_px": "" if start is None else f"{start.mean_side_px:.2f}",
                        "target_id": args.target_id,
                        "target_visible": int(target is not None),
                        "target_source": "" if target is None else target.source,
                        "distance_m": "" if target_measurement.distance_m is None else f"{target_measurement.distance_m:.4f}",
                        "distance_method": target_measurement.distance_method,
                        "pose_x_m": "" if target_measurement.translation_m is None else f"{target_measurement.translation_m[0]:.5f}",
                        "pose_y_m": "" if target_measurement.translation_m is None else f"{target_measurement.translation_m[1]:.5f}",
                        "pose_z_m": "" if target_measurement.translation_m is None else f"{target_measurement.translation_m[2]:.5f}",
                        "side_px": "" if target is None else f"{target.mean_side_px:.2f}",
                        "state": command.state.value,
                        "speed": f"{command.speed:.2f}",
                        "linear_speed": f"{command.speed:.2f}",
                        "angular_speed": f"{float(getattr(command, 'angular_speed', 0.0)):.2f}",
                        "active_marker_id": getattr(command, "active_marker_id", args.target_id),
                        "target_arrived": int(bool(getattr(command, "target_arrived", command.state.value == "STOPPED"))),
                        "mission_complete": int(bool(getattr(command, "mission_complete", command.state.value == "STOPPED"))),
                    }
                )
                csv_stream.flush()

                key = -1
                if not args.headless:
                    cv2.imshow("Indoor delivery robot - ArUco PoC", frame)
                    key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    controller.reset()
                if key == ord("s"):
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snapshot_dir / f"frame_{frame_index:06d}.jpg"), frame)
                frame_index += 1
                if (
                    args.exit_on_complete
                    and completion_detected_frame is not None
                    and frame_index - completion_detected_frame >= args.complete_hold_frames
                ):
                    break
                if args.max_frames and frame_index >= args.max_frames:
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    processed = max(frame_index, 1)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["result"] = {
        "frames": frame_index,
        "average_vision_ms": vision_total_ms / processed,
        "average_total_ms": total_total_ms / processed,
        "target_frames": target_frame_count,
        "start_frames": start_frame_count,
        "first_stop_frame": first_stop_frame,
        "first_target_frame": first_target_frame,
        "home_complete_frame": home_complete_frame,
        "mission_completed": home_complete_frame is not None or first_stop_frame is not None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"Processed {frame_index} frames; "
        f"avg_vision_ms={vision_total_ms / processed:.3f}; "
        f"avg_total_ms={total_total_ms / processed:.3f}; "
        f"target_frames={target_frame_count}; "
        f"start_frames={start_frame_count}; "
        f"first_target_frame={first_target_frame}; "
        f"home_complete_frame={home_complete_frame}; "
        f"first_stop_frame={first_stop_frame}; metrics={csv_path}; metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
