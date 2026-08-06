from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from poc.live_webcam_demo import (
    ClassicalDetector,
    MarkerDetection,
    estimate_distance,
    estimate_marker_pose,
)
from poc.mission_controller import MarkerObservation, MissionState, RoundTripController
from poc.robot_controller import RobotState, StopController


class PocTests(unittest.TestCase):
    def make_detection(self):
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 3, 120
        )
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        frame[60:180, 100:220] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        detections = ClassicalDetector().detect(frame)
        self.assertEqual(len(detections), 1)
        return detections[0]

    def test_classical_detection_and_distance(self) -> None:
        detection = self.make_detection()
        self.assertEqual(detection.marker_id, 3)
        distance = estimate_distance(detection, marker_size_m=0.1, focal_length_px=600)
        self.assertAlmostEqual(distance, 0.5, delta=0.02)

    def test_stop_requires_consecutive_confirmations(self) -> None:
        controller = StopController(
            target_id=3,
            stop_side_px=150,
            slow_side_px=80,
            confirm_frames=3,
            smoothing_window=1,
        )
        self.assertEqual(
            controller.update(True, side_px=90).state, RobotState.SLOWING
        )
        self.assertNotEqual(
            controller.update(True, side_px=160).state, RobotState.STOPPED
        )
        self.assertNotEqual(
            controller.update(True, side_px=160).state, RobotState.STOPPED
        )
        self.assertEqual(
            controller.update(True, side_px=160).state, RobotState.STOPPED
        )

    def test_stop_is_latched_until_reset(self) -> None:
        controller = StopController(
            target_id=3,
            stop_side_px=150,
            slow_side_px=80,
            confirm_frames=1,
            smoothing_window=1,
        )
        controller.update(True, side_px=200)
        self.assertEqual(controller.update(False).state, RobotState.STOPPED)
        controller.reset()
        self.assertEqual(controller.update(False).state, RobotState.SEARCHING)

    def test_solvepnp_pose_distance(self) -> None:
        camera_matrix = np.asarray([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
        distortion = np.zeros((5, 1), dtype=np.float64)
        half = 0.05
        object_points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )
        image_points, _ = cv2.projectPoints(
            object_points,
            np.zeros(3),
            np.asarray([0.0, 0.0, 0.8]),
            camera_matrix,
            distortion,
        )
        detection = MarkerDetection(3, image_points.reshape(4, 2), (0, 0, 1, 1), 1.0, "test")
        pose = estimate_marker_pose(detection, 0.10, (camera_matrix, distortion))
        self.assertIsNotNone(pose)
        distance, translation = pose
        self.assertAlmostEqual(distance, 0.8, places=3)
        self.assertAlmostEqual(float(translation[2]), 0.8, places=3)

    def test_roundtrip_state_machine_completes_at_start(self) -> None:
        controller = RoundTripController(
            start_id=0,
            target_id=1,
            stop_side_px=150,
            slow_side_px=80,
            confirm_frames=2,
            start_confirm_frames=2,
            target_dwell_frames=2,
            turn_frames=2,
            smoothing_window=1,
        )
        missing = MarkerObservation(False)
        start_close = MarkerObservation(True, side_px=200)
        target_close = MarkerObservation(True, side_px=200)

        self.assertEqual(controller.update(start_close, missing).state, MissionState.WAITING_FOR_START)
        self.assertEqual(controller.update(start_close, missing).state, MissionState.OUTBOUND)
        self.assertEqual(controller.update(missing, target_close).state, MissionState.OUTBOUND)
        target_command = controller.update(missing, target_close)
        self.assertEqual(target_command.state, MissionState.AT_TARGET)
        self.assertTrue(target_command.target_arrived)

        controller.update(missing, target_close)
        self.assertEqual(controller.update(missing, target_close).state, MissionState.TURNING_HOME)
        controller.update(missing, missing)
        self.assertEqual(controller.update(missing, missing).state, MissionState.RETURNING)
        self.assertEqual(controller.update(start_close, missing).state, MissionState.RETURNING)
        complete = controller.update(start_close, missing)
        self.assertEqual(complete.state, MissionState.HOME_COMPLETE)
        self.assertEqual(complete.linear_speed, 0.0)
        self.assertTrue(complete.mission_complete)
        self.assertEqual(controller.update(missing, missing).state, MissionState.HOME_COMPLETE)

    def test_roundtrip_requires_distinct_marker_ids(self) -> None:
        with self.assertRaises(ValueError):
            RoundTripController(start_id=3, target_id=3)


if __name__ == "__main__":
    unittest.main()
