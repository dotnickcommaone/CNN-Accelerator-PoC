from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import cv2
import numpy as np

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from aruco_detector.loss import DetectionLoss, build_grid_targets
from aruco_detector.metrics import average_precision
from aruco_detector.network import ArucoMobileNetV2, decode_predictions
from export_int8 import fold_batch_norms
from scripts.collect_aruco_dataset import detect_labels


class ModelTests(unittest.TestCase):
    def test_forward_contract(self) -> None:
        model = ArucoMobileNetV2(width_mult=0.35)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 160, 160))
        self.assertEqual(tuple(output.shape), (2, 5, 5, 5))

    def test_target_assignment_and_loss(self) -> None:
        targets = [
            {
                "boxes_yolo": torch.tensor(
                    [[0.5, 0.5, 0.25, 0.3]], dtype=torch.float32
                )
            }
        ]
        obj, boxes, positive = build_grid_targets(
            targets, 5, 5, torch.device("cpu")
        )
        self.assertEqual(int(obj.sum()), 1)
        self.assertEqual(int(positive.sum()), 1)
        prediction = torch.zeros(1, 5, 5, 5, requires_grad=True)
        loss, parts = DetectionLoss()(prediction, targets)
        loss.backward()
        self.assertGreater(parts["loss"], 0)
        self.assertIsNotNone(prediction.grad)
        self.assertEqual(tuple(boxes.shape), (1, 4, 5, 5))

    def test_decode_empty_output(self) -> None:
        output = torch.full((1, 5, 5, 5), -20.0)
        decoded = decode_predictions(output, score_threshold=0.5)
        self.assertEqual(tuple(decoded[0]["boxes"].shape), (0, 4))

    def test_average_precision(self) -> None:
        detections = [(0.9, True), (0.8, False), (0.7, True)]
        self.assertAlmostEqual(average_precision(detections, 2), 5 / 6, places=5)

    def test_pretrained_width_mismatch_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "pretrained MobileNetV2-0.35"):
            ArucoMobileNetV2(width_mult=0.35, pretrained=True)

    def test_conv_batch_norm_folding(self) -> None:
        model = ArucoMobileNetV2(width_mult=0.35).eval()
        image = torch.rand(1, 3, 160, 160)
        with torch.inference_mode():
            expected = model(image)
            actual = fold_batch_norms(model)(image)
        self.assertLess(float((expected - actual).abs().max()), 1e-4)

    def test_aruco_auto_label(self) -> None:
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 120
        )
        image = np.full((200, 200, 3), 255, dtype=np.uint8)
        image[40:160, 40:160] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        labels, _ = detect_labels(image)
        self.assertEqual(len(labels), 1)
        self.assertEqual(int(labels[0][0]), 0)
        self.assertAlmostEqual(labels[0][1], 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
