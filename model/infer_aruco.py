from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aruco_detector.config import load_config, select_device
from aruco_detector.network import ArucoMobileNetV2, decode_predictions


def preprocess(image_bgr: np.ndarray, input_size: int) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (input_size, input_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)


def decode_marker_roi(
    image_bgr: np.ndarray, box: np.ndarray, padding: float = 0.20
) -> tuple[list[int], list[list[list[float]]]]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int((x1 - pad_x) * width))
    top = max(0, int((y1 - pad_y) * height))
    right = min(width, int((x2 + pad_x) * width))
    bottom = min(height, int((y2 + pad_y) * height))
    roi = image_bgr[top:bottom, left:right]
    if roi.size == 0:
        return [], []
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(roi)
    if ids is None:
        return [], []
    global_corners: list[list[list[float]]] = []
    for marker_corners in corners:
        points = marker_corners.reshape(-1, 2)
        points[:, 0] += left
        points[:, 1] += top
        global_corners.append(points.tolist())
    return ids.flatten().astype(int).tolist(), global_corners


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect marker ROI with CNN and decode its ArUco ID"
    )
    parser.add_argument("--config", default="model/configs/mobilenetv2_035.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device(config["device"])
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    model = ArucoMobileNetV2(config["model"]["width_mult"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.inference_mode():
        raw = model(preprocess(image, config["model"]["input_size"]).to(device))
        predictions = decode_predictions(
            raw,
            config["model"]["score_threshold"],
            config["model"]["nms_iou_threshold"],
        )[0]

    height, width = image.shape[:2]
    detections = []
    for box_tensor, score_tensor in zip(predictions["boxes"], predictions["scores"]):
        box = box_tensor.detach().cpu().numpy()
        ids, corners = decode_marker_roi(image, box)
        x1, y1, x2, y2 = box
        pixel_box = [
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        ]
        detections.append(
            {
                "score": float(score_tensor),
                "box_xyxy": pixel_box,
                "aruco_ids": ids,
                "aruco_corners": corners,
            }
        )
        cv2.rectangle(
            image, (pixel_box[0], pixel_box[1]), (pixel_box[2], pixel_box[3]), (0, 255, 0), 2
        )
        label = f"CNN {float(score_tensor):.2f}"
        if ids:
            label += f" ID={','.join(map(str, ids))}"
        cv2.putText(
            image,
            label,
            (pixel_box[0], max(18, pixel_box[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for marker_corners in corners:
            cv2.polylines(
                image,
                [np.asarray(marker_corners, dtype=np.int32)],
                True,
                (0, 180, 255),
                2,
            )

    result = {"image": args.image, "detections": detections}
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise OSError(f"Could not write output image: {output}")


if __name__ == "__main__":
    main()
