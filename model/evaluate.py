from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aruco_detector.config import load_config, select_device
from aruco_detector.dataset import ArucoYoloDataset, detection_collate
from aruco_detector.metrics import average_precision, detection_counts, ranked_detections
from aruco_detector.network import ArucoMobileNetV2, decode_predictions


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ArUco detector")
    parser.add_argument("--config", default="model/configs/mobilenetv2_035.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--predictions-csv", default=None)
    parser.add_argument("--pr-curve-csv", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root is not None:
        config["dataset"]["root"] = args.dataset_root
    device = select_device(config["device"])
    dataset_config = config["dataset"]
    dataset = ArucoYoloDataset(
        dataset_config["root"],
        dataset_config[f"{args.split}_manifest"],
        config["model"]["input_size"],
        augment=False,
        max_boxes=dataset_config["max_boxes"],
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        collate_fn=detection_collate,
    )
    model = ArucoMobileNetV2(config["model"]["width_mult"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    true_positive = false_positive = false_negative = 0
    ranked: list[tuple[float, bool]] = []
    ground_truth_count = 0
    elapsed = 0.0
    prediction_rows: list[dict] = []
    with torch.inference_mode():
        warmup = torch.zeros(
            1,
            3,
            config["model"]["input_size"],
            config["model"]["input_size"],
            device=device,
        )
        for _ in range(5):
            model(warmup)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for images, targets in loader:
            images = images.to(device)
            start = time.perf_counter()
            raw = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            image_elapsed = time.perf_counter() - start
            elapsed += image_elapsed
            decoded = decode_predictions(
                raw,
                config["model"]["score_threshold"],
                config["model"]["nms_iou_threshold"],
            )
            tp, fp, fn = detection_counts(decoded, targets)
            true_positive += tp
            false_positive += fp
            false_negative += fn
            ap_decoded = decode_predictions(
                raw, 1e-4, config["model"]["nms_iou_threshold"]
            )
            batch_ranked, batch_gt = ranked_detections(ap_decoded, targets)
            ranked.extend(batch_ranked)
            ground_truth_count += batch_gt
            image_id = int(targets[0]["image_id"])
            prediction_rows.append(
                {
                    "image": str(dataset.images[image_id]),
                    "ground_truth_boxes": len(targets[0]["boxes_yolo"]),
                    "predicted_boxes": len(decoded[0]["boxes"]),
                    "max_score": float(ap_decoded[0]["scores"].max()) if len(ap_decoded[0]["scores"]) else 0.0,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "latency_ms": image_elapsed * 1000,
                }
            )

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    result = {
        "images": len(dataset),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision50": precision,
        "recall50": recall,
        "f1_50": 2 * precision * recall / max(precision + recall, 1e-9),
        "ap50": average_precision(ranked, ground_truth_count),
        "model_latency_ms": 1000 * elapsed / max(len(dataset), 1),
        "model_fps": len(dataset) / max(elapsed, 1e-9),
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "split": args.split,
        "manifest": str(Path(dataset_config["root"]) / dataset_config[f"{args.split}_manifest"]),
        "manifest_sha256": sha256(Path(dataset_config["root"]) / dataset_config[f"{args.split}_manifest"]),
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.predictions_csv:
        output = Path(args.predictions_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    if args.pr_curve_csv:
        output = Path(args.pr_curve_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
        cumulative_tp = cumulative_fp = 0
        rows = []
        for rank, (score, is_tp) in enumerate(ordered, start=1):
            cumulative_tp += int(is_tp)
            cumulative_fp += int(not is_tp)
            rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "is_true_positive": int(is_tp),
                    "cumulative_tp": cumulative_tp,
                    "cumulative_fp": cumulative_fp,
                    "precision": cumulative_tp / max(cumulative_tp + cumulative_fp, 1),
                    "recall": cumulative_tp / max(ground_truth_count, 1),
                }
            )
        with output.open("w", newline="", encoding="utf-8") as stream:
            fields = ["rank", "score", "is_true_positive", "cumulative_tp", "cumulative_fp", "precision", "recall"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
