from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aruco_detector.config import load_config, seed_everything, select_device
from aruco_detector.dataset import ArucoYoloDataset, detection_collate
from aruco_detector.loss import DetectionLoss
from aruco_detector.metrics import average_precision, detection_counts, ranked_detections
from aruco_detector.network import ArucoMobileNetV2, decode_predictions


def make_loader(
    config: dict, manifest_key: str, augment: bool, shuffle: bool
) -> DataLoader:
    dataset_config = config["dataset"]
    training_config = config["training"]
    dataset = ArucoYoloDataset(
        root=dataset_config["root"],
        manifest=dataset_config[manifest_key],
        input_size=config["model"]["input_size"],
        augment=augment,
        max_boxes=dataset_config["max_boxes"],
    )
    return DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=shuffle,
        num_workers=dataset_config["num_workers"],
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: ArucoMobileNetV2,
    loader: DataLoader,
    criterion: DetectionLoss,
    device: torch.device,
    optimizer: AdamW | None,
    config: dict,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "object_loss": 0.0, "box_loss": 0.0}
    true_positive = false_positive = false_negative = 0
    ranked: list[tuple[float, bool]] = []
    ground_truth_count = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            predictions = model(images)
            loss, parts = criterion(predictions, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for key in totals:
                totals[key] += parts[key] * len(images)
            if not training:
                decoded = decode_predictions(
                    predictions,
                    config["model"]["score_threshold"],
                    config["model"]["nms_iou_threshold"],
                )
                tp, fp, fn = detection_counts(decoded, targets)
                true_positive += tp
                false_positive += fp
                false_negative += fn
                ap_decoded = decode_predictions(
                    predictions,
                    1e-4,
                    config["model"]["nms_iou_threshold"],
                )
                batch_ranked, batch_gt = ranked_detections(ap_decoded, targets)
                ranked.extend(batch_ranked)
                ground_truth_count += batch_gt

    sample_count = max(len(loader.dataset), 1)
    metrics = {key: value / sample_count for key, value in totals.items()}
    if not training:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        metrics.update(
            {
                "precision50": precision,
                "recall50": recall,
                "f1_50": 2 * precision * recall / max(precision + recall, 1e-9),
                "ap50": average_precision(ranked, ground_truth_count),
            }
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ArUco marker detector")
    parser.add_argument(
        "--config", default="model/configs/mobilenetv2_035.yaml"
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.dataset_root is not None:
        config["dataset"]["root"] = args.dataset_root
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    seed_everything(config["seed"])
    device = select_device(config["device"])
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ArucoMobileNetV2(
        width_mult=config["model"]["width_mult"],
        pretrained=config["model"]["pretrained"],
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=max(config["training"]["epochs"], 1)
    )
    start_epoch = 0
    best_selection = (-1.0, -1.0, float("-inf"))
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        previous_metrics = checkpoint["metrics"]
        best_selection = tuple(
            checkpoint.get(
                "best_selection",
                (
                    previous_metrics.get("ap50", 0.0),
                    previous_metrics["f1_50"],
                    -previous_metrics["loss"],
                ),
            )
        )

    criterion = DetectionLoss(
        positive_weight=config["training"]["positive_weight"],
        box_weight=config["training"]["box_loss_weight"],
    )
    train_loader = make_loader(config, "train_manifest", augment=True, shuffle=True)
    val_loader = make_loader(config, "val_manifest", augment=False, shuffle=False)
    history_path = output_dir / "history.jsonl"
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"Device: {device}; train={len(train_loader.dataset)}; val={len(val_loader.dataset)}")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, config
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, None, config)
        selection = (
            val_metrics["ap50"],
            val_metrics["f1_50"],
            -val_metrics["loss"],
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        scheduler.step()
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "metrics": val_metrics,
            "best_selection": max(best_selection, selection),
        }
        torch.save(state, output_dir / "last.pt")
        if selection > best_selection:
            best_selection = selection
            torch.save(state, output_dir / "best.pt")
        print(
            f"Epoch {epoch + 1:03d}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"P50={val_metrics['precision50']:.3f} "
            f"R50={val_metrics['recall50']:.3f} F1={val_metrics['f1_50']:.3f} "
            f"AP50={val_metrics['ap50']:.3f}"
        )


if __name__ == "__main__":
    main()
