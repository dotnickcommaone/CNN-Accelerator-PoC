from __future__ import annotations

import argparse
import csv
import hashlib
import json
import operator
import platform
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.ao.nn.quantized import Conv2d as QuantizedConv2d
from torch.ao.nn.quantized import Linear as QuantizedLinear
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aruco_detector.config import load_config, seed_everything
from aruco_detector.dataset import ArucoYoloDataset, detection_collate
from aruco_detector.metrics import average_precision, detection_counts, ranked_detections
from aruco_detector.network import ArucoMobileNetV2, decode_predictions


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def select_quantized_engine(requested: str) -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if requested != "auto":
        if requested not in supported:
            raise RuntimeError(
                f"Quantized engine '{requested}' is unavailable; supported={supported}"
            )
        return requested
    for candidate in ("onednn", "x86", "fbgemm", "qnnpack"):
        if candidate in supported:
            return candidate
    raise RuntimeError(f"No supported static INT8 engine was found: {supported}")


def calibrate_prepared_model(
    prepared: nn.Module,
    loader: DataLoader,
    max_samples: int,
) -> int:
    seen = 0
    with torch.inference_mode():
        for images, _targets in loader:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            images = images[:remaining]
            prepared(images)
            seen += len(images)
    return seen


def audit_quantized_graph(
    model: nn.Module,
    original_conv_count: int,
    original_linear_count: int,
    original_residual_add_count: int = 0,
) -> dict[str, Any]:
    quantized_conv_names = [
        name for name, module in model.named_modules() if isinstance(module, QuantizedConv2d)
    ]
    quantized_linear_names = [
        name for name, module in model.named_modules() if isinstance(module, QuantizedLinear)
    ]
    float_conv_names = [
        name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)
    ]
    float_linear_names = [
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    non_int8_weight_names = []
    for name, module in model.named_modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            if module.weight().dtype != torch.qint8:
                non_int8_weight_names.append(name)

    graph = getattr(model, "graph", None)
    quantized_add_nodes = []
    float_add_nodes = []
    input_quantize_nodes = []
    output_dequantize_nodes = []
    if graph is not None:
        for node in graph.nodes:
            target_text = str(node.target)
            if node.op == "call_function" and "quantized.add" in target_text:
                quantized_add_nodes.append(node.name)
            if node.op == "call_function" and node.target in {operator.add, torch.add}:
                float_add_nodes.append(node.name)
            if node.op == "call_function" and "quantize_per_tensor" in target_text:
                input_quantize_nodes.append(node.name)
            if node.op == "call_method" and node.target == "dequantize":
                output_dequantize_nodes.append(node.name)

    strict_integer_core = (
        len(quantized_conv_names) == original_conv_count
        and len(quantized_linear_names) == original_linear_count
        and not float_conv_names
        and not float_linear_names
        and not non_int8_weight_names
        and not float_add_nodes
        and len(quantized_add_nodes) == original_residual_add_count
        and len(input_quantize_nodes) == 1
        and len(output_dequantize_nodes) == 1
    )
    return {
        "strict_integer_core": strict_integer_core,
        "original_conv_count": original_conv_count,
        "quantized_conv_count": len(quantized_conv_names),
        "original_linear_count": original_linear_count,
        "quantized_linear_count": len(quantized_linear_names),
        "original_residual_add_count": original_residual_add_count,
        "quantized_residual_add_count": len(quantized_add_nodes),
        "input_quantize_node_count": len(input_quantize_nodes),
        "output_dequantize_node_count": len(output_dequantize_nodes),
        "float_conv_names": float_conv_names,
        "float_linear_names": float_linear_names,
        "float_add_nodes": float_add_nodes,
        "non_int8_weight_names": non_int8_weight_names,
        "quantized_conv_names": quantized_conv_names,
        "quantized_linear_names": quantized_linear_names,
    }


def build_int8_runtime(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    calibration_loader: DataLoader,
    calibration_samples: int,
    engine: str,
) -> tuple[nn.Module, dict[str, Any], int]:
    torch.backends.quantized.engine = engine
    model = ArucoMobileNetV2(config["model"]["width_mult"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    original_conv_count = sum(isinstance(module, nn.Conv2d) for module in model.modules())
    original_linear_count = sum(isinstance(module, nn.Linear) for module in model.modules())
    original_residual_add_count = sum(
        bool(getattr(module, "use_res_connect", False)) for module in model.modules()
    )
    example = torch.zeros(1, 3, config["model"]["input_size"], config["model"]["input_size"])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.ao\.quantization is deprecated.*",
            category=DeprecationWarning,
        )
        prepared = prepare_fx(
            model,
            get_default_qconfig_mapping(engine),
            (example,),
        )
    calibrated_samples = calibrate_prepared_model(
        prepared, calibration_loader, calibration_samples
    )
    if calibrated_samples <= 0:
        raise RuntimeError("INT8 calibration did not process any image")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.ao\.quantization is deprecated.*",
            category=DeprecationWarning,
        )
        quantized = convert_fx(prepared)
    quantized.eval()
    audit = audit_quantized_graph(
        quantized,
        original_conv_count=original_conv_count,
        original_linear_count=original_linear_count,
        original_residual_add_count=original_residual_add_count,
    )
    if not audit["strict_integer_core"]:
        raise RuntimeError(
            "Converted model contains a floating-point core fallback: "
            + json.dumps(audit, indent=2)
        )
    return quantized, audit, calibrated_samples


def write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_pr_curve(
    path: Path,
    ranked: list[tuple[float, bool]],
    ground_truth_count: int,
) -> None:
    ordered = sorted(ranked, key=lambda item: item[0], reverse=True)
    cumulative_tp = 0
    cumulative_fp = 0
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "rank",
            "score",
            "is_true_positive",
            "cumulative_tp",
            "cumulative_fp",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a PyTorch FX static-INT8 ArUco runtime on CPU"
    )
    parser.add_argument("--config", default="model/configs/mobilenetv2_035.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--calibration-manifest", default=None)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="PyTorch CPU threads; 0 keeps the environment default",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/evaluation/int8_runtime"
    )
    parser.add_argument(
        "--save-runtime",
        action="store_true",
        help="Save the converted runtime as int8_runtime.ts",
    )
    args = parser.parse_args()

    if args.calibration_samples <= 0:
        parser.error("--calibration-samples must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.threads < 0:
        parser.error("--threads must be non-negative")
    if args.threads:
        torch.set_num_threads(args.threads)

    config = load_config(args.config)
    seed_everything(config["seed"])
    if args.dataset_root is not None:
        config["dataset"]["root"] = args.dataset_root
    dataset_config = config["dataset"]
    calibration_manifest = (
        args.calibration_manifest or dataset_config["val_manifest"]
    )
    calibration_dataset = ArucoYoloDataset(
        dataset_config["root"],
        calibration_manifest,
        config["model"]["input_size"],
        augment=False,
        max_boxes=dataset_config["max_boxes"],
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=min(16, args.calibration_samples),
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        collate_fn=detection_collate,
    )
    test_dataset = ArucoYoloDataset(
        dataset_config["root"],
        dataset_config[f"{args.split}_manifest"],
        config["model"]["input_size"],
        augment=False,
        max_boxes=dataset_config["max_boxes"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=dataset_config["num_workers"],
        collate_fn=detection_collate,
    )

    engine = select_quantized_engine(args.engine)
    runtime, audit, calibrated_samples = build_int8_runtime(
        config=config,
        checkpoint_path=args.checkpoint,
        calibration_loader=calibration_loader,
        calibration_samples=args.calibration_samples,
        engine=engine,
    )

    warmup = torch.zeros(
        1, 3, config["model"]["input_size"], config["model"]["input_size"]
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            runtime(warmup)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    ranked: list[tuple[float, bool]] = []
    ground_truth_count = 0
    latency_samples: list[float] = []
    prediction_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            for images, targets in test_loader:
                start = time.perf_counter()
                raw = runtime(images)
                latency_ms = (time.perf_counter() - start) * 1000.0
                latency_samples.append(latency_ms)
                if repeat != 0:
                    continue
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
                        "image": str(test_dataset.images[image_id]),
                        "ground_truth_boxes": len(targets[0]["boxes_yolo"]),
                        "predicted_boxes": len(decoded[0]["boxes"]),
                        "max_score": float(ap_decoded[0]["scores"].max())
                        if len(ap_decoded[0]["scores"])
                        else 0.0,
                        "true_positive": tp,
                        "false_positive": fp,
                        "false_negative": fn,
                        "latency_ms": latency_ms,
                    }
                )

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    latency_mean = statistics.fmean(latency_samples)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_manifest_path = (
        Path(dataset_config["root"]) / dataset_config[f"{args.split}_manifest"]
    )
    calibration_manifest_path = Path(dataset_config["root"]) / calibration_manifest

    runtime_path = None
    runtime_trace_max_abs_error = None
    if args.save_runtime:
        runtime_path = output_dir / "int8_runtime.ts"
        with torch.inference_mode():
            scripted = torch.jit.trace(runtime, warmup, strict=False)
            expected = runtime(warmup)
            traced = scripted(warmup)
        runtime_trace_max_abs_error = float((expected - traced).abs().max())
        if runtime_trace_max_abs_error > 1e-6:
            raise RuntimeError(
                "Traced INT8 runtime changed the output by "
                f"{runtime_trace_max_abs_error}"
            )
        torch.jit.save(scripted, runtime_path)

    result = {
        "backend": "pytorch_fx_static_int8",
        "engine": engine,
        "arithmetic": "quint8_activation_qint8_weight_int32_accumulator",
        "scope": "INT8 CNN core with float output decode and NMS",
        "strict_integer_core": audit["strict_integer_core"],
        "images": len(test_dataset),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision50": precision,
        "recall50": recall,
        "f1_50": 2 * precision * recall / max(precision + recall, 1e-9),
        "ap50": average_precision(ranked, ground_truth_count),
        "model_latency_ms_mean": latency_mean,
        "model_latency_ms_median": statistics.median(latency_samples),
        "model_latency_ms_p95": percentile(latency_samples, 0.95),
        "model_latency_ms_min": min(latency_samples),
        "model_latency_ms_max": max(latency_samples),
        "model_fps_from_mean_latency": 1000.0 / max(latency_mean, 1e-9),
        "latency_samples": len(latency_samples),
        "warmup_iterations": args.warmup,
        "repeats": args.repeats,
        "batch_size": 1,
        "torch_num_threads": torch.get_num_threads(),
        "calibration": {
            "manifest": str(calibration_manifest_path),
            "requested_samples": args.calibration_samples,
            "processed_samples": calibrated_samples,
            "manifest_sha256": sha256(calibration_manifest_path),
        },
        "audit": audit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "split": args.split,
        "manifest": str(test_manifest_path),
        "manifest_sha256": sha256(test_manifest_path),
        "runtime_file": str(runtime_path) if runtime_path else None,
        "runtime_sha256": sha256(runtime_path) if runtime_path else None,
        "runtime_trace_max_abs_error": runtime_trace_max_abs_error,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "supported_quantized_engines": list(
                torch.backends.quantized.supported_engines
            ),
        },
        "warning": (
            "This is a CPU INT8 PoC runtime. Its latency is not FPGA latency. "
            "The CNN core is quantized; output decode and NMS remain float."
        ),
    }

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    pr_curve_path = output_dir / "pr_curve.csv"
    latency_path = output_dir / "latency_samples.csv"
    audit_path = output_dir / "quantization_audit.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_predictions(predictions_path, prediction_rows)
    write_pr_curve(pr_curve_path, ranked, ground_truth_count)
    with latency_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample", "latency_ms"])
        writer.writeheader()
        writer.writerows(
            {"sample": index, "latency_ms": value}
            for index, value in enumerate(latency_samples)
        )

    print(json.dumps(result, indent=2))
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"PR curve: {pr_curve_path}")
    print(f"Latency samples: {latency_path}")
    print(f"Quantization audit: {audit_path}")


if __name__ == "__main__":
    main()
