from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aruco_detector.config import load_config
from aruco_detector.dataset import ArucoYoloDataset, detection_collate
from aruco_detector.network import ArucoMobileNetV2


def quantize_per_output_channel(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    array = weight.detach().cpu().float().numpy()
    axes = tuple(range(1, array.ndim))
    maximum = np.max(np.abs(array), axis=axes, keepdims=True)
    scale = np.maximum(maximum / 127.0, 1e-12)
    quantized = np.clip(np.round(array / scale), -127, 127).astype(np.int8)
    return quantized, scale.reshape(-1).astype(np.float32)


def fold_batch_norms(module: nn.Module) -> nn.Module:
    """Return an eval-mode copy with adjacent Conv2d+BatchNorm2d pairs folded."""
    folded = copy.deepcopy(module).eval()

    def recurse(parent: nn.Module) -> None:
        for child in parent.children():
            recurse(child)
        names = list(parent._modules)
        for index in range(len(names) - 1):
            first_name, second_name = names[index], names[index + 1]
            first = parent._modules[first_name]
            second = parent._modules[second_name]
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d):
                parent._modules[first_name] = fuse_conv_bn_eval(first, second)
                parent._modules[second_name] = nn.Identity()

    recurse(folded)
    return folded


def calibrate_activations(
    model: nn.Module, loader: DataLoader, max_samples: int
) -> dict[str, dict[str, float]]:
    ranges: dict[str, dict[str, float]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor):
            entry = ranges.setdefault(name, {"input_abs_max": 0.0, "output_abs_max": 0.0})
            entry["input_abs_max"] = max(
                entry["input_abs_max"], float(inputs[0].detach().abs().max())
            )
            entry["output_abs_max"] = max(
                entry["output_abs_max"], float(output.detach().abs().max())
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
    seen = 0
    with torch.inference_mode():
        for images, _ in loader:
            model(images)
            seen += len(images)
            if seen >= max_samples:
                break
    for hook in hooks:
        hook.remove()
    for entry in ranges.values():
        entry["input_scale"] = max(entry["input_abs_max"] / 127.0, 1e-12)
        entry["output_scale"] = max(entry["output_abs_max"] / 127.0, 1e-12)
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model and INT8 FPGA weights")
    parser.add_argument("--config", default="model/configs/mobilenetv2_035.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--calibration-manifest", default=None)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.dataset_root is not None:
        config["dataset"]["root"] = args.dataset_root
    output_dir = Path(args.output_dir or config["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model = ArucoMobileNetV2(config["model"]["width_mult"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    example_input = torch.zeros(
        1, 3, config["model"]["input_size"], config["model"]["input_size"]
    )
    with torch.inference_mode():
        reference = model(example_input)
    model = fold_batch_norms(model)
    with torch.inference_mode():
        folded_output = model(example_input)
    fold_max_error = float((reference - folded_output).abs().max())
    if fold_max_error > 1e-4:
        raise RuntimeError(f"Conv-BN folding changed output by {fold_max_error}")

    calibration_ranges: dict[str, dict[str, float]] = {}
    calibration_manifest = (
        args.calibration_manifest or config["dataset"].get("val_manifest")
    )
    if calibration_manifest and args.calibration_samples > 0:
        calibration_dataset = ArucoYoloDataset(
            config["dataset"]["root"],
            calibration_manifest,
            config["model"]["input_size"],
            augment=False,
            max_boxes=config["dataset"]["max_boxes"],
        )
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=min(16, args.calibration_samples),
            shuffle=False,
            num_workers=config["dataset"]["num_workers"],
            collate_fn=detection_collate,
        )
        calibration_ranges = calibrate_activations(
            model, calibration_loader, args.calibration_samples
        )

    onnx_path = output_dir / "aruco_mobilenetv2_035_fp32.onnx"
    try:
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            input_names=["image"],
            output_names=["grid_prediction"],
            opset_version=config["export"]["onnx_opset"],
            dynamic_axes={"image": {0: "batch"}, "grid_prediction": {0: "batch"}},
            dynamo=False,
        )
        onnx_status = str(onnx_path)
    except (ImportError, ModuleNotFoundError, torch.onnx.OnnxExporterError) as error:
        onnx_status = f"not exported: {error}"

    arrays: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        quantized, scales = quantize_per_output_channel(module.weight)
        key = name.replace(".", "_")
        arrays[f"{key}_weight_int8"] = quantized
        arrays[f"{key}_weight_scale"] = scales
        activation = calibration_ranges.get(name)
        if module.bias is not None:
            arrays[f"{key}_bias_fp32"] = module.bias.detach().cpu().numpy()
            if activation:
                denominator = activation["input_scale"] * scales
                arrays[f"{key}_bias_int32"] = np.round(
                    module.bias.detach().cpu().numpy() / denominator
                ).astype(np.int32)
        manifest[name] = {
            "type": module.__class__.__name__,
            "weight_shape": list(module.weight.shape),
            "stride": list(module.stride) if isinstance(module, nn.Conv2d) else None,
            "padding": list(module.padding) if isinstance(module, nn.Conv2d) else None,
            "groups": module.groups if isinstance(module, nn.Conv2d) else None,
            "quantization": "symmetric_int8_per_output_channel",
            "activation_calibration": activation,
        }
    np.savez_compressed(output_dir / "aruco_mobilenetv2_035_weights_int8.npz", **arrays)
    metadata = {
        "input_shape": [1, 3, config["model"]["input_size"], config["model"]["input_size"]],
        "input_range": [0.0, 1.0],
        "output_channels": ["objectness", "center_x", "center_y", "width", "height"],
        "onnx": onnx_status,
        "conv_bn_fold_max_abs_error": fold_max_error,
        "calibration": {
            "enabled": bool(calibration_ranges),
            "manifest": str(calibration_manifest) if calibration_ranges else None,
            "requested_samples": args.calibration_samples,
            "method": "symmetric_absmax",
        },
        "layers": manifest,
        "warning": (
            "Weights and calibrated activation scales are exported for HLS design. "
            "A bit-accurate integer reference and accuracy comparison are still "
            "required before generating the final accelerator."
        ),
    }
    (output_dir / "int8_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"INT8 weights: {output_dir / 'aruco_mobilenetv2_035_weights_int8.npz'}")
    print(f"ONNX: {onnx_status}")


if __name__ == "__main__":
    main()
