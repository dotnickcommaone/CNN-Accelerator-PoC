# ArUco marker detector — MobileNetV2-0.35

This directory contains the replacement model for the original cat/dog
classifier. The CNN detects the marker region; OpenCV ArUco is intentionally
kept as the post-processing stage that decodes the marker ID and estimates its
pose.

## Model contract

- Input: RGB image, `160x160`, normalized to `[0, 1]`
- Backbone: MobileNetV2 with `width_mult=0.35`
- Output: `5x5x5` tensor
  - channel 0: marker objectness logit
  - channels 1–2: center offset inside a grid cell
  - channels 3–4: normalized box width and height
- Classes: one (`aruco_marker`)
- Label format: YOLO text, one row per marker:
  `class_id center_x center_y width height`, normalized to `[0, 1]`

The first implementation targets INT8. INT4 will only be introduced after an
INT8 baseline has been measured, because it normally requires
quantization-aware training.

## Quick start

Generate a small synthetic dataset:

```powershell
python model/scripts/generate_synthetic_aruco.py --output dataset/aruco --count 1000
```

Train:

```powershell
python model/train.py --config model/configs/mobilenetv2_035.yaml
```

Evaluate:

```powershell
python model/evaluate.py --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt
```

Export ONNX and symmetric INT8 weights:

```powershell
python model/export_int8.py --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --calibration-samples 100
```

The exporter uses the validation manifest for activation calibration and emits
per-output-channel INT8 weight scales, INT32 biases, and per-layer activation
ranges. This is an HLS handoff artifact, not yet a bit-accurate quantized model.
Conv/BatchNorm pairs are folded before ONNX and INT8 export; the exporter aborts
if folding changes the FP32 output by more than `1e-4`.

## CPU INT8 runtime PoC

Evaluate a real static-INT8 CPU runtime and export AP50 plus latency statistics:

```powershell
python model/evaluate_int8_runtime.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --split test `
  --calibration-samples 100 `
  --warmup 20 `
  --repeats 5 `
  --threads 8 `
  --save-runtime `
  --output-dir artifacts/evaluation/int8_runtime
```

The command applies PyTorch FX post-training static quantization and refuses to
report metrics if a float `Conv2d`, `Linear`, or residual add remains in the CNN
core. On the current `onednn` backend, activations are QUINT8, weights are QINT8,
and convolution accumulation is INT32. The graph has one input quantize node and
one output dequantize node; bounding-box decode and NMS remain floating point.

Generated files:

- `metrics.json`: AP50, Precision, Recall, mean/median/P95 latency and provenance;
- `predictions.csv`: one row per test image;
- `pr_curve.csv`: raw Precision–Recall points;
- `latency_samples.csv`: all timed inference samples;
- `quantization_audit.json`: quantized-layer and fallback audit;
- `int8_runtime.ts`: optional fixed-input TorchScript runtime.

This runtime is the CPU INT8 PoC baseline. Its latency is not FPGA latency, and
its observer scales are not guaranteed to be bit-identical to the separate NPZ
HLS handoff. A bit-accurate HLS reference is still required before synthesis.

## Current baseline limits

- The supplied configuration trains from scratch because torchvision does not
  publish pretrained MobileNetV2-0.35 weights.
- The `5x5` detection grid is intended for a small number of room markers, not
  dense general-purpose object detection.
- A CPU static-INT8 runtime is available for PoC accuracy and latency. A
  bit-accurate reference using the exact HLS requantization rules remains required
  before hardware synthesis.
- Synthetic results are sanity checks only and must not be reported as thesis
  accuracy.

The synthetic dataset is useful for smoke tests, but thesis measurements must
use a held-out real-camera test set containing different distances, viewing
angles, lighting levels, blur, occlusion, and non-marker backgrounds.

Audit dataset and export raw statistics:

```powershell
python model/scripts/audit_aruco_dataset.py --root dataset/aruco `
  --output artifacts/dataset_audit --hash-images
```

For real data stored as `images/<session>/...`, build leakage-safe manifests:

```powershell
python model/scripts/build_session_splits.py --root dataset/aruco --force
```

## Collect real-camera data

Capture frames and create YOLO labels automatically from markers that OpenCV
can decode:

```powershell
python model/scripts/collect_aruco_dataset.py --source 0 `
  --output dataset/aruco --session room_a_daylight `
  --every 10 --max-saved 500 --include-negatives --display
```

`--source` may be a USB camera index, video path, or ESP32-CAM stream URL.
Each recording creates its own session manifest. Build `train.txt`, `val.txt`,
and `test.txt` from whole sessions; do not randomly distribute adjacent frames
from one video among multiple splits.

Auto-labeling only captures markers already detectable by classical OpenCV.
Manually review the labels and add difficult examples—blur, partial occlusion,
small markers and steep viewing angles—otherwise the CNN will not learn the
conditions in which it is intended to improve robustness.

## Image inference

After training, run the complete model-stage pipeline:

```powershell
python model/infer_aruco.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --image path/to/frame.jpg `
  --output artifacts/aruco_result.jpg
```

The command detects marker ROIs with the CNN, decodes IDs with OpenCV ArUco,
prints machine-readable JSON, and optionally saves an annotated image.
