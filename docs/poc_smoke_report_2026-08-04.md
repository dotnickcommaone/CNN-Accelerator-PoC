# Báo cáo smoke test PoC — 2026-08-04

> **Không dùng như kết quả thực nghiệm chính thức.** Dataset và video đều được
> sinh tổng hợp; báo cáo chỉ chứng minh pipeline và cách xuất số liệu hoạt động.

## Môi trường

| Thành phần | Phiên bản |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.12.1+cpu |
| torchvision | 0.27.1+cpu |
| OpenCV | 4.13.0 |
| NumPy | 2.2.6 |
| Inference device | CPU |

## Dataset audit

Artifact: `artifacts/dataset_audit_current/`.

| Metric | Giá trị |
|---|---:|
| Images | 1000 |
| Bounding boxes | 847 |
| Negative images | 153 |
| Train/Validation/Test | 800/100/100 |
| Split overlap | 0 |
| Missing manifest images | 0 |
| Label errors | 0 |
| Median normalized box area | 0.2704 |
| Area P05–P95 | 0.0617–0.6026 |

## Synthetic model evaluation

Checkpoint: `artifacts/learnability_check_v2/best.pt`.

Artifact: `artifacts/evaluation_smoke/`.

| Metric | Giá trị |
|---|---:|
| Test images | 100 |
| TP / FP / FN | 49 / 86 / 38 |
| Precision50 | 0.3630 |
| Recall50 | 0.5632 |
| F1-50 | 0.4414 |
| AP50 | 0.3997 |
| Model-only latency | 14.260 ms |
| Model-only FPS | 70.126 |

Checkpoint chỉ được train trên synthetic learnability data nhỏ. Kết quả cho
thấy pipeline generalize một phần sang synthetic seed khác nhưng chưa đủ để
đánh giá camera thật.

## Offline approach-video benchmark

Input: `artifacts/webcam_poc/aruco_approach.avi`, 80 frame. Warm-up bị loại: 5
frame. Marker ID 3, slow threshold 90 px, stop threshold 170 px, confirm 3
frame. Artifact: `artifacts/thesis_tables_smoke/`.

| Backend | Vision mean±SD ms | Median | P95 | Total mean ms | Target rate | First STOP |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 2.912±1.089 | 2.629 | 5.156 | 4.391 | 1.00 | frame 52 |
| CNN | 18.555±3.217 | 18.674 | 23.996 | 20.046 | 0.76 | frame 55 |
| Hybrid | 19.612±3.008 | 19.173 | 24.998 | 21.118 | 1.00 | frame 52 |

Hybrid source rate:

```text
CNN + OpenCV ROI: 0.76
OpenCV fallback:  0.24
```

FPS trong log video headless không phải webcam FPS do video được đọc nhanh nhất
có thể. Latency compute vẫn hữu ích cho smoke comparison trên cùng máy/run.

## Cách tái tạo

```powershell
python model/scripts/audit_aruco_dataset.py --root dataset/aruco `
  --output artifacts/dataset_audit_current

python model/evaluate.py --config model/configs/mobilenetv2_035.yaml `
  --dataset-root dataset/aruco `
  --checkpoint artifacts/learnability_check_v2/best.pt --split test `
  --output-json artifacts/evaluation_smoke/model_summary.json `
  --predictions-csv artifacts/evaluation_smoke/model_predictions.csv `
  --pr-curve-csv artifacts/evaluation_smoke/pr_curve.csv

python analysis/export_thesis_tables.py `
  --run classical=artifacts/thesis_runs/classical.csv `
  --run cnn=artifacts/thesis_runs/cnn.csv `
  --run hybrid=artifacts/thesis_runs/hybrid.csv `
  --drop-warmup 5 --output artifacts/thesis_tables_smoke
```

## Thay thế trước khi đưa vào chương kết quả

1. Dùng real-camera sessions và split theo session.
2. Train checkpoint chính thức.
3. Dùng USB webcam/video test quay thật.
4. Ghi rõ CPU/webcam/resolution/lighting.
5. Chạy nhiều trial độc lập.
6. Bổ sung calibration và actual stop-distance measurement.
