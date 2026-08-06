# Hướng dẫn sử dụng project ArUco robot PoC

## 1. Yêu cầu hiện tại

### Phần cứng

- Laptop Windows/Linux;
- USB webcam hoặc video file;
- marker `DICT_4X4_50` được in;
- chưa cần FPGA board.

### Phần mềm

- Python 3.10+;
- PyTorch/torchvision;
- OpenCV contrib có `cv2.aruco`;
- NumPy, PyYAML;
- ONNX/ONNXScript nếu export.

Cài dependency:

```powershell
python -m pip install -r model/requirements.txt
```

Kiểm tra:

```powershell
python -c "import torch, cv2; print(torch.__version__, cv2.__version__, hasattr(cv2,'aruco'))"
```

## 2. Demo nhanh không cần model

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical --target-id 0
```

Controls:

- `Q`/`Esc`: thoát;
- `R`: reset STOP latch;
- `S`: lưu snapshot.

Output CSV mặc định:

```text
artifacts/webcam_poc/metrics.csv
```

## 3. Chọn camera/video

```powershell
# Camera 0
python poc/live_webcam_demo.py --source 0 --mode classical --target-id 0

# Camera 1
python poc/live_webcam_demo.py --source 1 --mode classical --target-id 0

# Video
python poc/live_webcam_demo.py --source path/to/video.avi `
  --mode classical --target-id 0
```

Camera settings:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --width 640 --height 480 --camera-fps 30
```

## 4. Điều chỉnh dừng bằng pixel proxy

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --slow-side-px 100 --stop-side-px 180 `
  --confirm-frames 3
```

Marker phải lớn dần khi robot tiến lại gần. Stop chỉ latch sau ba frame gần liên
tiếp.

## 5. Calibrate khoảng cách gần đúng

Đặt marker 10 cm cách camera 1 m:

```powershell
python poc/calibrate_focal_length.py --source 0 --marker-id 0 `
  --marker-size-m 0.10 --distance-m 1.0
```

Dùng focal length nhận được:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --marker-size-m 0.10 --focal-length-px 615.2 `
  --slow-distance-m 0.90 --stop-distance-m 0.45
```

Đây là pinhole approximation. Thực nghiệm cuối cần camera matrix/distortion và
pose estimation đầy đủ.

Hiệu chuẩn intrinsic và dùng solvePnP:

```powershell
python poc/calibrate_camera.py --source 0 `
  --board-cols 9 --board-rows 6 --square-size-m 0.025 `
  --samples 20 --output artifacts/calibration/camera.npz

python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --marker-size-m 0.10 `
  --camera-calibration artifacts/calibration/camera.npz
```

## 6. Tạo dataset

### Synthetic smoke dataset

```powershell
python model/scripts/generate_synthetic_aruco.py `
  --output dataset/aruco --count 1000
```

### Real camera session

```powershell
python model/scripts/collect_aruco_dataset.py --source 0 `
  --output dataset/aruco --session room_a_daylight `
  --every 10 --max-saved 500 --include-negatives --display
```

Review labels và tạo `train.txt`, `val.txt`, `test.txt` từ toàn bộ session.

```powershell
python model/scripts/build_session_splits.py --root dataset/aruco --force
python model/scripts/audit_aruco_dataset.py --root dataset/aruco `
  --output artifacts/dataset_audit --hash-images
```

## 7. Train model

```powershell
python model/train.py --config model/configs/mobilenetv2_035.yaml
```

Override để smoke test:

```powershell
python model/train.py --config model/configs/mobilenetv2_035.yaml `
  --epochs 2 --batch-size 4 --output-dir artifacts/smoke_train
```

Resume:

```powershell
python model/train.py --config model/configs/mobilenetv2_035.yaml `
  --resume artifacts/aruco_mbv2_035/last.pt
```

Output chính:

```text
artifacts/aruco_mbv2_035/
├── best.pt
├── last.pt
├── history.jsonl
└── resolved_config.json
```

## 8. Evaluate

```powershell
python model/evaluate.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --split test `
  --output-json artifacts/evaluation/model_summary.json `
  --predictions-csv artifacts/evaluation/model_predictions.csv `
  --pr-curve-csv artifacts/evaluation/pr_curve.csv
```

Không báo accuracy nếu checkpoint chỉ train synthetic smoke data.

## 9. CNN/Hybrid webcam mode

```powershell
python poc/live_webcam_demo.py --source 0 --mode cnn `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

```powershell
python poc/live_webcam_demo.py --source 0 --mode hybrid `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

Nếu checkpoint thiếu, chương trình liệt kê checkpoint local. Dùng classical để
kiểm tra camera trước.

## 10. Export ONNX/INT8

```powershell
python model/export_int8.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --calibration-samples 100
```

Output:

```text
artifacts/aruco_mbv2_035/export/
├── aruco_mobilenetv2_035_fp32.onnx
├── aruco_mobilenetv2_035_weights_int8.npz
└── int8_manifest.json
```

## 11. Offline test không cần webcam

```powershell
python poc/make_demo_video.py --marker-id 0 `
  --output artifacts/webcam_poc/aruco_approach.avi
```

```powershell
python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/aruco_approach.avi `
  --mode classical --target-id 0 --headless `
  --output-video artifacts/webcam_poc/result.avi
```

## 12. Chạy tests

```powershell
python -m unittest discover -s model/tests -v
python -m unittest discover -s poc/tests -v
```

### 12.1. Workflow khứ hồi bằng marker xuất phát

PoC hỗ trợ nhiệm vụ `start -> target -> start` bằng hai marker khác ID. Demo không
cần webcam/robot:

```powershell
python poc/make_roundtrip_demo_video.py --start-id 0 --target-id 1 `
  --output artifacts/webcam_poc/roundtrip_input.avi

python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/roundtrip_input.avi `
  --mode classical --mission roundtrip --start-id 0 --target-id 1 `
  --headless --output-video artifacts/webcam_poc/roundtrip_result.avi `
  --csv artifacts/webcam_poc/roundtrip_metrics.csv
```

Với webcam, đổi source thành `0` và bỏ `--headless`. State kết thúc hợp lệ là
`HOME_COMPLETE`. `--target-dwell-frames` đặt thời gian dừng tại phòng;
`--turn-frames` và `--turn-angular-speed` mô phỏng thao tác quay đầu. Xem workflow,
cách bố trí marker và chỉ tiêu đo tại [Round-trip demo](../poc/ROUNDTRIP_DEMO.md).

## 13. Troubleshooting

### Checkpoint không tồn tại

Chạy classical, train model, hoặc chỉ định checkpoint thật sự có trong
`artifacts/`.

### Không mở được camera

- thử `--source 1`;
- đóng ứng dụng khác đang dùng webcam;
- kiểm tra quyền camera của hệ điều hành;
- thử video file để tách lỗi camera khỏi pipeline.

### Không thấy marker

- xác nhận `DICT_4X4_50`;
- tăng kích thước marker/ánh sáng;
- giữ đủ viền trắng;
- thử classical mode;
- target ID phải đúng.

### Robot không STOP

- xem `side_px`/`distance_m` trong CSV;
- giảm `stop-side-px` hoặc tăng `stop-distance-m`;
- kiểm tra `confirm-frames`;
- chỉ target ID mới được phép dừng.

### Hybrid luôn fallback

Checkpoint không generalize với camera thật. Thu thập/review real dataset rồi
train lại; không giảm threshold mù quáng.

## 14. FPGA status

Bitstream cat/dog legacy đã được loại khỏi working tree và không dùng cho model
ArUco. PYNQ deployment cho model mới chỉ bắt đầu sau integer reference và HLS
accelerator.

## 15. Xuất bảng cho khóa luận

Sau khi chạy Classical/CNN/Hybrid và tạo ba CSV:

```powershell
python analysis/export_thesis_tables.py `
  --run classical=artifacts/runs/classical.csv `
  --run cnn=artifacts/runs/cnn.csv `
  --run hybrid=artifacts/runs/hybrid.csv `
  --drop-warmup 5 --output artifacts/thesis_tables
```

Xem [tài liệu Methods/Results](thesis_poc_methods_results.md) để biết bảng, hình,
metric và nội dung có thể đưa vào khóa luận.
