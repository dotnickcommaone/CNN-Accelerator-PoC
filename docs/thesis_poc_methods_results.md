# Nội dung PoC đề xuất cho khóa luận

Tài liệu này cung cấp cấu trúc, đoạn mô tả và bảng biểu để đưa PoC vào khóa
luận trong giai đoạn chưa có FPGA board. Thay các ô `TBD` bằng số liệu được xuất
từ raw CSV; không dùng smoke data tổng hợp làm kết quả chính thức.

## 1. Phạm vi đóng góp trong giai đoạn PoC

Trong phạm vi PoC, hệ thống hoàn thiện chuỗi xử lý từ camera đến quyết định dừng
robot mà không phụ thuộc phần cứng FPGA. PoC đóng vai trò:

1. software reference cho accelerator tương lai;
2. công cụ thu thập và kiểm tra dataset;
3. baseline CPU cho Classical, CNN và Hybrid;
4. môi trường đánh giá latency, FPS, ArUco ID và stop controller;
5. nguồn test vector/metric để kiểm chứng FPGA sau này.

Phần chưa thực hiện gồm HLS accelerator, Vivado overlay, tài nguyên FPGA, power
trên board và điều khiển motor vật lý. Các nội dung này được trình bày như hướng
phát triển, không phải kết quả đã đạt.

## 2. Mô tả hệ thống có thể dùng trong chương thiết kế

> Hệ thống PoC sử dụng USB webcam để thu nhận khung hình BGR. Tùy chế độ hoạt
> động, marker được phát hiện trực tiếp bằng OpenCV ArUco, bởi MobileNetV2-0.35
> kết hợp giải mã ArUco trong ROI, hoặc bằng phương pháp hybrid có fallback.
> Marker có ID trùng với phòng đích được lựa chọn để ước lượng khoảng cách bằng
> mô hình pinhole hoặc solvePnP. Chuỗi khoảng cách được lọc median và đưa vào bộ
> điều khiển trạng thái SEARCHING–APPROACHING–SLOWING–STOPPED. Điều kiện dừng
> phải tồn tại liên tiếp trong nhiều frame nhằm hạn chế nhiễu và dừng giả.

Sơ đồ dùng trong khóa luận:

- [Draw.io, trang 01 PoC](diagrams/aruco_robot_system_architecture.drawio);
- [Draw.io, trang 02 FPGA target](diagrams/aruco_robot_system_architecture.drawio).

## 3. Thiết lập thực nghiệm cần ghi

### 3.1 Phần cứng/phần mềm

| Thuộc tính | Giá trị |
|---|---|
| Máy tính/CPU | TBD |
| GPU | TBD hoặc không sử dụng |
| RAM | TBD |
| Hệ điều hành | TBD |
| Python | Xuất bằng `python --version` |
| PyTorch/torchvision | Xuất bằng lệnh bên dưới |
| OpenCV | Xuất bằng lệnh bên dưới |
| Webcam | Hãng/model TBD |
| Resolution/FPS yêu cầu | Ví dụ 640×480 @ 30 FPS |
| Marker dictionary | `DICT_4X4_50` |
| Marker physical size | TBD m |
| Target ID | TBD |

Xuất version:

```powershell
python -c "import sys,torch,torchvision,cv2,numpy; print(sys.version); print('torch',torch.__version__); print('torchvision',torchvision.__version__); print('opencv',cv2.__version__); print('numpy',numpy.__version__)"
```

### 3.2 Camera calibration

Sử dụng checkerboard với số inner corner và square size đã biết:

```powershell
python poc/calibrate_camera.py --source 0 `
  --board-cols 9 --board-rows 6 --square-size-m 0.025 `
  --samples 20 --output artifacts/calibration/camera.npz
```

Báo cáo:

- số calibration views;
- image resolution;
- camera matrix;
- distortion coefficients;
- RMS reprojection error;
- mean per-view reprojection error.

Các giá trị được xuất tự động trong `camera.json`.

## 4. Dataset protocol

### 4.1 Thu thập

Mỗi điều kiện được ghi thành session độc lập:

```powershell
python model/scripts/collect_aruco_dataset.py --source 0 `
  --output dataset/aruco --session room_a_daylight `
  --every 10 --max-saved 500 --include-negatives --display
```

Session đề xuất:

- nhiều phòng/nền;
- daylight/low light/backlight;
- khoảng cách 0.3–3 m;
- góc 0°, 15°, 30°, 45°, 60°;
- đứng yên/chuyển động;
- blur/occlusion;
- negative sessions không có marker.

### 4.2 Chia tập

Chia nguyên session để tránh temporal leakage:

```powershell
python model/scripts/build_session_splits.py --root dataset/aruco `
  --train 0.70 --val 0.15 --test 0.15 --seed 42 --force
```

### 4.3 Audit và số liệu dataset

```powershell
python model/scripts/audit_aruco_dataset.py --root dataset/aruco `
  --output artifacts/dataset_audit --hash-images
```

Sử dụng:

- `summary.json`: tổng số ảnh/box/negative/error;
- `split_summary.csv`: số ảnh mỗi split;
- `boxes.csv`: phân bố width/height/area;
- `images.csv`: resolution, session và split;
- `errors.txt`: nhãn lỗi, overlap hoặc missing file.

### 4.4 Bảng dataset

| Split | Sessions | Images | Positive | Negative | Boxes |
|---|---:|---:|---:|---:|---:|
| Train | TBD | TBD | TBD | TBD | TBD |
| Validation | TBD | TBD | TBD | TBD | TBD |
| Test | TBD | TBD | TBD | TBD | TBD |

## 5. Mô hình và training

> MobileNetV2-0.35 nhận ảnh RGB 160×160 và sinh tensor 5×5×5 gồm objectness,
> offset tâm và kích thước bounding box. Mạng chỉ học lớp `aruco_marker`; ID
> được OpenCV giải mã ở bước sau. Loss là tổng của binary cross entropy cho
> objectness và Smooth-L1 cho box positive cells.

Train:

```powershell
python model/train.py --config model/configs/mobilenetv2_035.yaml
```

Lưu vào khóa luận:

- config YAML/resolved JSON;
- random seed;
- epoch/batch size/optimizer/lr/weight decay;
- best checkpoint selection rule;
- `history.jsonl` để vẽ train/validation loss theo epoch.

## 6. Đánh giá model và xuất raw data

```powershell
python model/evaluate.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --split test `
  --output-json artifacts/evaluation/model_summary.json `
  --predictions-csv artifacts/evaluation/model_predictions.csv `
  --pr-curve-csv artifacts/evaluation/pr_curve.csv
```

### Metric

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
FPS       = number_of_frames / elapsed_time
```

AP50 được tính từ prediction xếp theo confidence tại IoU threshold 0.5, không
bị cắt bởi operating score threshold.

### Bảng model

| Variant | Precision50 | Recall50 | F1-50 | AP50 | Latency ms | FPS |
|---|---:|---:|---:|---:|---:|---:|
| CNN FP32 CPU | TBD | TBD | TBD | TBD | TBD | TBD |
| CNN INT8 software | TBD | TBD | TBD | TBD | TBD | TBD |

## 7. Benchmark PoC

Chạy cùng một video/test sequence cho từng backend:

```powershell
python poc/live_webcam_demo.py --source test.avi --mode classical `
  --target-id 0 --headless --csv artifacts/runs/classical.csv

python poc/live_webcam_demo.py --source test.avi --mode cnn `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0 `
  --headless --csv artifacts/runs/cnn.csv

python poc/live_webcam_demo.py --source test.avi --mode hybrid `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0 `
  --headless --csv artifacts/runs/hybrid.csv
```

Sau đó:

```powershell
python analysis/export_thesis_tables.py `
  --run classical=artifacts/runs/classical.csv `
  --run cnn=artifacts/runs/cnn.csv `
  --run hybrid=artifacts/runs/hybrid.csv `
  --drop-warmup 5 --output artifacts/thesis_tables
```

Mỗi PoC CSV có sidecar `.meta.json` chứa command arguments, Python/library
versions, actual camera properties và SHA-256 checkpoint/video/calibration.
`analysis_manifest.json` tiếp tục lưu SHA-256 của các raw CSV. Nộp các file này
cùng bảng tổng hợp để bảo đảm khả năng tái tạo.

### Bảng hiệu năng

| Backend | Vision mean±SD ms | Median ms | P95 ms | End-to-end mean ms | FPS | Target rate |
|---|---:|---:|---:|---:|---:|---:|
| Classical | TBD | TBD | TBD | TBD | TBD | TBD |
| CNN | TBD | TBD | TBD | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD | TBD | TBD | TBD |

Hybrid cần báo thêm `cnn_source_rate` và `opencv_source_rate`; không được xem
fallback result là CNN detection.

## 8. Stop experiment

Mỗi trial phải có reset controller và tiến lại gần từ vị trí ban đầu tương tự.
Ground-truth stop distance được đo độc lập bằng thước hoặc mốc sàn.

| Trial | Condition | Target ID | Commanded stop m | Actual stop m | Absolute error cm | False stop |
|---:|---|---:|---:|---:|---:|---:|
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |

Tối thiểu báo mean, SD, median, P95 và max absolute error. `stop_trials.csv`
từ script phân tích cung cấp first-stop frame/distance; actual physical error
phải được bổ sung từ phép đo bên ngoài.

### 8.1. Thực nghiệm nhiệm vụ khứ hồi

Để đánh giá chức năng robot quay lại trạm, dùng marker start và target có ID khác
nhau. Một trial chỉ được tính thành công khi state đi qua `AT_TARGET` và kết thúc ở
`HOME_COMPLETE`; chỉ phát hiện lại start marker nhưng chưa vào state hoàn tất không
được tính thành công.

| Trial | Start ID | Target ID | Target frame | Home frame | Mission time s | Target stop error cm | Home stop error cm | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 1 | TBD | TBD | TBD | TBD | TBD | TBD |

Tối thiểu 10 trial cho mỗi điều kiện ánh sáng/backend. Báo tỷ lệ hoàn tất,
thời gian đi tới target, thời gian quay về, sai số dừng ở hai đầu và failure mode.
`first_target_frame`, `home_complete_frame` và `mission_completed` được xuất tự
động; sai số vật lý phải đo độc lập. Trong Prism, nhập mỗi trial thành một row,
backend/điều kiện thành column; dùng scatter có median/IQR hoặc mean/SD và giữ raw
points. Tỷ lệ thành công nên báo kèm tử số/mẫu số, không chỉ phần trăm.

## 9. Hình và biểu đồ đề xuất

1. System architecture PoC.
2. Target PS–PL architecture.
3. Dataset split và box-size distribution.
4. Train/validation loss theo epoch.
5. Precision–Recall curve.
6. Latency distribution Classical/CNN/Hybrid.
7. FPS distribution.
8. State timeline theo frame.
9. Stop-distance error qua nhiều trial.
10. Timeline sáu state của nhiệm vụ khứ hồi.
11. Thời gian target/home và tỷ lệ mission complete qua các backend.
12. Ví dụ true positive, false positive, false negative và difficult cases.

Không chỉ dùng bar chart của giá trị mean; nên thể hiện raw points, distribution
hoặc error bars và ghi rõ `n`.

## 10. Statistical reporting

- latency/FPS: báo mean±SD và median/P95;
- accuracy: báo TP/FP/FN cùng tỷ lệ;
- repeated stop trials: mean±SD, median, min/max;
- paired backend comparison phải chạy trên cùng frame/trial;
- ghi warm-up removal và outlier rule trước khi phân tích;
- giữ raw CSV và SHA-256 manifest.

Nếu dùng kiểm định thống kê, lựa chọn test phải dựa vào thiết kế paired/unpaired
và phân phối dữ liệu; không chọn test chỉ dựa trên p-value mong muốn.

## 11. Đoạn thảo luận mẫu

> PoC chứng minh khả năng tách backend inference khỏi tầng ứng dụng. Classical
> mode cung cấp baseline hoạt động không cần checkpoint, trong khi CNN mode thể
> hiện pipeline dự kiến khi thay CPU inference bằng FPGA. Hybrid mode tăng độ
> bền demo nhưng fallback rate phải được báo cáo để tránh đánh giá quá cao đóng
> góp của CNN. Việc chưa có FPGA không ngăn cản kiểm chứng camera, dataset,
> ArUco, pose, state machine và phương pháp benchmark; tuy nhiên tài nguyên,
> power và speedup FPGA được để lại cho giai đoạn triển khai phần cứng.

## 12. Giới hạn cần ghi trung thực

- chưa có MobileNetV2 accelerator/board measurement;
- synthetic data chỉ dùng smoke test;
- checkpoint chính thức cần real-camera dataset;
- solvePnP phụ thuộc calibration và marker size chính xác;
- speed command hiện mô phỏng, chưa nối motor;
- video headless FPS không tương đương USB camera FPS;
- power laptop không thay thế power FPGA.

## 13. Artifact checklist khi nộp khóa luận

- source code commit hash;
- environment versions;
- config/resolved config;
- train/val/test manifests;
- dataset audit report;
- checkpoint hash;
- model summary/prediction/PR CSV;
- raw PoC logs;
- generated thesis tables và manifest;
- calibration NPZ/JSON;
- stop trial logs/videos;
- sau này: HLS/Vivado/resource/timing/power reports.
