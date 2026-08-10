# Số liệu cập nhật dùng cho khóa luận — 10/08/2026

## 1. Phạm vi và quy ước

File này tổng hợp các số liệu mới nhất đang có trong project. Kết quả được chia theo
nguồn để tránh trình bày dữ liệu tổng hợp như phép đo trên hệ thống thật.

| Nhãn | Ý nghĩa |
|---|---|
| **SYNTHETIC** | Đo trên bộ ảnh ArUco được sinh bằng chương trình; không đại diện cho độ chính xác trên camera thật. |
| **CAMERA POC** | Đo trực tiếp từ USB webcam và pipeline Python trên laptop. |
| **CALIBRATION** | Kết quả hiệu chuẩn camera riêng; phiên round-trip bên dưới chưa nạp file calibration này. |
| **DERIVED** | Giá trị được tính lại từ artifact gốc, không phải một phép đo độc lập. |
| **TBD** | Chưa có số liệu vì chưa triển khai FPGA hoặc robot vật lý. |

Không có số liệu dummy nào được dùng trong các bảng kết quả của file này.

## 2. Cấu hình mô hình và dataset

### 2.1 Mô hình

| Thuộc tính | Giá trị | Nguồn |
|---|---:|---|
| Backbone | MobileNetV2-0.35 | `resolved_config.json` |
| Input | RGB, 160 × 160 | `int8_manifest.json` |
| Output | Grid 5 × 5 × 5 | source model |
| Số lớp đối tượng | 1 — `aruco_marker` | config/source model |
| Số tham số | 452.761 | đếm trực tiếp bằng PyTorch |
| Số tham số trainable | 452.761 | đếm trực tiếp bằng PyTorch |
| Pretrained ImageNet | Không | `pretrained=false` |
| Seed | 42 | `resolved_config.json` |
| Batch size | 32 | `resolved_config.json` |
| Learning rate | 0,001 | `resolved_config.json` |
| Weight decay | 0,0001 | `resolved_config.json` |

### 2.2 Dataset **SYNTHETIC**

Dataset hiện tại gồm 1.000 ảnh synthetic và được chia 80/10/10.

| Split | Tổng ảnh | Ảnh có marker | Ảnh negative | Tỷ lệ |
|---|---:|---:|---:|---:|
| Train | 800 | 679 | 121 | 80% |
| Validation | 100 | 81 | 19 | 10% |
| Test | 100 | 87 | 13 | 10% |
| **Tổng** | **1.000** | **847** | **153** | **100%** |

Tỷ lệ toàn dataset là 84,7% ảnh có marker và 15,3% ảnh negative. Tên file đều có
dạng `synthetic_xxxxxx`; vì vậy các metric từ dataset này phải ghi rõ là kết quả
synthetic, không phải kết quả camera thực.

### 2.3 Session camera mới `room_a_daylight` **CAMERA POC**

Ngoài 1.000 ảnh synthetic, project đã có thêm một session webcam chưa được đưa vào
các split huấn luyện và đánh giá phía trên.

| Chỉ tiêu | Giá trị |
|---|---:|
| Số ảnh trong manifest | 68 |
| Ảnh positive | 12 |
| Ảnh negative | 56 |
| Tỷ lệ positive | 17,65% |
| Tỷ lệ negative | 82,35% |
| Label bị thiếu | 0 |

Session này có thể dùng làm nguồn dữ liệu camera thật cho lần chia split tiếp theo,
nhưng chưa đủ để báo AP50 camera thực. Khi bổ sung nhiều session, phải chia nguyên
session vào train/validation/test để tránh rò rỉ các frame liền kề.

## 3. Kết quả huấn luyện **SYNTHETIC**

File history chứa hai chu kỳ 60 epoch giống hệt nhau do lần chạy thứ hai được append
vào cùng file. Bảng dưới chỉ báo cáo một chu kỳ duy nhất, tránh ghi sai thành một lần
huấn luyện 120 epoch.

| Chỉ tiêu | Giá trị |
|---|---:|
| Số epoch mỗi lần chạy | 60 |
| Epoch được lưu vào `best.pt` | 23 |
| Validation loss tại epoch 23 | 0,163518 |
| Validation Precision@0.5 | 1,000 |
| Validation Recall@0.5 | 1,000 |
| Validation F1@0.5 | 1,000 |
| Validation AP50 | 1,000 |
| Epoch có validation loss nhỏ nhất | 10 |
| Validation loss nhỏ nhất | 0,158747 |
| AP50 tại epoch có loss nhỏ nhất | 0,909105 |
| Train loss ở epoch 59 | 0,014470 |
| Validation loss ở epoch 59 | 0,263228 |
| Validation AP50 ở epoch 59 | 1,000 |

Checkpoint được chọn theo thứ tự ưu tiên AP50, F1 và sau đó là validation loss.
Do đó epoch 23 được lưu dù epoch 10 có validation loss thấp hơn.

## 4. Đánh giá checkpoint trên test split **SYNTHETIC**

Lệnh đánh giá được chạy lại ngày 10/08/2026:

```powershell
python model/evaluate.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --split test
```

| Chỉ tiêu | Giá trị |
|---|---:|
| Số ảnh test | 100 |
| Ground-truth marker | 87 |
| True positive | 87 |
| False positive | 0 |
| False negative | 0 |
| Precision@IoU 0.5 | 1,000 |
| Recall@IoU 0.5 | 1,000 |
| F1@IoU 0.5 | 1,000 |
| AP50 | 1,000 |
| Model-only latency trung bình | 9,357 ms/ảnh |
| Model-only throughput | 106,868 FPS |
| Thiết bị | CPU |

Thông tin tái lập:

- checkpoint SHA-256: `604d5f82f3ec0b38514155b3d8f00a237b903efd7d487654ca70abc4fcf651ce`;
- config SHA-256: `23f44a70793f0c9977acfba110c43a076472003d691d3ff2a60000912d297afb`;
- test manifest SHA-256: `8fca9495d4cf72c94464aeb4ac186b68e684964b980f19da64fd82ceb3709b7c`.

AP50 bằng 1,000 chỉ chứng minh mô hình học tốt phân phối synthetic hiện tại. Không
được dùng giá trị này để kết luận mô hình đạt 100% trên môi trường indoor thực.

## 5. Kết quả round-trip bằng USB webcam **CAMERA POC**

### 5.1 Cấu hình phiên chạy

| Thuộc tính | Giá trị |
|---|---|
| Thời gian bắt đầu | 10/08/2026 04:05:00 UTC |
| Backend | `classical` — OpenCV ArUco |
| Mission | `roundtrip` |
| Start marker ID | 0 |
| Target marker ID | 1 |
| Độ phân giải camera | 640 × 480 |
| Camera FPS cấu hình/thực tế báo về | 30/30 FPS |
| Stop confirmation | 3 frame |
| Start confirmation | 3 frame |
| Target dwell | 15 frame |
| Turn duration | 30 frame |
| Camera calibration được nạp | Không |
| Kết quả | Hoàn tất `HOME_COMPLETE` |

Phiên này dùng OpenCV ArUco và không dùng CNN checkpoint. Vì vậy latency trong phần
này là latency của pipeline classical, không phải latency của MobileNetV2.

### 5.2 Tốc độ xử lý

CSV có 2.186 dòng dữ liệu, với chỉ số frame từ 0 đến 2.185. Metadata ghi `frames=2185`
do trường này lưu chỉ số frame cuối; khi báo cáo nên dùng **2.186 frame đã log**.

| Chỉ tiêu | Mean | Median | P95 |
|---|---:|---:|---:|
| Vision latency | 3,853 ms | 3,878 ms | 4,805 ms |
| End-to-end latency | 23,621 ms | 22,760 ms | 43,729 ms |

| Chỉ tiêu bổ sung | Giá trị |
|---|---:|
| Tổng thời gian log | 72,817 s |
| Throughput quan sát từ timestamp | 30,007 FPS |
| Frame thấy start marker | 369 |
| Frame thấy target marker | 609 |
| Frame thấy đồng thời cả hai marker | 0 |
| Detection source | `opencv` |

Throughput quan sát bị giới hạn gần 30 FPS bởi camera. Không lấy `1000 / mean
end-to-end latency` làm FPS thực tế của phiên này vì camera không cung cấp frame nhanh
hơn 30 FPS.

### 5.3 Timeline nhiệm vụ round-trip

| State | Frame đầu | Frame cuối | Thời điểm bắt đầu | Số frame trong state |
|---|---:|---:|---:|---:|
| `WAITING_FOR_START` | 0 | 78 | 0,000 s | 79 |
| `OUTBOUND` | 79 | 1.247 | 2,626 s | 1.169 |
| `AT_TARGET` | 1.248 | 1.262 | 41,585 s | 15 |
| `TURNING_HOME` | 1.263 | 1.292 | 42,082 s | 30 |
| `RETURNING` | 1.293 | 2.052 | 43,088 s | 760 |
| `HOME_COMPLETE` | 2.053 | 2.185 | 68,433 s | 133 |

| Pha nhiệm vụ | Thời lượng |
|---|---:|
| Chờ xác nhận marker start | 2,626 s |
| Đi từ start đến target | 38,959 s |
| Dừng tại target | 0,497 s |
| Quay về hướng start | 1,006 s |
| Trở về start | 25,345 s |
| Tổng thời gian đến `HOME_COMPLETE` | **68,433 s** |
| Thời gian tiếp tục giữ trạng thái complete trong log | 4,384 s |

Kết quả camera-driven controller là **1/1 phiên hoàn tất**. Đây chỉ là một trial và
lệnh vận tốc/góc quay vẫn được mô phỏng; không được ghi thành tỷ lệ thành công robot
100% hoặc kết quả điều khiển robot vật lý.

## 6. Hiệu chuẩn camera **CALIBRATION**

| Thuộc tính | Giá trị |
|---|---:|
| Số view checkerboard | 20 |
| Độ phân giải | 640 × 480 |
| Số góc trong checkerboard | 9 × 6 |
| Kích thước ô | 0,025 m |
| RMS reprojection error | 0,106532 |
| Mean per-view error | 0,014473 px |
| `fx` | 238,711770 px |
| `fy` | 231,107854 px |
| `cx` | 322,258103 px |
| `cy` | 222,649701 px |

Hệ số méo:

```text
[0.02483122, -0.00756445, -0.00402597, 0.00016240, 0.00192251]
```

File calibration tồn tại nhưng phiên round-trip không truyền tham số
`--camera-calibration`. Vì vậy không được dùng bảng này để khẳng định khoảng cách trong
phiên round-trip đã được hiệu chỉnh bằng `solvePnP`.

## 7. Kết quả export INT8 và hardware handoff

| Chỉ tiêu | Giá trị | Loại |
|---|---:|---|
| ONNX opset | 17 | export config |
| Input tensor | 1 × 3 × 160 × 160 | manifest |
| Số layer Conv2d được export | 54 | manifest |
| Depthwise convolution | 17 | manifest |
| Pointwise 1 × 1 convolution | 36 | manifest |
| Standard convolution còn lại | 1 | **DERIVED** |
| Calibration samples yêu cầu | 100 validation images | manifest |
| Sai số lớn nhất sau fold Conv-BN | 1,431 × 10⁻⁶ | export verification |
| Kích thước checkpoint `best.pt` | 5,427 MiB | file size |
| Kích thước FP32 ONNX | 1,742 MiB | file size |
| Kích thước INT8 NPZ | 0,516 MiB | file size |
| NPZ nhỏ hơn ONNX | 70,36% | **DERIVED** |
| Tỷ số kích thước ONNX/NPZ | 3,37× | **DERIVED** |
| Bộ nhớ tham số FP32 lý thuyết | 1,727 MiB | **DERIVED** |
| Bộ nhớ tham số INT8 lý thuyết | 0,432 MiB | **DERIVED** |

So sánh 70,36% chỉ là kích thước hai file serialization khác định dạng. Khi viết
luận văn, dùng giá trị 4× cho chênh lệch bộ nhớ tham số lý thuyết FP32/INT8; không gọi
tỷ lệ file ONNX/NPZ là mức nén phần cứng chính xác.

### 7.1 Kết quả CPU INT8 runtime PoC **SYNTHETIC**

Runtime được tạo bằng PyTorch FX post-training static quantization và backend
oneDNN. Đây là phép chạy INT8 thật trên CPU, không phải fake quantization: activation
dùng QUINT8, weight dùng QINT8 và convolution tích lũy vào INT32. Output CNN được
dequantize để dùng chung bounding-box decoder và NMS floating-point.

Lệnh đo chính thức:

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

| Chỉ tiêu | Giá trị |
|---|---:|
| Backend | PyTorch FX static INT8 / oneDNN |
| Calibration images | 100 |
| Test images | 100 |
| True positive | 87 |
| False positive | 0 |
| False negative | 0 |
| Precision@0.5 | 1,000 |
| Recall@0.5 | 1,000 |
| F1@0.5 | 1,000 |
| **Integer-runtime AP50** | **1,000** |
| Mean model latency | 8,837 ms |
| Median model latency | 8,676 ms |
| P95 model latency | 11,174 ms |
| Minimum model latency | 5,278 ms |
| Maximum model latency | 61,149 ms |
| Throughput suy ra từ mean latency | 113,161 FPS |
| Số mẫu latency | 500 |
| Warm-up | 20 inference |
| CPU threads | 8 |

Kiểm tra integer core:

| Audit item | Kết quả |
|---|---:|
| Quantized Conv2d | 54/54 |
| Quantized residual add | 10/10 |
| Float Conv2d còn lại | 0 |
| Float Linear còn lại | 0 |
| Float residual add còn lại | 0 |
| Input quantize nodes | 1 |
| Output dequantize nodes | 1 |
| TorchScript trace max absolute error | 0,000000 |
| Kích thước `int8_runtime.ts` | 0,898 MiB |

AP50 INT8 bằng AP50 FP32 trên test synthetic hiện tại, tương ứng chênh lệch AP50 bằng
0,000. Kết quả này chưa chứng minh không suy giảm trên dữ liệu camera thật. Mean latency
8,837 ms là latency của CPU INT8 runtime, không phải FPGA latency; giá trị maximum
61,149 ms cho thấy có outlier hệ điều hành nên cần báo cả median và P95.

Runtime FX dùng scale do observer của PyTorch sinh ra. Nó kiểm chứng PoC INT8 nhưng
không bit-identical với gói NPZ cho HLS; trước khi triển khai FPGA vẫn cần đối chiếu
với quy tắc requantization, rounding và saturation chính xác của accelerator.

## 8. Môi trường phần mềm của phép đo camera

| Thành phần | Phiên bản |
|---|---|
| Python | 3.10.11, 64-bit |
| PyTorch | 2.12.1+cpu |
| OpenCV | 4.13.0 |
| NumPy | 2.2.6 |
| Hệ điều hành | Windows 10, build 26300 |
| Inference device | CPU |

## 9. Các số liệu chưa được phép báo cáo

| Hạng mục | Trạng thái |
|---|---|
| AP50 trên dataset camera thực độc lập | **TBD** |
| Hybrid fallback rate trong phiên round-trip mới | **TBD** — phiên mới dùng classical |
| Độ chính xác giải mã ID có ground truth | **TBD** |
| Sai số khoảng cách/pose so với thước đo chuẩn | **TBD** |
| Sai số vị trí dừng của robot vật lý | **TBD** |
| Tỷ lệ hoàn tất trên nhiều trial robot | **TBD** |
| LUT, FF, BRAM, DSP | **TBD** — chưa synthesize FPGA |
| FPGA latency/FPS | **TBD** |
| FPGA/board power | **TBD** |
| FPGA accuracy sau INT8 | **TBD** — đã có CPU INT8 PoC nhưng chưa có board |

Không thay các ô TBD bằng dữ liệu dummy trong chương kết quả chính thức.

## 10. Đoạn văn có thể đưa vào khóa luận

### Kết quả mô hình

> Mô hình MobileNetV2-0.35 gồm 452.761 tham số và được huấn luyện trong 60 epoch
> trên 1.000 ảnh ArUco tổng hợp. Trên test split gồm 100 ảnh, trong đó có 87 ảnh
> chứa marker và 13 ảnh negative, mô hình đạt Precision, Recall, F1 và AP50 bằng
> 1,000. Model-only latency trung bình trên CPU là 9,357 ms, tương ứng 106,868 FPS.
> Do toàn bộ test split là dữ liệu tổng hợp, kết quả này chỉ xác nhận tính học được
> của kiến trúc và chưa phản ánh khả năng tổng quát hóa trên camera thực.

### Kết quả PoC camera

> Pipeline classical sử dụng USB webcam 640 × 480 đã xử lý 2.186 frame trong
> 72,817 s, đạt throughput quan sát 30,007 FPS. Vision latency trung bình là
> 3,853 ms và end-to-end latency trung bình là 23,621 ms. Bộ điều khiển lần lượt
> đi qua sáu trạng thái WAITING_FOR_START, OUTBOUND, AT_TARGET, TURNING_HOME,
> RETURNING và HOME_COMPLETE. Nhiệm vụ hoàn tất sau 68,433 s kể từ frame đầu.
> Thử nghiệm này kiểm chứng luồng camera và state machine; các lệnh motor vẫn được
> mô phỏng và chưa đại diện cho robot vật lý.

### Kết quả export

> Quá trình export tạo mô hình ONNX FP32 dung lượng 1,742 MiB và gói trọng số INT8
> dung lượng 0,516 MiB. Manifest gồm 54 lớp convolution, trong đó có 17 depthwise
> convolution và 36 pointwise convolution. Sai số lớn nhất đo được sau phép fold
> Conv-BatchNorm là 1,431 × 10⁻⁶. Các artifact này là đầu vào cho giai đoạn thiết
> kế accelerator, chưa phải kết quả triển khai hoặc đo đạc trên FPGA.

### Kết quả CPU INT8 runtime

> CPU INT8 runtime sử dụng PyTorch FX static quantization và backend oneDNN đạt
> AP50 bằng 1,000 trên 100 ảnh test tổng hợp. Với 20 lượt warm-up và 5 lần lặp toàn
> bộ test split, model latency mean, median và P95 lần lượt là 8,837 ms, 8,676 ms
> và 11,174 ms. Audit graph xác nhận cả 54 lớp convolution và 10 phép cộng residual
> đều chạy bằng quantized operator, không còn float convolution fallback. Output
> CNN được dequantize trước bước decode và NMS. Đây là baseline INT8 trên CPU,
> không phải latency hoặc accuracy đo trên FPGA.

## 11. Artifact nguồn

Hash dưới đây khóa đúng snapshot đã dùng để lập bảng. Nếu chạy demo hoặc train lại,
cần tạo file kết quả mới thay vì tiếp tục trích số liệu theo hash cũ.

| Artifact | SHA-256 |
|---|---|
| `metrics.csv` | `bcf1d6805b9f039136baf5037c56d1ae599bd6075d3a1c40d889d3cb017439d2` |
| `metrics.meta.json` | `3d338588c9fdf1c3df9f772eb630a934f3ea7cf8fd4f87a9f6d722fdf074e81d` |
| `history.jsonl` | `5fab29273c8abddc5091a83042daaf529037268ddfd4f0d018dd0e32276beb4b` |
| `camera.json` | `d018e24d373fd7ec809eb206368b4502097af8b50de43b8f04249736c2902285` |
| `room_a_daylight.txt` | `f595c1b43d9865e5ce73fb43ff12777e688d04cb61f7ff917aee81bafb796993` |
| INT8 `metrics.json` | `c623534d14fa85bcf31df98a28eb1f54534404909b2e5b7891510a2ab6f43422` |
| `int8_runtime.ts` | `f3ca78514ce651eb693a1d8c69925fcd3d2bb054a7a4509c2b27952e7e022b47` |
| INT8 `latency_samples.csv` | `9b3741e89e8f41998884f1312f10ad41b4861e08e47086c415d96c3171c4f740` |
| `quantization_audit.json` | `55cd14535ecbb8a139777796006aa89a25913670f42c23bbddb6cca9a925fdf7` |

- `artifacts/aruco_mbv2_035/history.jsonl`
- `artifacts/aruco_mbv2_035/best.pt`
- `artifacts/aruco_mbv2_035/resolved_config.json`
- `artifacts/aruco_mbv2_035/export/int8_manifest.json`
- `artifacts/aruco_mbv2_035/export/aruco_mobilenetv2_035_fp32.onnx`
- `artifacts/aruco_mbv2_035/export/aruco_mobilenetv2_035_weights_int8.npz`
- `artifacts/webcam_poc/metrics.csv`
- `artifacts/webcam_poc/metrics.meta.json`
- `artifacts/calibration/camera.json`
- `dataset/aruco/train.txt`
- `dataset/aruco/val.txt`
- `dataset/aruco/test.txt`
- `dataset/aruco/room_a_daylight.txt`
- `artifacts/evaluation/int8_runtime/metrics.json`
- `artifacts/evaluation/int8_runtime/predictions.csv`
- `artifacts/evaluation/int8_runtime/pr_curve.csv`
- `artifacts/evaluation/int8_runtime/latency_samples.csv`
- `artifacts/evaluation/int8_runtime/quantization_audit.json`
- `artifacts/evaluation/int8_runtime/int8_runtime.ts`
