# Kết quả và trạng thái thực nghiệm

## 1. Nguyên tắc

Tài liệu chỉ ghi kết quả có nguồn artifact/log. Số liệu cat/dog legacy, estimate
DSP/BRAM/power và synthetic smoke test không được trình bày như kết quả luận
văn ArUco.

## 2. Kết quả engineering verification hiện có

### Model/toolchain

| Kiểm tra | Kết quả |
|---|---|
| Model forward contract | `[B,3,160,160] → [B,5,5,5]` |
| Model parameters | 452.761 |
| Loss/backprop/optimizer step | Pass |
| Sanity overfit nhỏ | Có thể học box, TP=3/FP=0/FN=0 trên 3 positive training samples |
| Conv–BN folding | Max error `1.431e-6` trong export hiện tại |
| ONNX checker | Pass, opset 17 trong smoke export |
| INT8 handoff | Weight/scales/bias/calibration manifest được tạo |
| CPU INT8 runtime | 54/54 Conv2d và 10/10 residual add được quantize; không có float core fallback |
| CPU INT8 synthetic test | AP50=1,000; mean/median/P95 latency=8,837/8,676/11,174 ms |
| Model unit tests | Pass tại lần kiểm tra gần nhất |

Sanity overfit chỉ xác nhận code/loss học được; không phản ánh generalization.

### PoC offline

Video tổng hợp marker tiến gần camera đã xác nhận:

- classical detection đọc đúng marker;
- hybrid có thể sử dụng CNN và fallback;
- state chuyển APPROACHING → SLOWING → STOPPED;
- STOP chỉ xảy ra sau nhiều frame xác nhận;
- STOP latch cho tới reset;
- CSV và output video được tạo.

Trong một smoke run 80 frame, STOP xuất hiện lần đầu tại frame 52 khi cạnh
marker khoảng 182 pixel với stop threshold 170 pixel. Đây là deterministic
functional test, không phải kết quả robot thật.

## 3. Kết quả chưa có

| Kết quả | Trạng thái |
|---|---|
| Accuracy trên camera test set độc lập | Chưa có |
| AP50 checkpoint chính thức | Chưa có |
| FP32–INT8 accuracy delta trên synthetic test | 0,000; không đại diện camera thật |
| FP32–INT8 accuracy delta trên camera test | Chưa có |
| USB webcam sustained FPS đã chuẩn hóa | Chưa có report chính thức |
| FPGA inference latency/FPS | Chưa có accelerator mới |
| FPGA LUT/FF/BRAM/DSP/Fmax | Chưa có report mới |
| FPGA power/J per frame | Chưa đo |
| Robot stop error | Chưa có robot thật |

## 4. Bảng kết quả chính thức cần điền

### Detection và ID

| Backend | Precision50 | Recall50 | F1-50 | AP50 | ID decode rate | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|
| Classical CPU | N/A | TBD | TBD | N/A | TBD | N/A |
| CNN FP32 CPU | TBD | TBD | TBD | TBD | TBD | N/A |
| Hybrid CPU | TBD | TBD | TBD | TBD | TBD | TBD |
| CNN INT8 FPGA | TBD | TBD | TBD | TBD | TBD | N/A |

### Latency

| Backend | Preprocess ms | CNN ms | ArUco ms | End-to-end ms | FPS |
|---|---:|---:|---:|---:|---:|
| Classical CPU | TBD | N/A | TBD | TBD | TBD |
| CNN FP32 CPU | TBD | TBD | TBD | TBD | TBD |
| Hybrid CPU | TBD | TBD | TBD | TBD | TBD |
| CNN INT8 FPGA | TBD | TBD | TBD | TBD | TBD |

### FPGA

| LUT | FF | BRAM | DSP | Fmax | WNS | Active W | J/frame |
|---:|---:|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Robot

| Condition | Trials | Stop success | Mean error cm | Max error cm | Std cm |
|---|---:|---:|---:|---:|---:|
| Frontal/daylight | TBD | TBD | TBD | TBD | TBD |
| 30°/daylight | TBD | TBD | TBD | TBD | TBD |
| Frontal/low light | TBD | TBD | TBD | TBD | TBD |
| Motion blur | TBD | TBD | TBD | TBD | TBD |

## 5. Artifact policy

Mỗi bảng phải liên kết tới:

- config/checkpoint hash;
- test manifest;
- raw prediction file;
- latency CSV;
- Vivado reports;
- power raw samples;
- robot trial log/video.

Nếu thiếu artifact, giá trị phải là `TBD`, không điền estimate.

## 6. Legacy results

Ảnh, weight, bitstream và báo cáo cat/dog đã được loại khỏi working tree. Chúng
chỉ còn trong Git history và không được so sánh trực tiếp với model ArUco mới.

## 7. Công cụ xuất kết quả hiện tại

- Dataset: `model/scripts/audit_aruco_dataset.py`;
- Model summary/raw predictions/PR: `model/evaluate.py --output-json ...`;
- CPU INT8 AP50/latency/audit: `model/evaluate_int8_runtime.py`;
- PoC latency/state: CSV từ `poc/live_webcam_demo.py`;
- Tổng hợp nhiều backend: `analysis/export_thesis_tables.py`;
- Camera calibration: NPZ/JSON từ `poc/calibrate_camera.py`.

Ví dụ engineering-only hiện có được ghi tại
[poc_smoke_report_2026-08-04.md](poc_smoke_report_2026-08-04.md).
Snapshot kết quả mới nhất được tổng hợp tại
[thesis_latest_measurements_2026-08-10.md](thesis_latest_measurements_2026-08-10.md).
