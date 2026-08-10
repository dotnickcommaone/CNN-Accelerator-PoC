# CNN Accelerator cho nhận diện ArUco trên robot giao hàng indoor

Project nghiên cứu pipeline thị giác thời gian thực cho robot giao hàng trong
nhà: camera nhận hình ảnh, CNN nhẹ tìm vùng marker, OpenCV ArUco giải mã ID,
hệ thống ước lượng khoảng cách và phát lệnh giảm tốc/dừng. Đích cuối cùng là
chuyển MobileNetV2-0.35 INT8 sang accelerator trên PYNQ-Z2.

> **Trạng thái hiện tại:** PoC USB webcam trên laptop đã hoạt động. Model,
> training, evaluation, ONNX/INT8 export và controller đã có.
## Kiến trúc

```text
Hiện tại — laptop
USB webcam → MobileNetV2-0.35 / OpenCV → ArUco ID
           → distance filter → stop/round-trip controller → UI + CSV

Đích — PYNQ-Z2
Camera → ARM preprocessing → FPGA INT8 CNN accelerator
       → ARM ArUco/pose/controller → UART/GPIO → robot
```

Tài liệu tổng quan:

- [Documentation index](docs/README.md)
- [System overview](docs/system_overview_aruco_robot.md)
- [Current overall architecture for Prism](docs/diagrams/current_poc_overall_architecture.drawio)
- [Simplified thesis diagrams](docs/diagrams/thesis_architecture_diagrams.drawio)
- [Draw.io architecture](docs/diagrams/aruco_robot_system_architecture.drawio)
- [Software design](docs/software_design.md)
- [Hardware design](docs/hardware_design.md)
- [User manual](docs/user_manual.md)
- [Round-trip demo workflow](poc/ROUNDTRIP_DEMO.md)
- [Experiment runner và logging](experiments/README.md)
- [Dummy comparison data cho bản nháp](docs/dummy_backend_results.md)

## Vì sao kết hợp CNN và ArUco?

| Khối | Vai trò |
|---|---|
| MobileNetV2-0.35 | Phát hiện ROI có marker trong ảnh |
| OpenCV ArUco | Giải mã ID và tìm bốn góc marker |
| Distance estimator | Khoảng cách pinhole hoặc kích thước pixel tương đối |
| Stop controller | SEARCHING → APPROACHING → SLOWING → STOPPED |

ArUco cổ điển là baseline nhanh và đáng tin cậy khi marker rõ. CNN được nghiên
cứu để tăng độ bền trong nền phức tạp, marker nhỏ, thiếu sáng, blur hoặc góc
nghiêng. Chế độ hybrid dùng CNN trước và fallback OpenCV toàn khung.

## Model baseline

| Thuộc tính | Giá trị |
|---|---|
| Backbone | MobileNetV2-0.35 |
| Input | RGB 160×160, `[0,1]` |
| Số tham số | khoảng 452 nghìn |
| Output | grid 5×5×5 |
| Class | `aruco_marker` |
| Nhãn | YOLO normalized |
| Đích số học | INT8 weight/activation, INT32 accumulator |

Detection head sinh objectness, offset tâm, width và height. ArUco ID không do
CNN phân loại mà do OpenCV giải mã trong ROI.

## Chạy PoC ngay

Không cần checkpoint:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical --target-id 0
```

Hybrid cần checkpoint đã train:

```powershell
python poc/live_webcam_demo.py --source 0 --mode hybrid `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

Nếu checkpoint chưa tồn tại, dùng `classical` hoặc train model trước. Các
checkpoint `learnability_check*` chỉ là sanity check trên dữ liệu tổng hợp.

Hướng dẫn PoC đầy đủ: [poc/README.md](poc/README.md).

Demo khứ hồi hai marker (`start -> target -> start`) không cần webcam:

```powershell
python poc/make_roundtrip_demo_video.py --start-id 0 --target-id 1
python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/roundtrip_input.avi `
  --mode classical --mission roundtrip --start-id 0 --target-id 1 --headless
```

Thu nhiều trial và gom toàn bộ raw log, video, bảng Prism và metadata vào một thư
mục experiment:

```powershell
python experiments/run_poc_experiment.py `
  --source synthetic --name offline_roundtrip --trials 3
```

## Dataset

Sinh dữ liệu tổng hợp để kiểm tra code:

```powershell
python model/scripts/generate_synthetic_aruco.py `
  --output dataset/aruco --count 1000
```

Thu thập dữ liệu camera thật và auto-label:

```powershell
python model/scripts/collect_aruco_dataset.py --source 0 `
  --output dataset/aruco --session room_a_daylight `
  --every 10 --max-saved 500 --include-negatives --display
```

Phải chia train/validation/test theo toàn bộ recording session, không chia ngẫu
nhiên các frame liền nhau của cùng video.

## Train và đánh giá

```powershell
python model/train.py --config model/configs/mobilenetv2_035.yaml
```

```powershell
python model/evaluate.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt
```

Metric hiện có: Precision@0.5, Recall@0.5, F1@0.5, AP50, model latency và
model FPS. Kết quả tổng hợp không được dùng làm kết quả luận văn.

## Export ONNX và INT8 handoff

```powershell
python model/export_int8.py `
  --config model/configs/mobilenetv2_035.yaml `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --calibration-samples 100
```

Exporter thực hiện:

- fold Conv + BatchNorm;
- ONNX opset 17;
- INT8 weight theo output channel;
- INT32 bias;
- activation abs-max calibration;
- manifest mô tả layer cho HLS.

Đây là handoff artifact, chưa phải integer-only reference. Trước HLS cần mô
phỏng requantization, saturation và residual add bit-accurate.

CPU INT8 runtime cho PoC có thể đánh giá riêng bằng:

```powershell
python model/evaluate_int8_runtime.py `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --calibration-samples 100 --warmup 20 --repeats 5 `
  --threads 8 --save-runtime
```

Runtime này dùng quantized CPU kernels và tự audit float fallback. Nó cung cấp
AP50/latency INT8 cho PoC, nhưng chưa bit-identical với requantization của HLS.

## Cấu trúc repository

```text
CNN-Accelerator/
├── model/                  # Model, train, evaluate, INT8 export
├── poc/                    # USB webcam PoC và mission controller
├── analysis/               # Export bảng và dummy-data generators
├── experiments/            # Runner nhiều trial và logging
├── dataset/aruco/          # Hướng dẫn dataset; ảnh/nhãn local không commit
├── docs/                   # Tài liệu project hiện tại
└── artifacts/              # Checkpoint/log/output local, bị Git ignore
```

## Trạng thái triển khai

| Hạng mục | Trạng thái |
|---|---|
| PoC webcam/video | Hoàn thành |
| Classical/CNN/Hybrid interface | Hoàn thành |
| Stop state machine và CSV | Hoàn thành |
| Round-trip state machine hai marker | Hoàn thành ở PoC mô phỏng |
| Synthetic/real capture tools | Hoàn thành |
| Dataset audit/session split | Hoàn thành |
| Camera intrinsic/solvePnP tools | Hoàn thành; đã có calibration 20 view cho webcam hiện tại |
| FP32 training/evaluation | Hoàn thành về code |
| Dataset camera thật đủ lớn | Chưa hoàn thành |
| Checkpoint chính thức | Chưa hoàn thành |
| Calibrated INT8 export | Hoàn thành về tool |
| CPU INT8 runtime và AP50/latency | Hoàn thành cho PoC synthetic |
| HLS bit-accurate integer reference | Chưa hoàn thành |
| MobileNetV2 HLS accelerator | Chưa hoàn thành |
| Vivado/PYNQ overlay cho model mới | Chưa hoàn thành |
| Robot motor integration | Chưa hoàn thành |
| FPGA FPS/power/resource measurements | Chưa hoàn thành |


## Bước tiếp theo

1. Thu thập và review dataset camera thật.
2. Train checkpoint chính thức và đánh giá classical/CNN/hybrid.
3. Xây HLS bit-accurate reference và đối chiếu với CPU INT8 runtime hiện tại.
4. Thiết kế depthwise/pointwise HLS accelerator.
5. Tạo Vivado overlay mới và thay backend PyTorch bằng PYNQ backend.
6. Tích hợp robot, đo FPS, power, tài nguyên và sai số dừng.
