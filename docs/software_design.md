# Thiết kế phần mềm hệ thống ArUco robot

## 1. Mục tiêu

Phần mềm được thiết kế theo detector backend interface để cùng một application
pipeline chạy được với OpenCV, PyTorch CPU/GPU hoặc FPGA trong tương lai.

```text
detect(frame) → list[MarkerDetection]
```

Nhờ đó việc thay PyTorch bằng PYNQ overlay không làm thay đổi ArUco decoding,
distance estimation, robot controller, UI hoặc logging.

## 2. Các package chính

```text
model/
├── aruco_detector/
│   ├── network.py          # MobileNetV2-0.35 + detection head
│   ├── dataset.py          # YOLO dataset loader
│   ├── loss.py             # Objectness + box regression
│   ├── metrics.py          # Precision/Recall/F1/AP50
│   └── config.py
├── train.py
├── evaluate.py
├── export_int8.py
└── infer_aruco.py

poc/
├── live_webcam_demo.py     # Classical/CNN/Hybrid runtime
├── robot_controller.py     # One-way stop state machine
├── mission_controller.py   # Two-marker round-trip state machine
├── calibrate_focal_length.py
├── make_demo_video.py
└── tests/
```

## 3. Detector backends

### ClassicalDetector

OpenCV ArUco chạy toàn frame. Đây là immediate baseline và không cần model.

### CnnRoiDetector

1. Resize RGB 160×160.
2. MobileNetV2 inference.
3. Decode 5×5 grid.
4. NMS.
5. Crop ROI có padding.
6. OpenCV ArUco giải mã ID/corners trong ROI.

### HybridDetector

Chạy CNN trước; nếu không có marker đã giải mã thì fallback ClassicalDetector.
CSV ghi nguồn `cnn+opencv` hoặc `opencv`.

### FPGA backend dự kiến

```python
class PynqDetector:
    def detect(self, frame):
        # preprocess → DMA buffer → accelerator → output decode
        return detections
```

Đây mới là interface dự kiến, chưa có implementation cho model ArUco mới.

## 4. Training pipeline

### Dataset contract

- RGB image;
- YOLO row: `0 cx cy width height`;
- coordinates normalized `[0,1]`;
- class 0 duy nhất là `aruco_marker`;
- negative image có label file rỗng.

### Target assignment

Marker được gán vào grid cell chứa tâm. Target gồm objectness, center offset và
normalized width/height. Nếu nhiều marker có cùng cell, baseline hiện chỉ giữ
một target; dense-marker scenes không phải phạm vi chính.

### Checkpoint

- `last.pt`: trạng thái epoch gần nhất;
- `best.pt`: chọn theo AP50, sau đó F1 và validation loss;
- chứa model, optimizer, scheduler, config và metrics;
- `resolved_config.json` và `history.jsonl` phục vụ tái lập.

Resume khôi phục optimizer/scheduler và không tự ghi đè checkpoint tốt hơn.

## 5. Evaluation

Metric:

- Precision@IoU0.5;
- Recall@IoU0.5;
- F1@IoU0.5;
- AP50 không phụ thuộc operating score threshold;
- model-only latency/FPS.

End-to-end FPS được đo riêng trong PoC vì còn bao gồm camera, preprocess,
ArUco, controller, rendering và logging.

## 6. INT8 export

`model/export_int8.py`:

1. load checkpoint;
2. fold adjacent Conv2d + BatchNorm2d;
3. kiểm tra max output error;
4. calibration activation bằng validation manifest;
5. quantize weight symmetric per-output-channel;
6. tạo INT32 bias;
7. export ONNX opset 17 và NPZ/JSON manifest.

Đây chưa phải integer-only inference. Cần bổ sung reference thực hiện multiplier,
shift, rounding, saturation và residual add đúng bit.

## 7. Robot controller

State:

```text
SEARCHING → APPROACHING → SLOWING → STOPPED
```

- median filter làm mượt distance/side-pixel;
- close condition phải đúng N frame liên tiếp;
- mất marker ngắn hạn giữ tốc độ thấp;
- STOP được latch cho tới khi reset;
- chỉ marker có `target_id` tác động controller.

### 7.1. Round-trip mission controller

`poc/mission_controller.py` nhận hai observation độc lập (`start_id`, `target_id`)
và chạy chuỗi:

```text
WAITING_FOR_START -> OUTBOUND -> AT_TARGET -> TURNING_HOME
                  -> RETURNING -> HOME_COMPLETE
```

Output gồm `linear_speed`, `angular_speed`, `active_marker_id`,
`target_arrived` và `mission_complete`. `HOME_COMPLETE` khóa dừng cho đến reset.
PoC quay đầu theo số frame; encoder/IMU là phần nâng cấp cần có trên robot thật.

Speed hiện là số mô phỏng `[0,1]`. UART/GPIO/PWM chưa được nối robot thật.

## 8. Logging

PoC CSV gồm:

- timestamp/frame;
- mode;
- vision/total latency;
- smoothed FPS;
- detected IDs;
- target visibility/source;
- distance method, pose x/y/z và side length;
- controller state/speed.

Mỗi run tạo thêm `.meta.json` chứa CLI arguments, version môi trường, actual
capture properties và SHA-256 checkpoint/input/calibration.

FPS từ headless video file không phải camera real-time FPS vì file được đọc nhanh
nhất có thể.

## 9. Error handling

- checkpoint thiếu: liệt kê checkpoint hiện có và gợi ý classical mode;
- manifest rỗng/ảnh thiếu: fail sớm;
- config input không chia hết stride 32: reject;
- camera/video không mở được: dừng với lỗi rõ;
- output writer lỗi: không im lặng tiếp tục.

## 10. Test strategy

Unit tests hiện kiểm tra:

- output tensor contract;
- target/loss/backprop;
- AP calculation;
- Conv–BN folding;
- ArUco auto-label;
- classical detection/distance;
- multi-frame stop confirmation;
- STOP latch/reset.

Integration test dùng video marker tiến gần camera để kiểm tra state transitions
và CSV mà không cần webcam vật lý.
