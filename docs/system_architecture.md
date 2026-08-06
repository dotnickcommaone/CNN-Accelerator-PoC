# Kiến trúc hệ thống ArUco CNN Accelerator

Tài liệu này mô tả ranh giới và dataflow của hệ thống. Bản mô tả đầy đủ hơn và
sơ đồ Draw.io nằm tại:

- [System overview](system_overview_aruco_robot.md)
- [Draw.io editable diagram](diagrams/aruco_robot_system_architecture.drawio)

## 1. Hai cấu hình hệ thống

### PoC hiện tại

```text
USB webcam/video
  → detector backend: classical / CNN / hybrid
  → ArUco ID + corners
  → distance estimator
  → temporal filter
  → stop controller
  → simulated motor command + UI + CSV
```

PoC chạy trên laptop bằng Python, PyTorch và OpenCV.

### Hệ thống FPGA đích

```text
Camera
  → ARM capture/preprocess
  → FPGA MobileNetV2-0.35 INT8 inference
  → ARM NMS/ArUco/pose/controller
  → UART/GPIO/PWM
  → robot
```

FPGA backend chưa được triển khai cho model mới.

## 2. PoC component architecture

| Component | Input | Output | Implementation |
|---|---|---|---|
| Video capture | USB/video/stream | BGR frame | OpenCV |
| Classical detector | Full frame | ID/corners | OpenCV ArUco |
| CNN detector | RGB 160×160 | ROI boxes | PyTorch MobileNetV2 |
| ROI decoder | Cropped BGR | ID/corners | OpenCV ArUco |
| Distance estimator | Corners + calibration | meter/side pixel | Pinhole/proxy |
| Stop controller | target visibility/distance | state/speed | Python state machine |
| Logger/UI | Pipeline result | CSV/video/window | OpenCV/Python |

## 3. Detection mode behavior

```text
classical: full-frame ArUco
cnn:       CNN ROI → ArUco within ROI
hybrid:    CNN ROI → ArUco; fallback full-frame ArUco if empty
```

Hybrid không được xem mặc định là CNN thành công. `target_source` trong CSV phải
được dùng để tính fallback rate.

## 4. Model data contract

### Input

```text
dtype: float32 (software baseline)
shape: [B, 3, 160, 160]
layout: NCHW
order: RGB
range: [0,1]
```

### Output

```text
shape: [B, 5, 5, 5]
channels: objectness, dx, dy, width, height
```

Output được sigmoid/decode về normalized xyxy box, lọc score và NMS.

### FPGA target contract

```text
input/output activations: INT8
weights: INT8 per output channel
bias/accumulator: INT32
layout/scale: phải khóa từ integer reference trước HLS
```

## 5. Runtime sequence

1. Capture frame và timestamp.
2. Detector backend xử lý frame.
3. Chọn marker đúng target ID.
4. Ước lượng distance hoặc side-pixel proxy.
5. Median filter.
6. Controller cập nhật state.
7. Overlay và log CSV.
8. Sau N close frames, latch STOP.

## 6. State machine

| State | Entry condition | Output |
|---|---|---|
| SEARCHING | Không thấy target | Search speed |
| APPROACHING | Target xa | Nominal speed |
| SLOWING | Target trong slow zone hoặc mất tạm thời | Low speed |
| STOPPED | Close condition đủ N frame | Speed 0, latch |

Marker sai ID không được phép làm robot dừng.

## 7. PS–PL architecture dự kiến

### Control path

```text
ARM → AXI-Lite registers → accelerator start/config
accelerator done/interrupt → ARM
```

### Data path

```text
PYNQ contiguous DDR buffers
  ↔ AXI DMA hoặc HLS m_axi qua HP port
  ↔ BRAM tile buffers
  ↔ depthwise/pointwise compute engines
```

### Coherency

- ARM flush buffer trước PL read;
- ARM invalidate output trước read;
- physical address phải đến từ PYNQ allocation;
- double buffering là optimization sau khi single-frame path đúng.

## 8. Failure handling

| Failure | Safe behavior |
|---|---|
| Camera frame fail | Stop processing/log error |
| CNN no box | Hybrid fallback hoặc target not visible |
| ArUco cannot decode ROI | Không phát target command |
| Marker lost ngắn | SLOWING trong tolerance window |
| Marker lost dài | SEARCHING |
| Accelerator timeout | Không dùng stale output; báo lỗi |
| Invalid magic/shape/scale | Abort inference |

## 9. Performance boundaries

Đo riêng:

```text
capture_ms
preprocess_ms
cnn_ms
aruco_ms
control_ms
render_log_ms
end_to_end_ms
```

`model_fps = 1000/cnn_ms` không được gọi là end-to-end FPS.

## 10. Migration path

1. Khóa checkpoint FP32 và test set.
2. Khóa integer input/output contract.
3. Viết bit-accurate reference.
4. Implement HLS primitives.
5. So sánh layer-by-layer.
6. Tích hợp Vivado/PYNQ.
7. Thêm `PynqDetector.detect(frame)`.
8. Chạy lại cùng PoC video/test set.

Các module sau không đổi khi migrate: ArUco, distance, controller, UI, CSV và
offline video tests.
