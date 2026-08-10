# Phương pháp phân tích hiệu năng

## 1. Phạm vi

Hiện chưa có FPGA accelerator cho MobileNetV2-0.35 ArUco. Tài liệu này định
nghĩa cách đo CPU/PoC/FPGA để kết quả sau này có thể so sánh công bằng.

## 2. Backend cần so sánh

1. Classical OpenCV ArUco trên CPU.
2. CNN FP32 PyTorch trên CPU.
3. Hybrid CPU: CNN ROI + OpenCV fallback.
4. CNN INT8 CPU runtime PoC (PyTorch FX/oneDNN).
5. CNN INT8 FPGA + ARM post-processing.

Mọi backend dùng cùng frame order, resolution, test manifest và target ID.

## 3. Phân rã latency

```text
T_end_to_end = T_capture
             + T_preprocess
             + T_inference
             + T_nms_roi
             + T_aruco_pose
             + T_controller
             + T_render_log
```

Phải báo ít nhất:

- warm-up count;
- số frame;
- mean, median, P95, min và max;
- model-only latency;
- end-to-end latency;
- sustained FPS.

CUDA cần synchronize trước khi dừng timer. FPGA cần phân biệt PS wall-clock,
PL cycle count và transfer overhead.

## 4. Camera và video benchmark

### USB webcam

Ghi:

- camera model/backend;
- requested và actual resolution/FPS;
- ánh sáng;
- display on/off;
- CPU/GPU model;
- power mode;
- số frame bỏ đầu.

### Video file

Headless video file thường được đọc nhanh hơn real-time. FPS này chỉ dùng để so
sánh software compute, không được gọi là camera FPS.

## 5. Detection performance

- CNN: Precision50, Recall50, F1, AP50.
- Pipeline: ArUco ID decode rate.
- Hybrid: fallback rate.
- Robot safety: false stop rate và missed stop rate.

AP50 phải được tính từ ranked predictions trước operating score threshold.

## 6. FPGA performance

### Latency

```text
T_fpga_e2e = T_PS_preprocess
           + T_buffer_flush
           + T_DMA/input
           + T_PL_compute
           + T_DMA/output
           + T_buffer_invalidate
           + T_ARM_postprocess
```

### Throughput

Single-frame latency và pipelined throughput có thể khác nhau. Nếu dùng ping-
pong buffer, báo cả latency/frame và sustained FPS.

### Resource

Chỉ dùng post-route utilization report:

- LUT/FF;
- BRAM18/36 equivalence;
- DSP48E1;
- clock buffers;
- WNS/TNS và achieved clock.

Không suy ra DSP từ số MAC trong source.

## 7. Power và năng lượng

Đo ít nhất hai trạng thái:

```text
idle: board booted, accelerator inactive
active: sustained inference cùng workload
```

```text
dynamic_power ≈ active_power - idle_power
energy_per_frame = active_power / sustained_fps
```

Ghi thiết bị đo, sample rate, thời gian đo, điện áp và phạm vi hệ thống (PL hay
toàn board). Vivado power estimate và đo ngoài board phải được ghi riêng.

## 8. Quantization comparison

| Variant | AP50 | Model size | Latency | Notes |
|---|---:|---:|---:|---|
| FP32 | 1,000 synthetic | 1,742 MiB ONNX | 9,357 ms mean | CPU reference; phép đo một lượt |
| INT8 CPU FX | 1,000 synthetic | 0,898 MiB TorchScript | 8,837 ms mean | 500 mẫu; không bit-identical với HLS |
| INT8 HLS reference | TBD | TBD | TBD | Phải dùng đúng requantization phần cứng |
| INT8 FPGA | TBD | TBD | TBD | Must match reference |
| INT4 QAT | Optional | TBD | TBD | Chỉ làm sau INT8 |

Hai latency CPU hiện được thu bằng số lượt khác nhau nên chỉ là baseline riêng, chưa
được dùng để kết luận speedup FP32→INT8. Khi so sánh chính thức phải chạy cùng warm-up,
repeats, thread count và test order. So sánh tensor layer-by-layer trước khi chỉ nhìn
AP cuối.

## 9. Robot experiment

Mỗi condition chạy nhiều trial độc lập. Ghi:

- target ID;
- approach speed;
- stop threshold;
- marker size;
- camera pose;
- ground-truth stop distance;
- measured stop error;
- false/missed stop;
- video và CSV.

## 10. Smoke numbers hiện có

Các số đo local trước đây chỉ xác nhận code:

- CPU model inference cỡ hàng chục millisecond trên máy phát triển;
- classical headless synthetic video nhanh hơn camera real-time;
- STOP transition xảy ra đúng trong video tạo sẵn.

Không dùng các số này trong bảng kết quả chính thức vì chưa khóa hardware,
dataset và protocol.

## 11. Acceptance criteria đề xuất

Tiêu chí cuối cùng phải được chốt sau baseline camera thật. Một bộ tiêu chí hợp
lý cần bao gồm:

- end-to-end FPS đáp ứng camera target;
- AP50/ID rate không giảm quá mức sau INT8;
- FPGA fit và timing closure;
- không false stop trên negative test;
- robot dừng trong sai số cho phép qua nhiều trial;
- energy/frame thấp hơn CPU baseline phù hợp.
