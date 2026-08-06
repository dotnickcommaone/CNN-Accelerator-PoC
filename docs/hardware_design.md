# Thiết kế phần cứng dự kiến cho CNN ArUco accelerator

## 1. Phạm vi tài liệu

Tài liệu mô tả kiến trúc phần cứng **đích** cho MobileNetV2-0.35 INT8 trên
PYNQ-Z2. Accelerator này chưa được tổng hợp. Vì vậy mọi tài nguyên, Fmax, FPS và
power phải được xem là chỉ tiêu cần đo, không phải kết quả đã đạt.

Accelerator cat/dog cũ đã được loại khỏi working tree vì không thực hiện model
ArUco hiện tại. Nếu cần đối chiếu quy trình AXI/PYNQ, có thể xem lại trong Git
history; không tái sử dụng bitstream hoặc weight cũ làm kết quả đề tài.

## 2. Nền tảng mục tiêu

| Thành phần | Lựa chọn |
|---|---|
| Board | PYNQ-Z2 |
| SoC | Zynq-7000 XC7Z020 |
| PS | Dual ARM Cortex-A9 |
| PL | FPGA fabric với LUT, FF, BRAM và DSP48E1 |
| Tool flow dự kiến | Vitis HLS/Vivado + PYNQ Python |
| Mô hình | MobileNetV2-0.35, input RGB 160×160 |
| Độ chính xác | INT8 weight/activation, INT32 accumulator |

## 3. Phân chia PS–PL

| Chức năng | ARM/PS | FPGA/PL |
|---|:---:|:---:|
| Camera capture | ✓ | |
| Resize/normalize | ✓ | Có thể chuyển sau |
| Depthwise convolution | | ✓ |
| Pointwise convolution | | ✓ |
| ReLU6/requantization | | ✓ |
| Residual add | | ✓ |
| Detection head | | ✓ |
| Decode grid/NMS | ✓ | |
| ArUco ID/pose | ✓ | |
| Stop controller | ✓ | |
| UART/GPIO/UI/logging | ✓ | |

PL chỉ tăng tốc khối tính toán đều và song song cao. NMS, ArUco và controller có
nhiều nhánh điều kiện nên được giữ trên ARM.

## 4. Dataflow mục tiêu

```text
DDR input tensor
  → AXI DMA / HP port
  → Load tile vào BRAM
  → Depthwise 3×3 hoặc Pointwise 1×1
  → INT32 bias/accumulation
  → Requantize + saturate INT8
  → ReLU6
  → Optional residual add
  → Store tile về DDR
  → Detection head 5×5×5
```

## 5. Compute engine

### 5.1 Depthwise engine

- line buffer cho ba hàng ảnh;
- sliding window 3×3;
- một kernel riêng cho từng channel;
- song song hóa theo channel trong giới hạn DSP/BRAM;
- hỗ trợ stride 1 và stride 2.

### 5.2 Pointwise engine

- dot product giữa input-channel vector và weight 1×1;
- tiling theo input/output channel;
- reuse input activation cho nhiều output channel;
- reuse factor cấu hình được để cân bằng latency và DSP.

### 5.3 Residual và requantization

MobileNetV2 có residual add khi shape/stride cho phép. Hai nhánh phải được đưa
về scale tương thích trước phép cộng. Pipeline cần:

```text
INT32 accumulator × multiplier → shift/round → saturate INT8
```

Scale, multiplier và shift phải đến từ integer-only reference, không suy đoán
trực tiếp trong HLS.

## 6. Bộ nhớ

| Vùng | Vai trò |
|---|---|
| DDR | Input, output, weights hoặc feature map ngoài tile |
| BRAM line buffer | Cửa sổ depthwise 3×3 |
| BRAM activation tile | Ping-pong load/compute/store |
| BRAM weight tile | Giảm truy cập DDR lặp lại |
| Register/LUTRAM | Partial sum nhỏ, scale và control |

Thiết kế phải chọn tile từ report thực tế, không giả định toàn bộ feature map
hoặc weights đều nằm vừa BRAM.

## 7. Giao tiếp dự kiến

### AXI-Lite control

- `ap_start`, `ap_done`, `ap_idle`, `ap_ready`;
- input/output physical address;
- layer configuration;
- tensor dimensions, stride, activation mode;
- interrupt enable/status.

### AXI master hoặc DMA

- truyền activation/weight bằng burst;
- sử dụng PYNQ contiguous buffers;
- flush trước khi PL đọc và invalidate trước khi ARM đọc;
- ưu tiên double buffering để overlap PS preprocessing với PL inference.

## 8. HLS optimization cần đánh giá

- `PIPELINE` và initiation interval thực tế;
- `DATAFLOW` giữa load/compute/store;
- array partition theo channel/PE;
- BRAM binding cho tile buffers;
- loop unroll có giới hạn;
- burst length và memory coalescing;
- reuse factor;
- weight stationary so với output stationary.

Mỗi pragma phải được đối chiếu với synthesis report, không suy luận rằng
`PIPELINE II=1` tự động đạt được.

## 9. Verification plan

1. Python FP32 reference.
2. Python integer-only reference.
3. HLS C simulation từng primitive.
4. So sánh tensor layer-by-layer.
5. HLS synthesis và timing estimate.
6. RTL co-simulation.
7. Vivado implementation/timing closure.
8. PYNQ board inference cùng test vector.
9. End-to-end camera validation.

Sai số phải được kiểm tra ở tensor, bounding box và AP50; chỉ kiểm tra class/ID
cuối là không đủ để phát hiện lỗi quantization.

## 10. Report cần lưu

```text
hls/reports/
  csynth.rpt
  cosim.rpt
vivado/reports/
  utilization_post_synth.rpt
  utilization_post_route.rpt
  timing_summary.rpt
  power_report.rpt
experiments/fpga/
  latency.csv
  accuracy.json
  power.csv
```

## 11. Chỉ tiêu đánh giá

| Nhóm | Metric |
|---|---|
| Tài nguyên | LUT, FF, BRAM, DSP |
| Timing | target clock, achieved Fmax, WNS/TNS |
| Model | FP32 AP50, INT8 AP50, delta AP50 |
| Hiệu năng | PL latency, end-to-end latency, FPS |
| Năng lượng | idle W, active W, J/frame |
| Robot | stop success rate, mean/max/std stop error |

## 12. Legacy hardware hiện có

Legacy RTL metadata và tài liệu cũ không thống nhất về DSP/clock/performance.
Không được tái sử dụng các con số đó cho luận văn ArUco. Có thể tái sử dụng có
chọn lọc:

- cách tạo HLS IP;
- AXI-Lite control pattern;
- `pynq.allocate()` và cache coherency;
- Vivado PS–PL integration;
- `.bit`/`.hwh` overlay loading.

Không tái sử dụng trực tiếp kiến trúc Conv–Pool–FC, cat/dog weights hoặc
activation-hotspot bounding box.
