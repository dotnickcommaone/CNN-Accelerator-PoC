# Bộ số liệu dummy so sánh backend

## Trạng thái dữ liệu

Các số liệu tại `artifacts/dummy_results/backend_comparison/` là **giả lập**, chỉ
dùng để thử bố cục bảng, biểu đồ, pipeline Prism và viết bản nháp. Chúng không phải
kết quả đo từ webcam, checkpoint chính thức hoặc FPGA.

- `DUMMY_SIMULATED`: dữ liệu sinh theo xác suất giả định bằng seed cố định.
- `PROJECTED_NOT_MEASURED`: dự phóng FPGA, hoàn toàn chưa được tổng hợp/đo trên board.

Mọi CSV đều có cột `data_label`; mọi biểu đồ đều có watermark cảnh báo.

## Giả định

- 1.000 frame, gồm 800 positive và 200 negative;
- marker ID 0-9 cân bằng;
- `DICT_4X4_50`;
- năm điều kiện: bình thường, thiếu sáng, motion blur, góc nghiêng và marker nhỏ;
- CNN MobileNetV2-0.35 chỉ phát hiện ROI;
- OpenCV ArUco tiếp tục đảm nhiệm giải mã ID;
- accuracy FPGA được sao chép từ INT8 CPU, còn latency/power là dự phóng.

## Bảng dummy với seed mặc định 20260806

| Backend | Precision | Recall | F1 | Correct ID end-to-end | Latency mean | Processing FPS | Power dummy |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCV Classical CPU | 99,1% | 85,5% | 91,8% | 84,1% | 4,30 ms | 233,2 | 18,0 W |
| MobileNetV2-0.35 FP32 CPU | 94,3% | 94,6% | 94,4% | 93,8% | 14,77 ms | 68,1 | 26,1 W |
| MobileNetV2-0.35 INT8 CPU | 94,1% | 93,8% | 93,9% | 92,8% | 10,09 ms | 99,6 | 22,0 W |
| Hybrid CNN + fallback CPU | 96,4% | 96,8% | 96,6% | 96,0% | 16,81 ms | 59,9 | 28,0 W |
| MobileNetV2-0.35 INT8 FPGA | 94,1% | 93,8% | 93,9% | 92,8% | 8,97 ms | 111,9 | 6,5 W |

Power ở bảng trên chỉ là placeholder và khác measurement boundary giữa laptop với
board. Không dùng để tính energy efficiency trong kết quả chính thức.

## Cách diễn giải trong bản nháp

Có thể dùng đoạn sau nhưng phải giữ từ “giả định/dự kiến”:

> Bộ dữ liệu giả lập minh họa trade-off dự kiến giữa độ bền nhận diện và tốc độ xử
> lý. OpenCV Classical có latency thấp nhất nhưng recall suy giảm trong các điều
> kiện khó. MobileNetV2-0.35 FP32 cải thiện recall và tỷ lệ giải mã ID end-to-end,
> đổi lại latency tăng. Quantization INT8 giả định làm giảm khoảng 0,9 điểm phần
> trăm recall so với FP32, đồng thời giảm khoảng 31,7% latency CPU và 73,4% dung
> lượng weight. Hybrid đạt recall cao nhất nhờ fallback nhưng là backend CPU chậm
> nhất. Backend FPGA INT8 chỉ là dự phóng và phải được thay bằng báo cáo synthesis,
> power measurement và benchmark trên board.

Không viết “kết quả chứng minh” hoặc “thực nghiệm cho thấy” với bảng này.

## File cho Prism và biểu đồ

- `backend_summary_dummy.csv`: bảng tổng hợp;
- `accuracy_by_condition_dummy.csv`: grouped table backend × condition;
- `latency_prism_wide_dummy.csv`: latency và FPS dạng wide;
- `frame_predictions_dummy.csv`: raw data 5.000 row;
- `id_confusion_matrix_dummy.csv`: confusion matrix ID;
- `figure_accuracy_comparison_dummy.png`;
- `figure_latency_comparison_dummy.png`;
- `figure_fps_comparison_dummy.png`.

## Dummy AP50 và PR curve

| Backend | AP50 dummy | Trạng thái |
|---|---:|---|
| MobileNetV2-0.35 FP32 CPU | 0,958 | `DUMMY_SIMULATED` |
| MobileNetV2-0.35 INT8 CPU | 0,950 | `DUMMY_SIMULATED` |
| Hybrid CNN + ArUco fallback | 0,978 | `DUMMY_SIMULATED` |
| MobileNetV2-0.35 INT8 FPGA | 0,950 | `PROJECTED_NOT_MEASURED` |

OpenCV Classical không có AP50 vì detector không sinh scored bounding boxes theo
cùng giao thức với model học sâu. Dùng Precision/Recall detection để so sánh
Classical, không tự gán AP50 cho nó.

File: `ap50_summary_dummy.csv`, `pr_curve_dummy.csv` và
`figure_pr_curve_dummy.png`.

## Dummy fallback

Trên 1.000 frame giả lập:

- fallback được gọi ở 22,8% tổng số frame;
- fallback được gọi ở 5,4% positive frame;
- phục hồi được 34,9% trường hợp CNN primary miss;
- recall tăng từ 94,6% lên 96,5%;
- tỷ lệ ID đúng end-to-end sau fallback đạt 95,6%;
- overhead fallback trung bình khoảng 0,63 ms;
- latency hybrid trung bình khoảng 16,20 ms.

File: `fallback_raw_dummy.csv`, `fallback_summary_dummy.csv` và
`figure_fallback_recall_dummy.png`.

## Dummy tài nguyên FPGA

Các số dưới đây dùng capacity giả định của PYNQ-Z2/XC7Z020 và chưa chạy Vivado:

| Candidate | LUT | BRAM | DSP | Timing 125 MHz | FPS dummy | Power dummy |
|---|---:|---:|---:|---|---:|---:|
| HLS INT8 PE4 | 35,5% | 41,4% | 32,7% | Pass | 36,0 | 4,1 W |
| HLS INT8 PE8 balanced | 53,9% | 58,6% | 61,8% | Pass | 73,0 | 5,5 W |
| HLS INT8 PE12 | 78,8% | 80,7% | 92,7% | Fail | 98,0 | 7,0 W |
| DPU-lite candidate | 60,3% | 67,1% | 72,7% | Fail | 78,1 | 6,3 W |

PE8 được dựng làm phương án cân bằng minh họa. PE12 nhanh hơn nhưng giả định không
đạt timing và gần hết DSP; đây không phải kết luận synthesis.

File: `fpga_resource_estimates_dummy.csv` và
`figure_fpga_resource_utilization_dummy.png`.

## Dummy kết quả robot

Mỗi backend có 30 trial giả lập trên tuyến dài 5 m:

| Backend | Mission complete | Target MAE | Home MAE | Sai số góc quay TB |
|---|---:|---:|---:|---:|
| OpenCV Classical CPU | 83,3% | 5,12 cm | 5,18 cm | 5,86° |
| Hybrid FP32 CPU | 93,3% | 2,49 cm | 2,32 cm | 3,11° |
| MobileNetV2 INT8 FPGA | 93,3% | 2,84 cm | 3,76 cm | 3,92° |

Kết quả FPGA và toàn bộ chuyển động robot đều chưa đo, dù mang hình thức raw trial.
File: `robot_trials_raw_dummy.csv`, `robot_results_summary_dummy.csv`,
`robot_trials_prism_wide_dummy.csv` và `figure_robot_mission_results_dummy.png`.

## Sinh lại

```powershell
python analysis/generate_dummy_backend_results.py `
  --seed 20260806 `
  --output artifacts/dummy_results/backend_comparison

python analysis/generate_dummy_extended_results.py `
  --seed 20260806 --robot-trials 30 `
  --output artifacts/dummy_results/backend_comparison
```

Đổi seed sẽ làm các TP/FP/FN thay đổi. Giữ seed trong `dummy_manifest.json` để bản
nháp có thể tái tạo.

## Thay thế bằng dữ liệu thật

1. Dùng `model/evaluate.py` trên test set độc lập để lấy Precision, Recall, F1,
   AP50, prediction và PR curve.
2. Dùng `experiments/run_poc_experiment.py` với cùng video/camera cho từng backend.
3. Dùng `analysis/export_thesis_tables.py` hoặc
   `experiments/summarize_experiment.py` để lấy latency/FPS raw và summary.
4. Chỉ thêm FPGA khi có bitstream đúng model, synthesis report và phép đo board.
5. Đổi nhãn sang `MEASURED` chỉ sau khi kiểm tra metadata, checkpoint SHA-256 và
   điều kiện benchmark.
