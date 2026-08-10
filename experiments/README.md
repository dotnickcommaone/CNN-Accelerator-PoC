# Bộ script thu thập số liệu PoC/khóa luận

## 1. Mục đích

`run_poc_experiment.py` gom các bước tạo/chọn input, chạy detector, ghi video,
ghi raw metrics, trích hình và xuất bảng thống kê vào **một thư mục experiment có
timestamp**. `summarize_experiment.py` tính lại toàn bộ bảng derived từ raw CSV.

Không sửa `trials/*/metrics.csv` bằng tay. Các phép đo không thể tự động khi chưa có
robot/board được để trống trong template, không sinh số giả.

## 2. Chạy kiểm chứng offline

```powershell
python experiments/run_poc_experiment.py `
  --source synthetic --name offline_roundtrip `
  --mission roundtrip --mode classical `
  --start-id 0 --target-id 1 --trials 3
```

Lệnh này không cần webcam, checkpoint, robot hoặc FPGA. FPS thu được là throughput
xử lý file offline, không phải FPS camera thời gian thực. Các trường
`source_kind=synthetic_video` và `motion_time_valid=0` được ghi để tránh sử dụng
sai trong báo cáo.

## 3. Thu trial USB webcam

Ví dụ 10 trial Classical:

```powershell
python experiments/run_poc_experiment.py `
  --source 0 --name webcam_classical_roomA `
  --mission roundtrip --mode classical `
  --start-id 0 --target-id 1 --trials 10 `
  --camera-fps 30 --width 640 --height 480
```

Trước mỗi trial, runner chờ Enter. Khi controller đạt `STOPPED` hoặc
`HOME_COMPLETE`, trial tự đóng sau 15 frame để bảo toàn bằng chứng trạng thái cuối.
Nhấn `Q`/`Esc` nếu cần kết thúc một trial không hoàn tất.

Nếu đã hiệu chuẩn camera:

```powershell
--marker-size-m 0.10 --camera-calibration artifacts/calibration/camera.npz
```

CNN/Hybrid:

```powershell
python experiments/run_poc_experiment.py `
  --source 0 --name webcam_hybrid_roomA --trials 10 `
  --mode hybrid --mission roundtrip --start-id 0 --target-id 1 `
  --checkpoint artifacts/aruco_mbv2_035/best.pt
```

## 4. Gộp cả dataset audit và model accuracy

Khi có checkpoint chính thức và test set độc lập:

```powershell
python experiments/run_poc_experiment.py `
  --source 0 --name thesis_hybrid_complete --trials 10 `
  --mode hybrid --mission roundtrip --start-id 0 --target-id 1 `
  --checkpoint artifacts/aruco_mbv2_035/best.pt `
  --dataset-root dataset/aruco `
  --audit-dataset --hash-images --evaluate-model --evaluation-split test
```

Các output bổ sung nằm trong `dataset_audit/` và `model_evaluation/`. Dataset audit
sẽ fail nếu phát hiện label lỗi, split overlap hoặc manifest trỏ tới file thiếu.

## 5. Cấu trúc kết quả

```text
artifacts/experiments/<timestamp>_<name>/
├── experiment_manifest.json       # CLI, version, git commit/status, trạng thái
├── experiment_summary.json        # tổng hợp toàn experiment
├── runner.log                     # toàn bộ console output có timestamp
├── commands.log                   # mọi command đã chạy
├── manual_trial_measurements.csv  # sai số/power/góc quay nhập sau phép đo thật
├── hardware_measurements.csv      # LUT/FF/BRAM/DSP/power cho FPGA sau này
├── trials/
│   └── trial_001/
│       ├── metrics.csv            # raw từng frame
│       ├── metrics.meta.json      # cấu hình và provenance
│       ├── console.log
│       ├── input.avi              # chỉ có khi source synthetic
│       ├── annotated.avi
│       ├── snapshots/
│       └── figures/
├── analysis/
│   ├── trial_summary.csv          # một row cho mỗi trial
│   ├── latency_fps_wide.csv       # định dạng wide cho latency/FPS
│   ├── poc_raw_long.csv
│   ├── poc_summary.csv
│   ├── state_counts.csv
│   ├── stop_trials.csv
│   ├── data_quality.csv
│   └── analysis_manifest.json
├── dataset_audit/                 # tùy chọn
└── model_evaluation/              # tùy chọn
```

## 6. Số liệu được lấy tự động

| Nhóm | Trường chính |
|---|---|
| Hiệu năng | vision/total latency mean, median, SD, P95, min, max; FPS |
| Detection | marker IDs, target/start visible rate, CNN/fallback source rate |
| Khoảng cách | solvePnP/pinhole distance, pose XYZ hoặc `side_px` fallback |
| Nhiệm vụ | state sequence/count, first target frame, home-complete frame |
| Thời gian | time-to-target, return time, mission duration |
| Chất lượng trial | complete/incomplete/invalid sequence, metadata/video present |
| Model tùy chọn | Precision, Recall, F1, AP50, model latency/FPS, raw predictions, PR curve |
| Dataset tùy chọn | số image/box/negative, split, overlap, lỗi label, duplicate hash |

Thời gian chuyển động chỉ hợp lệ khi `motion_time_valid=1`. Với video offline,
timestamp đo tốc độ xử lý máy tính chứ không mô tả robot di chuyển.

## 7. Số liệu cần nhập thủ công

Sau mỗi trial vật lý, điền `manual_trial_measurements.csv`:

- điều kiện và ánh sáng/lux;
- khoảng cách dừng thực tế ở target và home;
- sai số dừng target/home;
- sai số góc quay;
- công suất trung bình/đỉnh và năng lượng;
- dụng cụ đo;
- đánh giá success của người vận hành;
- số false stop, số lần mất marker và ghi chú.

Khi có FPGA, điền `hardware_measurements.csv`: board, clock, precision, LUT, FF,
BRAM, DSP, estimated/measured power. Không điền ước lượng vào cột measured power.

Sau khi nhập số đo thủ công, chạy lại:

```powershell
python experiments/summarize_experiment.py `
  artifacts/experiments/<timestamp>_<name> --drop-warmup 5
```

Script merge dữ liệu tự động và thủ công vào `analysis/trial_summary.csv`,
đồng thời cập nhật `experiment_summary.json`.
