# Thesis data export

Chuyển một hoặc nhiều log PoC thành raw/summary tables:

```powershell
python analysis/export_thesis_tables.py `
  --run classical=artifacts/runs/classical.csv `
  --run cnn=artifacts/runs/cnn.csv `
  --run hybrid=artifacts/runs/hybrid.csv `
  --drop-warmup 5 `
  --output artifacts/thesis_tables
```

Output:

- `poc_raw_long.csv`: toàn bộ frame để audit/thống kê;
- `poc_summary.csv`: mean/median/SD/P95/min/max;
- `latency_fps_wide.csv`: mỗi backend là một cột, tiện nhập phần mềm vẽ biểu đồ;
- `state_counts.csv`: số frame theo state;
- `stop_trials.csv`: first stop của mỗi trial;
- `analysis_manifest.json`: SHA-256 của raw input.

## Nhiệm vụ khứ hồi

Exporter cũng nhận log `--mission roundtrip`. Ngoài các trường dừng một chiều,
`poc_summary.csv` và `stop_trials.csv` có thêm `first_target_frame`,
`first_target_distance_m`, `home_complete_frame`, `home_distance_m`,
`time_to_target_s`, `return_time_s`, `mission_duration_s`, `mission_completed`
và `start_frame_rate`. `state_counts.csv` chứa state của cả hai loại nhiệm vụ.

```powershell
python analysis/export_thesis_tables.py `
  --run roundtrip=artifacts/webcam_poc/roundtrip_metrics.csv `
  --output artifacts/thesis_tables/roundtrip
```

Mỗi run từ `live_webcam_demo.py` còn có sidecar `*.meta.json` chứa CLI arguments,
version môi trường, camera properties và SHA-256 checkpoint/input/calibration.

Không chỉnh sửa raw CSV bằng tay. Nếu cần đổi warm-up hoặc loại outlier, chạy
lại script và ghi rõ rule trong phương pháp nghiên cứu.
