# Workflow demo khứ hồi bằng hai marker

## 1. Mục tiêu

PoC thực hiện một nhiệm vụ khép kín:

```text
marker xuất phát (start ID)
  -> xác nhận điểm xuất phát
  -> tìm và tiến tới marker đích (target ID)
  -> dừng tại đích
  -> quay đầu 180 độ
  -> tìm và tiến về marker xuất phát
  -> dừng, khóa trạng thái hoàn tất
```

Trong PoC laptop, `linear_speed` và `angular_speed` là lệnh chuẩn hóa trong
`[-1, 1]`, chưa được gửi tới motor thật. Pipeline nhận diện, quyết định, overlay,
CSV và các điều kiện an toàn có thể kiểm thử đầy đủ mà không cần FPGA hoặc robot.

## 2. Quy ước marker và bố trí

- Marker xuất phát: ID 0, đặt tại trạm robot, mặt marker hướng về hành lang trở về.
- Marker đích: ID 1, đặt tại phòng giao hàng, mặt marker hướng về hướng robot đi tới.
- Hai ID bắt buộc khác nhau và thuộc `DICT_4X4_50`.
- Nên in hai marker cùng kích thước vật lý, giữ viền trắng và đặt ngang tầm camera.

Sau khi robot quay 180 độ tại đích, marker xuất phát phải nằm trong trường nhìn hoặc
có thể xuất hiện khi robot tiến theo hành lang. PoC hiện dùng thời gian/số frame để
quay đầu; robot thật cần hiệu chỉnh `turn_frames` theo tốc độ quay, hoặc thay bằng
encoder/IMU để đạt góc quay chính xác hơn.

## 3. State machine

| State | Marker đang theo dõi | Lệnh chính | Điều kiện chuyển state |
|---|---|---|---|
| `WAITING_FOR_START` | start | dừng | thấy start đủ `start_confirm_frames` |
| `OUTBOUND` | target | tiến/tìm target | target đủ gần trong `confirm_frames` |
| `AT_TARGET` | target | dừng | chờ đủ `target_dwell_frames` |
| `TURNING_HOME` | start | quay tại chỗ | đủ `turn_frames` |
| `RETURNING` | start | tiến/tìm start | start đủ gần trong `confirm_frames` |
| `HOME_COMPLETE` | start | dừng khóa | nhấn `R` để chạy nhiệm vụ mới |

Ngưỡng ưu tiên khoảng cách mét nếu có calibration. Nếu không có, hệ thống dùng
`side_px`: marker càng lớn thì robot càng gần. Median window và nhiều frame xác nhận
giảm dừng nhầm do một frame nhiễu.

## 4. Demo hoàn toàn offline

Chạy toàn bộ workflow bằng một lệnh (tạo video, chạy PoC, kiểm tra sáu state và
xuất bảng luận văn):

```powershell
python poc/run_roundtrip_workflow.py
```

Kết quả nằm trong `artifacts/webcam_poc/roundtrip_workflow/`. Workflow chỉ báo
`PASS` khi state đúng thứ tự và frame cuối có `mission_complete=1`.

Các bước tương đương để chạy riêng lẻ như sau.

Tạo video chứa đủ hai chặng:

```powershell
python poc/make_roundtrip_demo_video.py `
  --start-id 0 --target-id 1 `
  --output artifacts/webcam_poc/roundtrip_input.avi
```

Chạy pipeline và ghi video kết quả:

```powershell
python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/roundtrip_input.avi `
  --mode classical --mission roundtrip `
  --start-id 0 --target-id 1 --headless `
  --output-video artifacts/webcam_poc/roundtrip_result.avi `
  --csv artifacts/webcam_poc/roundtrip_metrics.csv
```

Kiểm tra nhanh kết quả:

```powershell
Import-Csv artifacts/webcam_poc/roundtrip_metrics.csv |
  Group-Object state | Select-Object Name, Count
```

Một lần chạy hợp lệ phải có đủ sáu state và kết thúc bằng `HOME_COMPLETE=1` trong
cột `mission_complete`. File sidecar `roundtrip_metrics.meta.json` phải có
`mission_completed: true`, `first_target_frame` và `home_complete_frame` khác null.

## 5. Demo bằng USB webcam

Đặt marker start trước camera, bắt đầu chương trình, sau đó lần lượt đưa target lại
gần camera, che target trong giai đoạn quay, rồi đưa start lại gần camera:

```powershell
python poc/live_webcam_demo.py --source 0 `
  --mode classical --mission roundtrip `
  --start-id 0 --target-id 1 `
  --stop-side-px 180 --slow-side-px 100 `
  --target-dwell-frames 30 --turn-frames 60 `
  --csv artifacts/runs/roundtrip_webcam_trial01.csv `
  --output-video artifacts/runs/roundtrip_webcam_trial01.avi
```

Ở camera 30 FPS, `turn_frames=60` tương ứng xấp xỉ 2 giây. Đây chỉ là giá trị khởi
đầu; phải đo thực nghiệm với cơ cấu robot. Nhấn `R` để reset nhiệm vụ, `S` để lưu
snapshot và `Q`/`Esc` để thoát.

Nếu đã hiệu chuẩn camera, thêm:

```powershell
--marker-size-m 0.10 --camera-calibration artifacts/calibration/camera.npz
```

## 6. Dữ liệu cần thu cho khóa luận

Để tự động tạo thư mục riêng cho từng experiment và từng trial, nên dùng runner
thống nhất thay vì đặt tên CSV/video thủ công:

```powershell
python experiments/run_poc_experiment.py `
  --source 0 --name webcam_classical_roomA `
  --mission roundtrip --mode classical `
  --start-id 0 --target-id 1 --trials 10
```

Xem [hướng dẫn experiment và logging](../experiments/README.md).

Mỗi cấu hình nên chạy ít nhất 10 lần và không ghi đè CSV. Ghi lại:

- tỷ lệ nhiệm vụ hoàn tất: số trial có `HOME_COMPLETE` / tổng trial;
- frame/thời gian tới đích và frame/thời gian trở về start;
- latency vision, latency toàn pipeline, FPS;
- khoảng cách hoặc `side_px` khi dừng ở target và start;
- số frame trong mỗi state;
- số lần mất marker, dừng nhầm hoặc không hoàn tất;
- với robot thật: sai số vị trí dừng và sai số góc sau khi quay.

Xuất bảng tổng hợp:

```powershell
python analysis/export_thesis_tables.py `
  --run trial01=artifacts/runs/roundtrip_webcam_trial01.csv `
  --run trial02=artifacts/runs/roundtrip_webcam_trial02.csv `
  --output artifacts/thesis_tables/roundtrip
```

`poc_summary.csv`, `state_counts.csv`, `stop_trials.csv` và
`latency_fps_wide.csv` có thể nhập trực tiếp vào Excel, Prism hoặc phần mềm thống kê.
Các cột thời gian nhiệm vụ được tính từ timestamp thực. Không dùng chúng làm thời
gian di chuyển robot khi chạy video file ở chế độ headless, vì file được xử lý nhanh
hơn thời gian thực; chỉ dùng số này ở trial webcam/robot chạy theo thời gian thật.

## 7. Ánh xạ sang robot thật sau này

Driver motor chỉ nên nhận `MissionCommand` sau một lớp safety gate:

```text
linear_speed  > 0, angular_speed = 0  -> hai bánh chạy tiến
linear_speed  = 0, angular_speed > 0  -> hai bánh quay ngược chiều
linear_speed  = 0, angular_speed = 0  -> brake/stop
HOME_COMPLETE                         -> khóa motor cho đến reset
```

Cần bổ sung watchdog mất frame/camera, emergency stop, giới hạn PWM, encoder hoặc
IMU và kiểm thử trên giá kê bánh trước khi cho robot chạy trên sàn. FPGA sau này chỉ
thay backend inference; state machine nhiệm vụ và định dạng log được giữ nguyên để
đối chiếu công bằng giữa CPU và FPGA.
