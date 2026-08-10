# PoC nhận diện ArUco bằng USB webcam

PoC này chạy toàn bộ pipeline ứng dụng trên laptop trong giai đoạn chưa có board
FPGA:

```text
USB webcam/video
  -> phát hiện marker
  -> giải mã ArUco ID
  -> lọc khoảng cách hoặc kích thước biểu kiến
  -> máy trạng thái dừng robot hoặc thực hiện nhiệm vụ khứ hồi
  -> giao diện hiển thị + CSV benchmark
```

Giá trị `speed` hiển thị trên giao diện là lệnh motor mô phỏng đã chuẩn hóa. Chương
trình hiện chưa gửi lệnh serial/GPIO tới robot vật lý.

Chế độ khứ hồi tạo riêng lệnh vận tốc tuyến tính và vận tốc góc đã chuẩn hóa. Xem
[quy trình demo khứ hồi](ROUNDTRIP_DEMO.md) để chạy đầy đủ nhiệm vụ sử dụng hai marker.

## 1. In marker

Từ điển mặc định là `DICT_4X4_50`. In marker ID 0 với kích thước vật lý đã biết,
ví dụ cạnh marker dài 10 cm. Giữ vùng viền trắng xung quanh marker và không thay đổi
tỷ lệ khi in.

## 2. Demo ngay khi chưa có CNN đã huấn luyện

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --stop-side-px 180 --slow-side-px 100
```

Chế độ này kiểm chứng webcam, bộ giải mã ArUco, giao diện, logging và logic dừng
robot. Hệ thống dùng kích thước biểu kiến của marker làm đại lượng gần đúng cho
khoảng cách.

Các phím điều khiển:

- `Q` hoặc `Esc`: thoát chương trình;
- `R`: bỏ trạng thái STOP đã khóa và khởi tạo lại nhiệm vụ;
- `S`: lưu frame hiện tại kèm chú thích.

Kết quả được ghi vào `artifacts/webcam_poc/metrics.csv`.

## 3. Hiệu chuẩn khoảng cách gần đúng

Đặt marker có kích thước đã biết tại một khoảng cách đã đo từ camera. Ví dụ, với
marker 10 cm đặt cách camera 1 m:

```powershell
python poc/calibrate_focal_length.py --source 0 --marker-id 0 `
  --marker-size-m 0.10 --distance-m 1.0
```

Dùng focal length mà chương trình trả về:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --marker-size-m 0.10 --focal-length-px 615.2 `
  --slow-distance-m 0.90 --stop-distance-m 0.45
```

Mô hình pinhole gần đúng đủ dùng cho PoC. Thí nghiệm pose trong khóa luận nên dùng
camera matrix, hệ số méo và `solvePnP`.

### Hiệu chuẩn nội tại đầy đủ và solvePnP

Dùng bảng checkerboard có 9×6 góc trong và kích thước ô đã biết:

```powershell
python poc/calibrate_camera.py --source 0 `
  --board-cols 9 --board-rows 6 --square-size-m 0.025 `
  --samples 20 --output artifacts/calibration/camera.npz
```

Sau đó chạy:

```powershell
python poc/live_webcam_demo.py --source 0 --mode classical `
  --target-id 0 --marker-size-m 0.10 `
  --camera-calibration artifacts/calibration/camera.npz
```

CSV bổ sung các cột `distance_method`, `pose_x_m`, `pose_y_m` và `pose_z_m`. Khi
có file calibration, phương pháp `solvePnP` được ưu tiên hơn phép xấp xỉ chỉ dùng
focal length.

## 4. Chế độ CNN và Hybrid

Chỉ dùng CNN:

```powershell
python poc/live_webcam_demo.py --source 0 --mode cnn `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

Chế độ Hybrid:

```powershell
python poc/live_webcam_demo.py --source 0 --mode hybrid `
  --checkpoint artifacts/aruco_mbv2_035/best.pt --target-id 0
```

Hybrid chạy MobileNetV2-0.35 để phát hiện ROI trước. Nếu không giải mã được marker
trong ROI, hệ thống fallback sang OpenCV ArUco trên toàn bộ frame. Cột
`target_source` trong CSV ghi `cnn+opencv` hoặc `opencv`; nhờ đó tỷ lệ fallback có
thể được đo và không bị che giấu.

Không dùng checkpoint smoke test làm kết quả độ chính xác. Phải huấn luyện bằng dữ
liệu camera thật đã được kiểm tra trước khi đánh giá chế độ CNN hoặc Hybrid.

## 5. Demo offline không cần webcam

Tạo video mô phỏng marker đang tiến gần camera:

```powershell
python poc/make_demo_video.py --marker-id 0 `
  --output artifacts/webcam_poc/aruco_approach.avi
```

Chạy toàn bộ PoC không hiển thị cửa sổ:

```powershell
python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/aruco_approach.avi `
  --mode classical --target-id 0 --headless `
  --output-video artifacts/webcam_poc/result.avi
```

FPS đo từ video file trong chế độ `--headless` không phải kết quả camera thời gian
thực vì file được xử lý nhanh nhất có thể. Khi báo cáo FPS USB webcam, phải cố định
độ phân giải, backend và trạng thái bật/tắt giao diện.

## 6. Demo khứ hồi: điểm đầu → điểm đích → điểm đầu

Chạy quy trình offline đã có kiểm tra tự động bằng một lệnh:

```powershell
python poc/run_roundtrip_workflow.py
```

Hoặc chạy riêng từng giai đoạn:

```powershell
python poc/make_roundtrip_demo_video.py --start-id 0 --target-id 1 `
  --output artifacts/webcam_poc/roundtrip_input.avi

python poc/live_webcam_demo.py `
  --source artifacts/webcam_poc/roundtrip_input.avi `
  --mode classical --mission roundtrip --start-id 0 --target-id 1 `
  --headless --output-video artifacts/webcam_poc/roundtrip_result.avi `
  --csv artifacts/webcam_poc/roundtrip_metrics.csv
```

Trạng thái kết thúc mong đợi là `HOME_COMPLETE`. Khi dùng webcam, thay nguồn bằng
`--source 0` và bỏ `--headless`. PoC hiện xác định thời gian quay theo số frame;
khi có robot vật lý phải hiệu chỉnh `--turn-frames` hoặc thay bằng phản hồi từ
encoder/IMU.

### Chạy khứ hồi với camera calibration

```powershell
python poc/live_webcam_demo.py `
  --source 0 `
  --mode classical `
  --mission roundtrip `
  --start-id 0 --target-id 1 `
  --camera-calibration artifacts/calibration/camera.npz `
  --marker-size-m 0.10 `
  --slow-distance-m 0.90 --stop-distance-m 0.45 `
  --start-confirm-frames 3 --confirm-frames 3 `
  --target-dwell-frames 15 --turn-frames 30 `
  --exit-on-complete `
  --csv artifacts/runs/roundtrip_calibrated_trial01.csv
```

Phải dùng file calibration `.npz`. Giá trị `--marker-size-m` phải bằng kích thước
cạnh marker được đo ngoài thực tế.

## 7. Các trường dữ liệu được ghi

CSV chứa:

- vision latency và total latency;
- FPS đã được làm mượt theo hàm mũ;
- các marker ID phát hiện được;
- nguồn detector;
- khoảng cách hoặc độ dài cạnh marker theo pixel;
- trạng thái dừng (`SEARCHING`, `APPROACHING`, `SLOWING`, `STOPPED`);
- trạng thái khứ hồi từ `WAITING_FOR_START` đến `HOME_COMPLETE`;
- trạng thái nhìn thấy, khoảng cách, pose và độ dài cạnh của marker start/target;
- marker ID đang được theo dõi và các cờ mốc nhiệm vụ;
- vận tốc tuyến tính và vận tốc góc mô phỏng đã chuẩn hóa.
