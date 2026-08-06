# Bộ hình PoC khứ hồi dùng cho khóa luận

Các hình được trích từ lần chạy workflow video tổng hợp với detector OpenCV
Classical, marker xuất phát ID 0 và marker đích ID 1. Đây là hình minh họa chức
năng phần mềm PoC, không phải bằng chứng FPS, accuracy hay chuyển động robot thật.

## 1. Hình nên dùng trong nội dung chính

### Trình tự sáu trạng thái

File: `artifacts/thesis_figures/roundtrip/figure_roundtrip_state_sequence.png`

Caption đề xuất:

> **Hình X. Trình tự hoạt động của nhiệm vụ giao hàng khứ hồi trong PoC.** Hệ
> thống xác nhận marker xuất phát ID 0, di chuyển logic tới marker đích ID 1, dừng
> tại đích, thực hiện pha quay đầu, tìm lại marker xuất phát và khóa trạng thái
> `HOME_COMPLETE`. Các frame được lấy từ video tổng hợp để kiểm chứng pipeline và
> state machine khi chưa có robot vật lý.

Vị trí phù hợp: chương Thiết kế hệ thống hoặc phần Demo PoC.

### So sánh thời điểm tới đích và trở về trạm

File: `artifacts/thesis_figures/roundtrip/figure_target_and_home_comparison.png`

Caption đề xuất:

> **Hình X. Kết quả nhận diện tại hai điểm dừng của nhiệm vụ khứ hồi.** Marker đích
> ID 1 được xác nhận tại frame 61 (`AT_TARGET`), trong khi marker xuất phát ID 0
> được xác nhận lại tại frame 156 (`HOME_COMPLETE`). Khung màu vàng biểu diễn
> marker đích và khung màu xanh biểu diễn marker xuất phát.

Vị trí phù hợp: chương Kết quả thực nghiệm PoC.

### Timeline state machine

File: `artifacts/thesis_figures/roundtrip/figure_roundtrip_state_timeline.png`

Caption đề xuất:

> **Hình X. Timeline trạng thái của một lần chạy nhiệm vụ khứ hồi gồm 180 frame.**
> Controller lần lượt trải qua `WAITING_FOR_START`, `OUTBOUND`, `AT_TARGET`,
> `TURNING_HOME`, `RETURNING` và `HOME_COMPLETE`; không xảy ra chuyển trạng thái
> ngược hoặc bỏ qua trạng thái bắt buộc.

Vị trí phù hợp: phần kiểm chứng state machine. Hình này thể hiện số frame, không
được diễn giải thành thời gian di chuyển thật vì video được xử lý offline.

## 2. Frame riêng cho phụ lục hoặc mô tả từng bước

| File | Nội dung |
|---|---|
| `01_waiting_for_start_f000.png` | Nhận marker xuất phát ID 0 và bắt đầu xác nhận |
| `02_outbound_f031.png` | Phát hiện marker đích ID 1 trong chặng đi |
| `03_at_target_f061.png` | Dừng tại marker đích |
| `04_turning_home_f091.png` | Pha quay đầu; không yêu cầu marker trong ảnh |
| `05_returning_f131.png` | Nhận lại marker start trong chặng về |
| `06_home_complete_f156.png` | Dừng và hoàn thành nhiệm vụ tại trạm |

Các frame riêng giữ overlay runtime để audit. Giá trị FPS trên overlay là tốc độ xử
lý video file headless, không phải FPS webcam thời gian thực và không được dùng ở
bảng hiệu năng.

## 3. Cách tái tạo từ một lần chạy khác

Sau khi có video kết quả và CSV tương ứng:

```powershell
python poc/extract_thesis_figures.py `
  --video artifacts/runs/roundtrip_trial01.avi `
  --csv artifacts/runs/roundtrip_trial01.csv `
  --output-dir artifacts/thesis_figures/roundtrip_trial01
```

Script tự chọn frame đại diện dựa trên cột `state`, không dựa trên số frame hard-code.
Nếu thiếu bất kỳ state nào trong sáu state bắt buộc, script dừng và báo lỗi.

## 4. Quy tắc sử dụng trong khóa luận

- Dùng file PNG gốc, không chụp lại từ màn hình hoặc nén sang JPEG.
- Ghi trong caption rằng input là video tổng hợp và detector là Classical.
- Không dùng hình này để tuyên bố hiệu năng CNN, FPGA hoặc độ chính xác robot thật.
- Khi có webcam/robot thật, chạy lại extractor trên video trial thật và giữ cùng
  bố cục để so sánh trực tiếp.
- Với hình tự tạo từ chương trình của đề tài, ghi “Nguồn: tác giả” nếu biểu mẫu của
  trường yêu cầu nguồn hình.
