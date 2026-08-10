# Hướng dẫn sử dụng bộ sơ đồ kiến trúc trong khóa luận

File nguồn chỉnh sửa: [thesis_architecture_diagrams.drawio](diagrams/thesis_architecture_diagrams.drawio).

File gồm năm trang độc lập. Mỗi trang được thiết kế ở tỷ lệ ngang, ít khối và có thể
export riêng từ diagrams.net bằng `File -> Export as -> SVG/PNG`.

| Trang Draw.IO | Vị trí đề xuất trong khóa luận | Caption đề xuất |
|---|---|---|
| 01 - Tổng quan PoC | Chương thiết kế hệ thống | Kiến trúc tổng quan PoC nhận diện marker ArUco cho robot giao hàng indoor. |
| 02 - Pipeline nhận diện | Phần thiết kế phần mềm/CNN | Luồng phát hiện ROI bằng MobileNetV2-0.35 và giải mã marker bằng OpenCV ArUco. |
| 03 - State machine khứ hồi | Phần thuật toán điều khiển robot | Máy trạng thái của nhiệm vụ di chuyển từ điểm đầu đến marker đích và quay lại điểm đầu. |
| 04 - Kiến trúc FPGA đích | Phần thiết kế phần cứng | Kiến trúc hardware/software co-design dự kiến trên PYNQ-Z2. |
| 05 - Quy trình thực nghiệm | Chương thực nghiệm và đánh giá | Quy trình tạo dữ liệu, huấn luyện, đánh giá, chạy PoC và xuất số liệu phân tích. |

## Quy ước trình bày

- Màu xanh lá: thành phần xử lý đã có trong PoC.
- Màu xanh dương: camera, dữ liệu hoặc thông tin quan sát.
- Màu cam: điều khiển robot.
- Màu vàng: đánh giá, logging và đầu ra Prism.
- Màu tím hoặc nét đứt: thành phần FPGA/robot vật lý chưa được triển khai.

## Thiết lập export đề xuất

- Với bản điện tử: ưu tiên SVG, bật `Crop` và giữ font được nhúng.
- Với PNG: đặt chiều rộng tối thiểu 2400 px hoặc độ phân giải tương đương 300 DPI.
- Chỉ export trang đang chọn để mỗi hình có thể đánh số và đặt caption riêng.
- Không bỏ ghi chú “chưa triển khai” trên hình FPGA nếu chưa có kết quả từ board thật.
