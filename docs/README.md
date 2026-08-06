# Documentation index

## Project hiện tại

| Tài liệu | Nội dung |
|---|---|
| [System overview](system_overview_aruco_robot.md) | Tổng quan PoC và kiến trúc FPGA đích |
| [Draw.io architecture](diagrams/aruco_robot_system_architecture.drawio) | Hai sơ đồ chỉnh sửa được |
| [System architecture](system_architecture.md) | Component, interface và runtime dataflow |
| [Software design](software_design.md) | Model, detector backends, controller, logging |
| [Hardware design](hardware_design.md) | Accelerator MobileNetV2 INT8 dự kiến |
| [User manual](user_manual.md) | Webcam PoC, dataset, train, evaluate, export |
| [Performance analysis](performance_analysis.md) | Benchmark methodology CPU/FPGA/robot |
| [Results](results.md) | Kết quả đã xác minh và bảng TBD |
| [PoC thesis methods/results](thesis_poc_methods_results.md) | Nội dung và bảng có thể đưa vào khóa luận |
| [PoC smoke report](poc_smoke_report_2026-08-04.md) | Ví dụ số liệu tổng hợp, chỉ dùng kiểm tra pipeline |
| [Round-trip demo workflow](../poc/ROUNDTRIP_DEMO.md) | Demo start-target-start, state machine và chỉ tiêu đo |
| [Round-trip thesis figures](thesis_figures_roundtrip.md) | Hình PNG, caption và lưu ý sử dụng trong khóa luận |
| [Experiment runner](../experiments/README.md) | Chạy nhiều trial, gom raw log, metadata và bảng Prism |
| [Dummy backend results](dummy_backend_results.md) | Số liệu minh họa có watermark và cách thay bằng phép đo thật |

## Tài liệu module

- [Model guide](../model/README.md)
- [PoC guide](../poc/README.md)
- [ArUco dataset guide](../dataset/aruco/README.md)
- [Thesis data export](../analysis/README.md)

## Legacy references

Tài sản cat/dog classifier trước đây đã được loại khỏi working tree để repository
chỉ chứa source và tài liệu ArUco hiện tại. Chúng vẫn còn trong Git history nhưng
không phải kết quả hoặc implementation của MobileNetV2 ArUco.
