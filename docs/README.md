# Documentation index

## Project hiện tại

| Tài liệu | Nội dung |
|---|---|
| [System overview](system_overview_aruco_robot.md) | Tổng quan PoC và kiến trúc FPGA đích |
| [System architecture](system_architecture.md) | Component, interface và runtime dataflow |
| [Software design](software_design.md) | Model, detector backends, controller, logging |
| [Hardware design](hardware_design.md) | Accelerator MobileNetV2 INT8 dự kiến |
| [User manual](user_manual.md) | Webcam PoC, dataset, train, evaluate, export |
| [Performance analysis](performance_analysis.md) | Benchmark methodology CPU/FPGA/robot |
| [Results](results.md) | Kết quả đã xác minh và bảng TBD |
| [PoC thesis methods/results](thesis_poc_methods_results.md) | Nội dung và bảng có thể đưa vào khóa luận |
| [Latest measured results 2026-08-10](thesis_latest_measurements_2026-08-10.md) | Một file tổng hợp số liệu train, test, camera round-trip, calibration và INT8 export |
| [PoC smoke report](poc_smoke_report_2026-08-04.md) | Ví dụ số liệu tổng hợp, chỉ dùng kiểm tra pipeline |
| [Round-trip demo workflow](../poc/ROUNDTRIP_DEMO.md) | Demo start-target-start, state machine và chỉ tiêu đo |
| [Round-trip thesis figures](thesis_figures_roundtrip.md) | Hình PNG, caption và lưu ý sử dụng trong khóa luận |
| [Experiment runner](../experiments/README.md) | Chạy nhiều trial, gom raw log, metadata và bảng tổng hợp |

## Tài liệu module

- [Model guide](../model/README.md)
- [PoC guide](../poc/README.md)
- [ArUco dataset guide](../dataset/aruco/README.md)
- [Thesis data export](../analysis/README.md)

## Legacy references

Tài sản cat/dog classifier trước đây đã được loại khỏi working tree để repository
chỉ chứa source và tài liệu ArUco hiện tại. Chúng vẫn còn trong Git history nhưng
không phải kết quả hoặc implementation của MobileNetV2 ArUco.
