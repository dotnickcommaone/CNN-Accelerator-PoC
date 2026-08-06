from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


CONDITION_COUNTS = {
    "normal": 400,
    "low_light": 150,
    "motion_blur": 150,
    "oblique_view": 150,
    "small_marker": 150,
}


@dataclass(frozen=True)
class BackendSpec:
    key: str
    display_name: str
    platform: str
    precision: str
    parameters: int | None
    model_size_mb: float | None
    recall_by_condition: dict[str, float]
    id_accuracy_by_condition: dict[str, float]
    negative_false_positive_rate: float
    ap50_dummy: float | None
    roi_ms: float
    decode_ms: float
    total_ms: float
    active_power_w: float
    result_origin: str = "DUMMY_SIMULATED"
    copy_predictions_from: str | None = None
    copy_drop_rate: float = 0.0
    copy_extra_false_positive_rate: float = 0.0
    copy_id_error_rate: float = 0.0


BACKENDS = (
    BackendSpec(
        "opencv_classical_cpu",
        "OpenCV ArUco / CPU",
        "Laptop CPU",
        "FP32/classical",
        None,
        None,
        {"normal": 0.965, "low_light": 0.820, "motion_blur": 0.760, "oblique_view": 0.850, "small_marker": 0.730},
        {"normal": 0.998, "low_light": 0.985, "motion_blur": 0.970, "oblique_view": 0.968, "small_marker": 0.950},
        0.050,
        None,
        2.30,
        0.45,
        4.30,
        18.0,
    ),
    BackendSpec(
        "mbv2_fp32_cpu",
        "MobileNetV2-0.35 FP32 / CPU",
        "Laptop CPU",
        "FP32",
        452_000,
        1.73,
        {"normal": 0.970, "low_light": 0.940, "motion_blur": 0.910, "oblique_view": 0.930, "small_marker": 0.900},
        {"normal": 0.998, "low_light": 0.991, "motion_blur": 0.983, "oblique_view": 0.980, "small_marker": 0.972},
        0.200,
        0.958,
        11.80,
        0.35,
        14.80,
        26.0,
    ),
    BackendSpec(
        "mbv2_int8_cpu",
        "MobileNetV2-0.35 INT8 / CPU",
        "Laptop CPU",
        "INT8",
        452_000,
        0.46,
        {"normal": 0.965, "low_light": 0.932, "motion_blur": 0.900, "oblique_view": 0.920, "small_marker": 0.890},
        {"normal": 0.997, "low_light": 0.990, "motion_blur": 0.981, "oblique_view": 0.978, "small_marker": 0.970},
        0.220,
        0.950,
        7.30,
        0.35,
        10.10,
        22.0,
        copy_predictions_from="mbv2_fp32_cpu",
        copy_drop_rate=0.008,
        copy_extra_false_positive_rate=0.010,
        copy_id_error_rate=0.002,
    ),
    BackendSpec(
        "hybrid_fp32_cpu",
        "Hybrid CNN + ArUco fallback / CPU",
        "Laptop CPU",
        "FP32",
        452_000,
        1.73,
        {"normal": 0.988, "low_light": 0.970, "motion_blur": 0.950, "oblique_view": 0.965, "small_marker": 0.940},
        {"normal": 0.999, "low_light": 0.996, "motion_blur": 0.992, "oblique_view": 0.990, "small_marker": 0.985},
        0.100,
        0.978,
        13.50,
        0.42,
        16.80,
        28.0,
    ),
    BackendSpec(
        "mbv2_int8_fpga_projected",
        "MobileNetV2-0.35 INT8 / FPGA",
        "PYNQ-Z2 projected",
        "INT8",
        452_000,
        0.46,
        {"normal": 0.965, "low_light": 0.932, "motion_blur": 0.900, "oblique_view": 0.920, "small_marker": 0.890},
        {"normal": 0.997, "low_light": 0.990, "motion_blur": 0.981, "oblique_view": 0.978, "small_marker": 0.970},
        0.220,
        0.950,
        5.20,
        0.60,
        9.00,
        6.5,
        result_origin="PROJECTED_NOT_MEASURED",
        copy_predictions_from="mbv2_int8_cpu",
    ),
)


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(rng: np.random.Generator) -> list[dict[str, object]]:
    conditions = [
        condition
        for condition, count in CONDITION_COUNTS.items()
        for _ in range(count)
    ]
    rng.shuffle(conditions)
    positives = np.asarray([1] * 800 + [0] * 200, dtype=np.int32)
    rng.shuffle(positives)
    marker_ids = np.tile(np.arange(10, dtype=np.int32), 80)
    rng.shuffle(marker_ids)
    positive_index = 0
    frames: list[dict[str, object]] = []
    for frame, (condition, positive) in enumerate(zip(conditions, positives)):
        marker_id = int(marker_ids[positive_index]) if positive else -1
        positive_index += int(positive)
        frames.append(
            {
                "frame": frame,
                "condition": condition,
                "marker_present": int(positive),
                "actual_marker_id": marker_id,
            }
        )
    return frames


def random_wrong_id(rng: np.random.Generator, actual_id: int) -> int:
    options = [marker_id for marker_id in range(10) if marker_id != actual_id]
    return int(rng.choice(options))


def simulate(
    frames: list[dict[str, object]], rng: np.random.Generator
) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    predictions: dict[str, list[tuple[int, int]]] = {}
    for spec in BACKENDS:
        backend_predictions: list[tuple[int, int]] = []
        source_predictions = predictions.get(spec.copy_predictions_from or "")
        for frame_data in frames:
            condition = str(frame_data["condition"])
            present = bool(frame_data["marker_present"])
            actual_id = int(frame_data["actual_marker_id"])
            if source_predictions is not None:
                detected, predicted_id = source_predictions[int(frame_data["frame"])]
                if present and detected and rng.random() < spec.copy_drop_rate:
                    detected, predicted_id = 0, -1
                elif (
                    present
                    and detected
                    and predicted_id == actual_id
                    and rng.random() < spec.copy_id_error_rate
                ):
                    predicted_id = random_wrong_id(rng, actual_id)
                elif (
                    not present
                    and not detected
                    and rng.random() < spec.copy_extra_false_positive_rate
                ):
                    detected, predicted_id = 1, int(rng.integers(0, 10))
            elif present:
                detected = int(rng.random() < spec.recall_by_condition[condition])
                if detected:
                    id_correct = rng.random() < spec.id_accuracy_by_condition[condition]
                    predicted_id = actual_id if id_correct else random_wrong_id(rng, actual_id)
                else:
                    predicted_id = -1
            else:
                detected = int(rng.random() < spec.negative_false_positive_rate)
                predicted_id = int(rng.integers(0, 10)) if detected else -1
            backend_predictions.append((detected, predicted_id))

            roi_ms = max(0.05, float(rng.normal(spec.roi_ms, spec.roi_ms * 0.09)))
            decode_ms = max(0.02, float(rng.normal(spec.decode_ms, spec.decode_ms * 0.12)))
            overhead_mean = max(spec.total_ms - spec.roi_ms - spec.decode_ms, 0.1)
            overhead_ms = max(0.05, float(rng.normal(overhead_mean, overhead_mean * 0.10)))
            total_ms = roi_ms + decode_ms + overhead_ms
            power_w = max(0.1, float(rng.normal(spec.active_power_w, spec.active_power_w * 0.04)))
            all_rows.append(
                {
                    "data_label": spec.result_origin,
                    "backend": spec.key,
                    "backend_name": spec.display_name,
                    **frame_data,
                    "detected": detected,
                    "predicted_marker_id": predicted_id,
                    "detection_correct": int(present and detected),
                    "id_correct": int(present and detected and predicted_id == actual_id),
                    "false_positive": int(not present and detected),
                    "missed": int(present and not detected),
                    "roi_or_detection_ms": round(roi_ms, 4),
                    "id_decode_ms": round(decode_ms, 4),
                    "total_ms": round(total_ms, 4),
                    "processing_fps": round(1000.0 / total_ms, 4),
                    "active_power_w_dummy": round(power_w, 4),
                }
            )
        predictions[spec.key] = backend_predictions
    return all_rows


def metric_row(spec: BackendSpec, rows: list[dict[str, object]]) -> dict[str, object]:
    positives = sum(int(row["marker_present"]) for row in rows)
    tp = sum(int(row["detection_correct"]) for row in rows)
    fp = sum(int(row["false_positive"]) for row in rows)
    fn = sum(int(row["missed"]) for row in rows)
    correct_ids = sum(int(row["id_correct"]) for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    total_ms = [float(row["total_ms"]) for row in rows]
    fps = [float(row["processing_fps"]) for row in rows]
    power = [float(row["active_power_w_dummy"]) for row in rows]
    return {
        "data_label": spec.result_origin,
        "backend": spec.key,
        "backend_name": spec.display_name,
        "platform": spec.platform,
        "precision_format": spec.precision,
        "parameters": "" if spec.parameters is None else spec.parameters,
        "model_size_mb_dummy": "" if spec.model_size_mb is None else spec.model_size_mb,
        "evaluation_frames": len(rows),
        "positive_frames": positives,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "marker_precision": precision,
        "marker_recall": recall,
        "marker_f1": f1,
        "marker_ap50_dummy": "" if spec.ap50_dummy is None else spec.ap50_dummy,
        "id_decode_accuracy_on_detected": correct_ids / max(tp, 1),
        "end_to_end_correct_id_rate": correct_ids / max(positives, 1),
        "total_ms_mean": statistics.fmean(total_ms),
        "total_ms_median": statistics.median(total_ms),
        "total_ms_p95": percentile(total_ms, 95),
        "processing_fps_mean": statistics.fmean(fps),
        "processing_fps_median": statistics.median(fps),
        "active_power_w_mean_dummy": statistics.fmean(power),
    }


def build_condition_rows(
    simulated: list[dict[str, object]], specs: tuple[BackendSpec, ...]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for spec in specs:
        backend_rows = [row for row in simulated if row["backend"] == spec.key]
        for condition in CONDITION_COUNTS:
            rows = [row for row in backend_rows if row["condition"] == condition]
            summary = metric_row(spec, rows)
            result.append(
                {
                    "data_label": spec.result_origin,
                    "backend": spec.key,
                    "condition": condition,
                    "frames": len(rows),
                    "marker_precision": summary["marker_precision"],
                    "marker_recall": summary["marker_recall"],
                    "marker_f1": summary["marker_f1"],
                    "id_decode_accuracy_on_detected": summary["id_decode_accuracy_on_detected"],
                    "end_to_end_correct_id_rate": summary["end_to_end_correct_id_rate"],
                }
            )
    return result


def build_confusion_rows(simulated: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for spec in BACKENDS:
        rows = [
            row
            for row in simulated
            if row["backend"] == spec.key and int(row["marker_present"]) == 1
        ]
        for actual_id in range(10):
            actual_rows = [row for row in rows if int(row["actual_marker_id"]) == actual_id]
            result: dict[str, object] = {
                "data_label": spec.result_origin,
                "backend": spec.key,
                "actual_id": actual_id,
            }
            for predicted_id in range(10):
                result[f"predicted_{predicted_id}"] = sum(
                    int(row["predicted_marker_id"]) == predicted_id for row in actual_rows
                )
            result["missed"] = sum(int(row["detected"]) == 0 for row in actual_rows)
            output.append(result)
    return output


def build_latency_wide(simulated: list[dict[str, object]]) -> list[dict[str, object]]:
    by_backend = {
        spec.key: [row for row in simulated if row["backend"] == spec.key]
        for spec in BACKENDS
    }
    rows: list[dict[str, object]] = []
    for index in range(len(next(iter(by_backend.values())))):
        row: dict[str, object] = {"sample": index}
        for spec in BACKENDS:
            item = by_backend[spec.key][index]
            row[f"{spec.key}_total_ms"] = item["total_ms"]
            row[f"{spec.key}_fps"] = item["processing_fps"]
        rows.append(row)
    return rows


def validate_simulation(
    simulated: list[dict[str, object]], summaries: list[dict[str, object]]
) -> None:
    expected_rows = sum(CONDITION_COUNTS.values()) * len(BACKENDS)
    if len(simulated) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} simulated rows, got {len(simulated)}")
    allowed_labels = {"DUMMY_SIMULATED", "PROJECTED_NOT_MEASURED"}
    if any(row["data_label"] not in allowed_labels for row in simulated):
        raise RuntimeError("A simulated row is missing an explicit dummy/projected label")
    for row in summaries:
        for field in (
            "marker_precision",
            "marker_recall",
            "marker_f1",
            "id_decode_accuracy_on_detected",
            "end_to_end_correct_id_rate",
        ):
            if not 0 <= float(row[field]) <= 1:
                raise RuntimeError(f"Invalid {field} for {row['backend']}: {row[field]}")
    int8 = next(row for row in summaries if row["backend"] == "mbv2_int8_cpu")
    fpga = next(
        row for row in summaries if row["backend"] == "mbv2_int8_fpga_projected"
    )
    for field in (
        "tp",
        "fp",
        "fn",
        "marker_precision",
        "marker_recall",
        "id_decode_accuracy_on_detected",
        "end_to_end_correct_id_rate",
    ):
        if int8[field] != fpga[field]:
            raise RuntimeError(f"FPGA projection must preserve INT8 accuracy field {field}")


def draw_bar_chart(
    path: Path,
    title: str,
    series: list[tuple[str, list[float]]],
    labels: list[str],
    maximum: float,
    suffix: str,
) -> None:
    width, height = 1800, 900
    left, right, top, bottom = 150, 70, 165, 190
    plot_w, plot_h = width - left - right, height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = [(48, 121, 200), (80, 170, 80), (190, 120, 55)]
    cv2.putText(image, title, (100, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "DUMMY / PROJECTED DATA - NOT MEASURED", (100, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (40, 40, 210), 2, cv2.LINE_AA)
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - int(value / maximum * plot_h)
        cv2.line(image, (left, y), (width - right, y), (225, 225, 225), 1)
        cv2.putText(image, f"{value:.0f}{suffix}", (30, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    group_width = plot_w / len(labels)
    bar_width = int(group_width / (len(series) + 1))
    for group, label in enumerate(labels):
        center = left + int((group + 0.5) * group_width)
        for series_index, (series_name, values) in enumerate(series):
            value = values[group]
            x1 = center + int((series_index - len(series) / 2) * bar_width)
            x2 = x1 + bar_width - 8
            y1 = top + plot_h - int(value / maximum * plot_h)
            cv2.rectangle(image, (x1, y1), (x2, top + plot_h), colors[series_index], -1)
            display = f"{value:.1f}{suffix}"
            cv2.putText(image, display, (x1, max(y1 - 8, top + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        short_label = label.replace("MobileNetV2-0.35", "MBv2-0.35").replace(" / ", "\n")
        for line_index, line in enumerate(short_label.split("\n")):
            cv2.putText(image, line, (center - int(group_width * 0.40), top + plot_h + 38 + 26 * line_index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1, cv2.LINE_AA)
    legend_x = left
    for index, (name, _values) in enumerate(series):
        x = legend_x + index * 500
        cv2.rectangle(image, (x, height - 45), (x + 24, height - 21), colors[index], -1)
        cv2.putText(image, name, (x + 34, height - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def write_readme(output: Path, summaries: list[dict[str, object]], seed: int) -> None:
    table = [
        "| Backend | Precision | Recall | Correct ID end-to-end | Mean latency ms | Mean FPS | Origin |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        table.append(
            f"| {row['backend_name']} | {float(row['marker_precision']):.3f} | "
            f"{float(row['marker_recall']):.3f} | {float(row['end_to_end_correct_id_rate']):.3f} | "
            f"{float(row['total_ms_mean']):.2f} | {float(row['processing_fps_mean']):.1f} | "
            f"{row['data_label']} |"
        )
    table_text = "\n".join(table)
    text = f"""# Dummy backend comparison

> **CẢNH BÁO:** Toàn bộ số liệu trong thư mục này là dữ liệu giả lập có seed
> `{seed}`, dùng để dựng bảng/biểu đồ và kiểm tra workflow. Không trình bày chúng
> như kết quả thực nghiệm trong khóa luận.

Giả định: 1.000 frame, 800 frame có marker, ID 0-9 cân bằng; các điều kiện gồm
normal, low light, motion blur, oblique view và small marker. CNN chỉ phát hiện ROI;
ID vẫn được giải mã bởi OpenCV ArUco. Kết quả FPGA giả định giữ accuracy INT8 CPU
và chỉ mô phỏng thay đổi latency/power.

{table_text}

## File

- `backend_summary_dummy.csv`: bảng so sánh chính;
- `accuracy_by_condition_dummy.csv`: kết quả theo điều kiện;
- `frame_predictions_dummy.csv`: raw prediction từng frame/backend;
- `id_confusion_matrix_dummy.csv`: confusion matrix ID và missed detection;
- `latency_prism_wide_dummy.csv`: dữ liệu wide để nhập Prism;
- `figure_accuracy_comparison_dummy.png`: biểu đồ accuracy;
- `figure_latency_comparison_dummy.png`: biểu đồ latency;
- `figure_fps_comparison_dummy.png`: biểu đồ FPS;
- `figure_latency_fps_comparison_dummy.png`: biểu đồ gộp, FPS được chia cho 10;
- `dummy_manifest.json`: giả định và provenance.

Thay các file này bằng output thật từ experiment runner trước khi hoàn thiện chương
Kết quả. Không xóa hậu tố `dummy` nếu số liệu chưa được thay thế bằng phép đo thật.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clearly labeled dummy backend results for thesis drafting"
    )
    parser.add_argument("--output", default="artifacts/dummy_results/backend_comparison")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    frames = make_dataset(rng)
    simulated = simulate(frames, rng)
    summaries = [
        metric_row(spec, [row for row in simulated if row["backend"] == spec.key])
        for spec in BACKENDS
    ]
    validate_simulation(simulated, summaries)
    condition_rows = build_condition_rows(simulated, BACKENDS)
    confusion_rows = build_confusion_rows(simulated)
    latency_wide = build_latency_wide(simulated)

    write_csv(output / "backend_summary_dummy.csv", summaries)
    write_csv(output / "accuracy_by_condition_dummy.csv", condition_rows)
    write_csv(output / "frame_predictions_dummy.csv", simulated)
    write_csv(output / "id_confusion_matrix_dummy.csv", confusion_rows)
    write_csv(output / "latency_prism_wide_dummy.csv", latency_wide)

    labels = [spec.display_name for spec in BACKENDS]
    draw_bar_chart(
        output / "figure_accuracy_comparison_dummy.png",
        "Illustrative marker detection and ID decoding comparison",
        [
            ("Precision", [100 * float(row["marker_precision"]) for row in summaries]),
            ("Recall", [100 * float(row["marker_recall"]) for row in summaries]),
            ("Correct ID end-to-end", [100 * float(row["end_to_end_correct_id_rate"]) for row in summaries]),
        ],
        labels,
        100.0,
        "%",
    )
    draw_bar_chart(
        output / "figure_latency_comparison_dummy.png",
        "Illustrative end-to-end latency comparison",
        [("Mean latency (ms)", [float(row["total_ms_mean"]) for row in summaries])],
        labels,
        20.0,
        " ms",
    )
    draw_bar_chart(
        output / "figure_fps_comparison_dummy.png",
        "Illustrative processing throughput comparison",
        [("Mean processing FPS", [float(row["processing_fps_mean"]) for row in summaries])],
        labels,
        250.0,
        "",
    )
    draw_bar_chart(
        output / "figure_latency_fps_comparison_dummy.png",
        "Illustrative processing speed comparison",
        [
            ("Mean latency (ms)", [float(row["total_ms_mean"]) for row in summaries]),
            ("Mean FPS / 10", [float(row["processing_fps_mean"]) / 10 for row in summaries]),
        ],
        labels,
        25.0,
        "",
    )

    manifest = {
        "data_status": "DUMMY_SIMULATED_NOT_FOR_FINAL_RESULTS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "frames": len(frames),
        "positive_frames": sum(int(frame["marker_present"]) for frame in frames),
        "marker_ids": list(range(10)),
        "conditions": CONDITION_COUNTS,
        "dictionary": "DICT_4X4_50",
        "cnn_role": "marker ROI detection; OpenCV ArUco decodes ID",
        "latency_scope": "illustrative end-to-end software pipeline",
        "power_scope": "illustrative active system power; measurement boundaries are not comparable",
        "fpga_assumption": "INT8 accuracy copied from CPU INT8; latency and power projected, not synthesized/measured",
        "classical_latency_anchor": "chosen near the local offline smoke-test range; still dummy",
    }
    (output / "dummy_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_readme(output, summaries, args.seed)
    print(f"Generated clearly labeled dummy results for {len(BACKENDS)} backends: {output}")


if __name__ == "__main__":
    main()
