from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

from generate_dummy_backend_results import draw_bar_chart, percentile, write_csv


AP50_TARGETS = {
    "mbv2_fp32_cpu": ("DUMMY_SIMULATED", 0.958),
    "mbv2_int8_cpu": ("DUMMY_SIMULATED", 0.950),
    "hybrid_fp32_cpu": ("DUMMY_SIMULATED", 0.978),
    "mbv2_int8_fpga_projected": ("PROJECTED_NOT_MEASURED", 0.950),
}

FPGA_CAPACITY = {"lut": 53_200, "ff": 106_400, "bram_36k": 140.0, "dsp": 220}

FPGA_CANDIDATES = (
    {
        "design": "hls_int8_pe4",
        "description": "Low-resource custom HLS",
        "lut": 18_900,
        "ff": 25_200,
        "bram_36k": 58.0,
        "dsp": 72,
        "target_clock_mhz": 125,
        "estimated_fmax_mhz": 148,
        "wns_ns_dummy": 1.24,
        "end_to_end_ms_dummy": 27.8,
        "fps_dummy": 36.0,
        "active_power_w_dummy": 4.1,
    },
    {
        "design": "hls_int8_pe8_balanced",
        "description": "Balanced custom HLS candidate",
        "lut": 28_700,
        "ff": 39_100,
        "bram_36k": 82.0,
        "dsp": 136,
        "target_clock_mhz": 125,
        "estimated_fmax_mhz": 136,
        "wns_ns_dummy": 0.65,
        "end_to_end_ms_dummy": 13.7,
        "fps_dummy": 73.0,
        "active_power_w_dummy": 5.5,
    },
    {
        "design": "hls_int8_pe12_aggressive",
        "description": "Aggressive parallel custom HLS",
        "lut": 41_900,
        "ff": 56_300,
        "bram_36k": 113.0,
        "dsp": 204,
        "target_clock_mhz": 125,
        "estimated_fmax_mhz": 118,
        "wns_ns_dummy": -0.47,
        "end_to_end_ms_dummy": 10.2,
        "fps_dummy": 98.0,
        "active_power_w_dummy": 7.0,
    },
    {
        "design": "dpu_lite_candidate",
        "description": "Illustrative Vitis-AI DPU-lite",
        "lut": 32_100,
        "ff": 47_500,
        "bram_36k": 94.0,
        "dsp": 160,
        "target_clock_mhz": 125,
        "estimated_fmax_mhz": 121,
        "wns_ns_dummy": -0.26,
        "end_to_end_ms_dummy": 12.8,
        "fps_dummy": 78.1,
        "active_power_w_dummy": 6.3,
    },
)

ROBOT_BACKENDS = {
    "opencv_classical_cpu": {
        "name": "OpenCV Classical CPU",
        "origin": "DUMMY_SIMULATED",
        "condition_success": {"normal": 0.96, "low_light": 0.80, "motion_blur": 0.76, "oblique_view": 0.84, "small_marker": 0.74},
        "home_factor": 0.97,
        "false_stop_rate": 0.070,
        "target_bias_m": 0.025,
        "target_sd_m": 0.055,
        "home_bias_m": 0.030,
        "home_sd_m": 0.060,
        "turn_sd_deg": 6.0,
        "duration_s": 25.0,
        "system_power_w": 24.0,
    },
    "hybrid_fp32_cpu": {
        "name": "Hybrid FP32 CPU",
        "origin": "DUMMY_SIMULATED",
        "condition_success": {"normal": 0.99, "low_light": 0.96, "motion_blur": 0.93, "oblique_view": 0.95, "small_marker": 0.92},
        "home_factor": 0.99,
        "false_stop_rate": 0.020,
        "target_bias_m": 0.012,
        "target_sd_m": 0.030,
        "home_bias_m": 0.015,
        "home_sd_m": 0.034,
        "turn_sd_deg": 4.0,
        "duration_s": 25.8,
        "system_power_w": 27.0,
    },
    "mbv2_int8_fpga_projected": {
        "name": "MobileNetV2 INT8 FPGA",
        "origin": "PROJECTED_NOT_MEASURED",
        "condition_success": {"normal": 0.99, "low_light": 0.96, "motion_blur": 0.93, "oblique_view": 0.95, "small_marker": 0.92},
        "home_factor": 0.99,
        "false_stop_rate": 0.020,
        "target_bias_m": 0.015,
        "target_sd_m": 0.035,
        "home_bias_m": 0.018,
        "home_sd_m": 0.038,
        "turn_sd_deg": 4.5,
        "duration_s": 24.6,
        "system_power_w": 18.0,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_values(rows: list[dict[str, object]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            values.append(float(value))
    return values


def generate_ap50(output: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    recall_points = np.linspace(0.0, 1.0, 101)
    for backend, (origin, target_ap) in AP50_TARGETS.items():
        exponent = 2.0
        coefficient = (1.0 - target_ap) * (exponent + 1.0)
        precision_points = np.clip(
            1.0 - coefficient * np.power(recall_points, exponent), 0.0, 1.0
        )
        computed_ap = float(np.trapezoid(precision_points, recall_points))
        for index, (recall, precision) in enumerate(
            zip(recall_points, precision_points)
        ):
            curve_rows.append(
                {
                    "data_label": origin,
                    "backend": backend,
                    "iou_threshold": 0.50,
                    "point": index,
                    "score_threshold_dummy": round(1.0 - float(recall) ** 0.65, 5),
                    "recall": round(float(recall), 6),
                    "precision": round(float(precision), 6),
                }
            )
        summary_rows.append(
            {
                "data_label": origin,
                "backend": backend,
                "iou_threshold": 0.50,
                "ap50_target_dummy": target_ap,
                "ap50_curve_auc_dummy": computed_ap,
                "evaluation_images_dummy": 1000,
                "note": "Classical OpenCV excluded because scored-box AP is not directly comparable",
            }
        )
    write_csv(output / "ap50_summary_dummy.csv", summary_rows)
    write_csv(output / "pr_curve_dummy.csv", curve_rows)
    draw_pr_curves(output / "figure_pr_curve_dummy.png", summary_rows, curve_rows)
    return summary_rows, curve_rows


def draw_pr_curves(
    path: Path,
    summaries: list[dict[str, object]],
    curves: list[dict[str, object]],
) -> None:
    width, height = 1500, 950
    left, right, top, bottom = 150, 80, 170, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = [(205, 120, 45), (60, 160, 70), (185, 75, 175), (60, 115, 205)]
    cv2.putText(image, "Illustrative Precision-Recall curves at IoU=0.50", (90, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(image, "DUMMY / PROJECTED DATA - NOT MEASURED", (90, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (40, 40, 210), 2, cv2.LINE_AA)
    for tick in range(0, 11, 2):
        value = tick / 10
        x = left + int(value * plot_w)
        y = top + plot_h - int(value * plot_h)
        cv2.line(image, (x, top), (x, top + plot_h), (230, 230, 230), 1)
        cv2.line(image, (left, y), (left + plot_w, y), (230, 230, 230), 1)
        cv2.putText(image, f"{value:.1f}", (x - 16, top + plot_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(image, f"{value:.1f}", (75, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(image, (left, top), (left, top + plot_h), (40, 40, 40), 2)
    cv2.line(image, (left, top + plot_h), (left + plot_w, top + plot_h), (40, 40, 40), 2)
    summary_by_backend = {str(row["backend"]): row for row in summaries}
    for index, backend in enumerate(AP50_TARGETS):
        rows = [row for row in curves if row["backend"] == backend]
        points = np.asarray(
            [
                [
                    left + int(float(row["recall"]) * plot_w),
                    top + plot_h - int(float(row["precision"]) * plot_h),
                ]
                for row in rows
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        cv2.polylines(image, [points], False, colors[index], 4, cv2.LINE_AA)
        summary = summary_by_backend[backend]
        legend_y = top + 32 + index * 43
        cv2.line(image, (left + 30, legend_y), (left + 80, legend_y), colors[index], 5)
        label = f"{backend}  AP50={float(summary['ap50_target_dummy']):.3f}"
        cv2.putText(image, label, (left + 95, legend_y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (45, 45, 45), 1, cv2.LINE_AA)
    cv2.putText(image, "Recall", (left + plot_w // 2 - 35, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(image, "Precision", (18, top + plot_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), image)


def generate_fallback(
    output: Path, base_rows: list[dict[str, str]], rng: np.random.Generator
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    primary = [row for row in base_rows if row["backend"] == "mbv2_fp32_cpu"]
    recovery_probability = {
        "normal": 0.65,
        "low_light": 0.45,
        "motion_blur": 0.35,
        "oblique_view": 0.40,
        "small_marker": 0.30,
    }
    id_probability = {
        "normal": 0.998,
        "low_light": 0.985,
        "motion_blur": 0.970,
        "oblique_view": 0.968,
        "small_marker": 0.950,
    }
    false_positive_probability = {
        "normal": 0.010,
        "low_light": 0.025,
        "motion_blur": 0.035,
        "oblique_view": 0.025,
        "small_marker": 0.030,
    }
    raw_rows: list[dict[str, object]] = []
    for row in primary:
        condition = row["condition"]
        present = row["marker_present"] == "1"
        primary_detected = row["detected"] == "1"
        if not present and primary_detected:
            # Most ROI false positives do not survive ArUco ID validation.
            primary_detected = rng.random() < 0.30
        fallback_invoked = not primary_detected
        fallback_detection = False
        fallback_id_correct = False
        if fallback_invoked and present:
            fallback_detection = rng.random() < recovery_probability[condition]
            fallback_id_correct = fallback_detection and rng.random() < id_probability[condition]
        elif fallback_invoked:
            fallback_detection = rng.random() < false_positive_probability[condition]
        final_detected = primary_detected or fallback_detection
        primary_id_correct = row["id_correct"] == "1"
        final_id_correct = primary_id_correct or fallback_id_correct
        fallback_ms = (
            max(0.1, float(rng.normal(2.75, 0.35))) if fallback_invoked else 0.0
        )
        total_ms = float(row["total_ms"]) + 0.8 + fallback_ms
        raw_rows.append(
            {
                "data_label": "DUMMY_SIMULATED",
                "frame": int(row["frame"]),
                "condition": condition,
                "marker_present": int(present),
                "actual_marker_id": int(row["actual_marker_id"]),
                "primary_detected": int(primary_detected),
                "primary_id_correct": int(primary_id_correct),
                "fallback_invoked": int(fallback_invoked),
                "fallback_detected": int(fallback_detection),
                "fallback_id_correct": int(fallback_id_correct),
                "final_detected": int(final_detected),
                "final_id_correct": int(final_id_correct),
                "fallback_extra_ms": round(fallback_ms, 4),
                "hybrid_total_ms": round(total_ms, 4),
            }
        )

    summary_rows: list[dict[str, object]] = []
    for condition in ["ALL", "normal", "low_light", "motion_blur", "oblique_view", "small_marker"]:
        rows = raw_rows if condition == "ALL" else [row for row in raw_rows if row["condition"] == condition]
        positive = [row for row in rows if row["marker_present"] == 1]
        negative = [row for row in rows if row["marker_present"] == 0]
        positive_fallback = [row for row in positive if row["fallback_invoked"] == 1]
        recovered = sum(row["fallback_detected"] == 1 for row in positive_fallback)
        summary_rows.append(
            {
                "data_label": "DUMMY_SIMULATED",
                "condition": condition,
                "frames": len(rows),
                "positive_frames": len(positive),
                "primary_recall": sum(row["primary_detected"] == 1 for row in positive) / max(len(positive), 1),
                "fallback_invocation_rate_all": sum(row["fallback_invoked"] == 1 for row in rows) / max(len(rows), 1),
                "fallback_invocation_rate_positive": len(positive_fallback) / max(len(positive), 1),
                "fallback_recovery_rate": recovered / max(len(positive_fallback), 1),
                "final_recall": sum(row["final_detected"] == 1 for row in positive) / max(len(positive), 1),
                "final_correct_id_rate": sum(row["final_id_correct"] == 1 for row in positive) / max(len(positive), 1),
                "negative_false_positive_rate": sum(row["final_detected"] == 1 for row in negative) / max(len(negative), 1),
                "fallback_extra_ms_mean": statistics.fmean(float(row["fallback_extra_ms"]) for row in rows),
                "hybrid_total_ms_mean": statistics.fmean(float(row["hybrid_total_ms"]) for row in rows),
                "hybrid_total_ms_p95": percentile([float(row["hybrid_total_ms"]) for row in rows], 95),
            }
        )
    write_csv(output / "fallback_raw_dummy.csv", raw_rows)
    write_csv(output / "fallback_summary_dummy.csv", summary_rows)
    condition_only = [row for row in summary_rows if row["condition"] != "ALL"]
    draw_bar_chart(
        output / "figure_fallback_recall_dummy.png",
        "Illustrative recall improvement from classical fallback",
        [
            ("CNN primary recall", [100 * float(row["primary_recall"]) for row in condition_only]),
            ("Final hybrid recall", [100 * float(row["final_recall"]) for row in condition_only]),
        ],
        [str(row["condition"]) for row in condition_only],
        100.0,
        "%",
    )
    return raw_rows, summary_rows


def generate_fpga_resources(output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in FPGA_CANDIDATES:
        row: dict[str, object] = {
            "data_label": "PROJECTED_NOT_MEASURED",
            "implementation_status": "NOT_SYNTHESIZED_DUMMY_ESTIMATE",
            **candidate,
        }
        for resource in ("lut", "ff", "bram_36k", "dsp"):
            row[f"{resource}_available"] = FPGA_CAPACITY[resource]
            row[f"{resource}_utilization_pct_dummy"] = (
                float(candidate[resource]) / FPGA_CAPACITY[resource] * 100.0
            )
        row["timing_met_dummy"] = int(float(candidate["wns_ns_dummy"]) >= 0)
        row["energy_mj_per_frame_dummy"] = (
            float(candidate["active_power_w_dummy"])
            / float(candidate["fps_dummy"])
            * 1000.0
        )
        rows.append(row)
    write_csv(output / "fpga_resource_estimates_dummy.csv", rows)

    labels = [str(row["design"]) for row in rows]
    draw_bar_chart(
        output / "figure_fpga_resource_utilization_dummy.png",
        "Illustrative PYNQ-Z2 resource utilization",
        [
            ("LUT", [float(row["lut_utilization_pct_dummy"]) for row in rows]),
            ("BRAM", [float(row["bram_36k_utilization_pct_dummy"]) for row in rows]),
            ("DSP", [float(row["dsp_utilization_pct_dummy"]) for row in rows]),
        ],
        labels,
        100.0,
        "%",
    )
    return rows


def simulate_robot_trials(
    output: Path, rng: np.random.Generator, trials_per_backend: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conditions = ("normal", "low_light", "motion_blur", "oblique_view", "small_marker")
    raw_rows: list[dict[str, object]] = []
    for backend, spec in ROBOT_BACKENDS.items():
        for trial_index in range(1, trials_per_backend + 1):
            condition = conditions[(trial_index - 1) % len(conditions)]
            target_reached = rng.random() < spec["condition_success"][condition]
            false_stop = rng.random() < spec["false_stop_rate"]
            home_reached = target_reached and rng.random() < spec["home_factor"]
            mission_complete = target_reached and home_reached and not false_stop
            target_actual = (
                max(0.15, float(rng.normal(0.45 + spec["target_bias_m"], spec["target_sd_m"])))
                if target_reached
                else None
            )
            home_actual = (
                max(0.15, float(rng.normal(0.45 + spec["home_bias_m"], spec["home_sd_m"])))
                if home_reached
                else None
            )
            turn_error = abs(float(rng.normal(0.0, spec["turn_sd_deg"]))) if target_reached else None
            duration = max(8.0, float(rng.normal(spec["duration_s"], 1.8)))
            marker_loss_base = {"normal": 0.2, "low_light": 1.0, "motion_blur": 1.4, "oblique_view": 0.9, "small_marker": 1.3}[condition]
            marker_loss = int(rng.poisson(marker_loss_base))
            average_power = max(1.0, float(rng.normal(spec["system_power_w"], spec["system_power_w"] * 0.05)))
            raw_rows.append(
                {
                    "data_label": spec["origin"],
                    "backend": backend,
                    "backend_name": spec["name"],
                    "trial": trial_index,
                    "condition": condition,
                    "route_length_m_dummy": 5.0,
                    "command_stop_distance_m": 0.45,
                    "target_reached": int(target_reached),
                    "home_reached": int(home_reached),
                    "false_stop": int(false_stop),
                    "mission_complete": int(mission_complete),
                    "actual_target_stop_distance_m": "" if target_actual is None else round(target_actual, 4),
                    "target_stop_error_cm": "" if target_actual is None else round(abs(target_actual - 0.45) * 100.0, 3),
                    "actual_home_stop_distance_m": "" if home_actual is None else round(home_actual, 4),
                    "home_stop_error_cm": "" if home_actual is None else round(abs(home_actual - 0.45) * 100.0, 3),
                    "turn_angle_error_deg": "" if turn_error is None else round(turn_error, 3),
                    "marker_loss_events": marker_loss,
                    "mission_duration_s": round(duration, 3),
                    "average_system_power_w_dummy": round(average_power, 3),
                    "mission_energy_wh_dummy": round(average_power * duration / 3600.0, 5),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for backend, spec in ROBOT_BACKENDS.items():
        rows = [row for row in raw_rows if row["backend"] == backend]
        target_error = finite_values(rows, "target_stop_error_cm")
        home_error = finite_values(rows, "home_stop_error_cm")
        turn_error = finite_values(rows, "turn_angle_error_deg")
        duration = finite_values(rows, "mission_duration_s")
        energy = finite_values(rows, "mission_energy_wh_dummy")
        summary_rows.append(
            {
                "data_label": spec["origin"],
                "backend": backend,
                "backend_name": spec["name"],
                "trials": len(rows),
                "target_reach_rate": sum(row["target_reached"] == 1 for row in rows) / len(rows),
                "home_reach_rate": sum(row["home_reached"] == 1 for row in rows) / len(rows),
                "mission_completion_rate": sum(row["mission_complete"] == 1 for row in rows) / len(rows),
                "false_stop_rate": sum(row["false_stop"] == 1 for row in rows) / len(rows),
                "target_stop_mae_cm": statistics.fmean(target_error),
                "target_stop_p95_cm": percentile(target_error, 95),
                "home_stop_mae_cm": statistics.fmean(home_error),
                "home_stop_p95_cm": percentile(home_error, 95),
                "turn_angle_error_mean_deg": statistics.fmean(turn_error),
                "marker_loss_events_mean": statistics.fmean(float(row["marker_loss_events"]) for row in rows),
                "mission_duration_mean_s": statistics.fmean(duration),
                "mission_energy_mean_wh_dummy": statistics.fmean(energy),
            }
        )
    write_csv(output / "robot_trials_raw_dummy.csv", raw_rows)
    write_csv(output / "robot_results_summary_dummy.csv", summary_rows)

    wide_rows: list[dict[str, object]] = []
    rows_by_backend = {
        backend: [row for row in raw_rows if row["backend"] == backend]
        for backend in ROBOT_BACKENDS
    }
    for index in range(trials_per_backend):
        row: dict[str, object] = {"trial": index + 1}
        for backend, rows in rows_by_backend.items():
            row[f"{backend}_target_error_cm"] = rows[index]["target_stop_error_cm"]
            row[f"{backend}_home_error_cm"] = rows[index]["home_stop_error_cm"]
            row[f"{backend}_duration_s"] = rows[index]["mission_duration_s"]
            row[f"{backend}_mission_complete"] = rows[index]["mission_complete"]
        wide_rows.append(row)
    write_csv(output / "robot_trials_prism_wide_dummy.csv", wide_rows)

    labels = [str(row["backend_name"]) for row in summary_rows]
    draw_bar_chart(
        output / "figure_robot_mission_results_dummy.png",
        "Illustrative robot round-trip mission results",
        [
            ("Mission complete", [100 * float(row["mission_completion_rate"]) for row in summary_rows]),
            ("Target reached", [100 * float(row["target_reach_rate"]) for row in summary_rows]),
            ("Home reached", [100 * float(row["home_reach_rate"]) for row in summary_rows]),
        ],
        labels,
        100.0,
        "%",
    )
    return raw_rows, summary_rows


def validate(
    ap_rows: list[dict[str, object]],
    fallback_summary: list[dict[str, object]],
    fpga_rows: list[dict[str, object]],
    robot_raw: list[dict[str, object]],
    robot_summary: list[dict[str, object]],
    trials_per_backend: int,
) -> None:
    if any(abs(float(row["ap50_target_dummy"]) - float(row["ap50_curve_auc_dummy"])) > 5e-6 for row in ap_rows):
        raise RuntimeError("Generated PR curve does not integrate to its AP50 target")
    fallback_all = next(row for row in fallback_summary if row["condition"] == "ALL")
    if float(fallback_all["final_recall"]) < float(fallback_all["primary_recall"]):
        raise RuntimeError("Fallback must not reduce simulated recall")
    if any(float(row["lut_utilization_pct_dummy"]) > 100 or float(row["dsp_utilization_pct_dummy"]) > 100 for row in fpga_rows):
        raise RuntimeError("A dummy FPGA candidate exceeds device capacity")
    if len(robot_raw) != len(ROBOT_BACKENDS) * trials_per_backend:
        raise RuntimeError("Unexpected robot raw row count")
    if any(not 0 <= float(row["mission_completion_rate"]) <= 1 for row in robot_summary):
        raise RuntimeError("Invalid robot completion rate")


def write_extended_readme(
    output: Path,
    fallback_summary: list[dict[str, object]],
    robot_summary: list[dict[str, object]],
) -> None:
    fallback = next(row for row in fallback_summary if row["condition"] == "ALL")
    lines = [
        "# Extended dummy results",
        "",
        "> All files are DUMMY_SIMULATED or PROJECTED_NOT_MEASURED. They are not thesis findings.",
        "",
        "## Fallback illustration",
        "",
        f"- invocation rate, all frames: {100 * float(fallback['fallback_invocation_rate_all']):.1f}%",
        f"- invocation rate, positive frames: {100 * float(fallback['fallback_invocation_rate_positive']):.1f}%",
        f"- recovery rate after primary miss: {100 * float(fallback['fallback_recovery_rate']):.1f}%",
        f"- primary recall: {100 * float(fallback['primary_recall']):.1f}%",
        f"- final recall: {100 * float(fallback['final_recall']):.1f}%",
        "",
        "## Robot illustration",
        "",
        "| Backend | Complete | Target MAE cm | Home MAE cm | Turn error deg |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in robot_summary:
        lines.append(
            f"| {row['backend_name']} | {100 * float(row['mission_completion_rate']):.1f}% | "
            f"{float(row['target_stop_mae_cm']):.2f} | {float(row['home_stop_mae_cm']):.2f} | "
            f"{float(row['turn_angle_error_mean_deg']):.2f} |"
        )
    lines.extend(
        [
            "",
            "OpenCV Classical has no AP50 row because its detector does not expose a directly comparable learned scored-box PR curve.",
            "FPGA resource, timing, power and robot results remain projections until a board and physical robot are measured.",
        ]
    )
    (output / "README_EXTENDED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dummy AP50, fallback, FPGA-resource and robot results"
    )
    parser.add_argument("--output", default="artifacts/dummy_results/backend_comparison")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--robot-trials", type=int, default=30)
    args = parser.parse_args()
    if args.robot_trials < 5:
        parser.error("--robot-trials must be at least 5 per backend")
    output = Path(args.output)
    base_raw_path = output / "frame_predictions_dummy.csv"
    if not base_raw_path.is_file():
        raise FileNotFoundError(
            f"Missing {base_raw_path}; run analysis/generate_dummy_backend_results.py first"
        )
    rng = np.random.default_rng(args.seed + 1)
    base_rows = read_csv(base_raw_path)

    ap_rows, _curve_rows = generate_ap50(output)
    _fallback_raw, fallback_summary = generate_fallback(output, base_rows, rng)
    fpga_rows = generate_fpga_resources(output)
    robot_raw, robot_summary = simulate_robot_trials(output, rng, args.robot_trials)
    validate(
        ap_rows,
        fallback_summary,
        fpga_rows,
        robot_raw,
        robot_summary,
        args.robot_trials,
    )
    manifest = {
        "data_status": "DUMMY_SIMULATED_AND_PROJECTED_NOT_MEASURED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed + 1,
        "robot_trials_per_backend": args.robot_trials,
        "ap50_note": "OpenCV Classical excluded from AP50 comparison",
        "fpga_device_capacity_assumption": FPGA_CAPACITY,
        "fpga_status": "No synthesis, implementation, timing or power report was used",
        "robot_status": "No physical robot trial was performed",
    }
    (output / "dummy_extended_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_extended_readme(output, fallback_summary, robot_summary)
    print(
        f"Generated dummy AP50, fallback, {len(fpga_rows)} FPGA candidates and "
        f"{len(robot_raw)} robot trials: {output}"
    )


if __name__ == "__main__":
    main()
