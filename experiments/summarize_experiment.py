from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUNDTRIP_STATES = (
    "WAITING_FOR_START",
    "OUTBOUND",
    "AT_TARGET",
    "TURNING_HOME",
    "RETURNING",
    "HOME_COMPLETE",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def compressed_states(rows: list[dict[str, str]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        state = row.get("state", "")
        if state and (not result or result[-1] != state):
            result.append(state)
    return result


def describe(values: list[float], name: str) -> dict[str, float | int | None]:
    return {
        f"{name}_n": len(values),
        f"{name}_mean": statistics.fmean(values) if values else None,
        f"{name}_median": statistics.median(values) if values else None,
        f"{name}_sd": statistics.stdev(values) if len(values) > 1 else None,
        f"{name}_min": min(values) if values else None,
        f"{name}_max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate all PoC trials in one experiment directory"
    )
    parser.add_argument("experiment_dir")
    parser.add_argument("--drop-warmup", type=int, default=5)
    args = parser.parse_args()
    experiment = Path(args.experiment_dir).resolve()
    manifest_path = experiment / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    source_kind = manifest.get("source_kind", "unknown")
    motion_time_valid = int(source_kind == "webcam")
    trials_dir = experiment / "trials"
    metrics_files = sorted(trials_dir.glob("trial_*/metrics.csv"))
    if not metrics_files:
        raise FileNotFoundError(f"No trials found under {trials_dir}")

    analysis_dir = experiment / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    export_command = [
        sys.executable,
        str(REPO_ROOT / "analysis/export_thesis_tables.py"),
    ]
    for metrics in metrics_files:
        export_command.extend(["--run", f"{metrics.parent.name}={metrics}"])
    export_command.extend(
        ["--drop-warmup", str(args.drop_warmup), "--output", str(analysis_dir)]
    )
    subprocess.run(export_command, cwd=REPO_ROOT, check=True)

    automatic_rows = {
        row["run"]: row for row in read_csv(analysis_dir / "stop_trials.csv")
    }
    performance_rows = {
        row["run"]: row for row in read_csv(analysis_dir / "poc_summary.csv")
    }
    manual_rows = {
        row.get("trial", ""): row
        for row in read_csv(experiment / "manual_trial_measurements.csv")
    }

    merged_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for metrics in metrics_files:
        trial = metrics.parent.name
        rows = read_csv(metrics)
        states = compressed_states(rows)
        mission = rows[0].get("mission", "unknown") if rows else "unknown"
        if mission == "roundtrip":
            if states == list(ROUNDTRIP_STATES):
                status = "COMPLETE"
            elif all(state in states for state in ROUNDTRIP_STATES):
                status = "INVALID_SEQUENCE"
            else:
                status = "INCOMPLETE"
        else:
            status = "COMPLETE" if rows and rows[-1].get("state") == "STOPPED" else "INCOMPLETE"

        automatic = automatic_rows.get(trial, {})
        performance = performance_rows.get(trial, {})
        manual = manual_rows.get(trial, {})
        merged_rows.append(
            {
                "trial": trial,
                "mission": mission,
                "detector_mode": rows[0].get("mode", "") if rows else "",
                "source_kind": source_kind,
                "motion_time_valid": motion_time_valid,
                "automatic_status": status,
                **{key: value for key, value in automatic.items() if key != "run"},
                "vision_ms_mean": performance.get("vision_ms_mean", ""),
                "vision_ms_p95": performance.get("vision_ms_p95", ""),
                "total_ms_mean": performance.get("total_ms_mean", ""),
                "total_ms_p95": performance.get("total_ms_p95", ""),
                "fps_mean": performance.get("fps_mean", ""),
                "fps_p95": performance.get("fps_p95", ""),
                **{key: value for key, value in manual.items() if key != "trial"},
            }
        )
        quality_rows.append(
            {
                "trial": trial,
                "rows": len(rows),
                "mission": mission,
                "state_sequence": " -> ".join(states),
                "final_state": states[-1] if states else "",
                "automatic_status": status,
                "metadata_present": int((metrics.parent / "metrics.meta.json").is_file()),
                "video_present": int((metrics.parent / "annotated.avi").is_file()),
            }
        )

    write_csv(analysis_dir / "prism_trial_summary.csv", merged_rows)
    write_csv(analysis_dir / "data_quality.csv", quality_rows)

    completed = sum(row["automatic_status"] == "COMPLETE" for row in merged_rows)
    overview: dict[str, object] = {
        "experiment_dir": str(experiment),
        "trials": len(merged_rows),
        "completed_trials": completed,
        "completion_rate": completed / len(merged_rows),
        "drop_warmup": args.drop_warmup,
        "source_kind": source_kind,
        "motion_time_valid": motion_time_valid,
        "timing_note": (
            "Wall-clock mission times are valid for physical motion analysis."
            if motion_time_valid
            else "Mission times are processing times only; do not report them as robot travel time."
        ),
    }
    for field in (
        "vision_ms_mean",
        "total_ms_mean",
        "fps_mean",
        "time_to_target_s",
        "return_time_s",
        "mission_duration_s",
        "target_stop_error_cm",
        "home_stop_error_cm",
        "turn_angle_error_deg",
        "average_power_w",
        "energy_wh",
    ):
        values = [
            value
            for row in merged_rows
            if (value := number(row.get(field))) is not None
        ]
        overview.update(describe(values, field))

    optional_summaries = {
        "dataset_audit": experiment / "dataset_audit" / "summary.json",
        "model_evaluation": experiment / "model_evaluation" / "model_summary.json",
    }
    for name, path in optional_summaries.items():
        if path.is_file():
            overview[name] = json.loads(path.read_text(encoding="utf-8"))

    (experiment / "experiment_summary.json").write_text(
        json.dumps(overview, indent=2), encoding="utf-8"
    )
    write_csv(
        analysis_dir / "experiment_overview.csv",
        [{"metric": key, "value": value} for key, value in overview.items()],
    )
    print(
        f"SUMMARY: {completed}/{len(merged_rows)} completed; "
        f"Prism table={analysis_dir / 'prism_trial_summary.csv'}"
    )


if __name__ == "__main__":
    main()
