from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


STATES = (
    "SEARCHING",
    "APPROACHING",
    "SLOWING",
    "STOPPED",
    "WAITING_FOR_START",
    "OUTBOUND",
    "AT_TARGET",
    "TURNING_HOME",
    "RETURNING",
    "HOME_COMPLETE",
)


def number(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: list[float], prefix: str) -> dict[str, float | int | None]:
    return {
        f"{prefix}_n": len(values),
        f"{prefix}_mean": statistics.fmean(values) if values else None,
        f"{prefix}_median": statistics.median(values) if values else None,
        f"{prefix}_sd": statistics.stdev(values) if len(values) > 1 else None,
        f"{prefix}_p95": percentile(values, 95),
        f"{prefix}_min": min(values) if values else None,
        f"{prefix}_max": max(values) if values else None,
    }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=CSV, received: {spec}")
    label, path = spec.split("=", 1)
    if not label.strip() or not Path(path).is_file():
        raise ValueError(f"Invalid run or missing file: {spec}")
    return label.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PoC raw data and thesis-ready CSV tables")
    parser.add_argument("--run", action="append", required=True, help="LABEL=metrics.csv; repeat for each backend/trial")
    parser.add_argument("--output", default="artifacts/thesis_tables")
    parser.add_argument("--drop-warmup", type=int, default=5)
    args = parser.parse_args()
    if args.drop_warmup < 0:
        raise ValueError("--drop-warmup must be non-negative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    summaries: list[dict] = []
    states_by_run: dict[str, Counter[str]] = {}
    columns: dict[str, list[float]] = {}
    stop_rows: list[dict] = []
    manifest = {"drop_warmup": args.drop_warmup, "runs": []}

    for spec in args.run:
        label, path = parse_run(spec)
        with path.open("r", newline="", encoding="utf-8") as stream:
            all_rows = list(csv.DictReader(stream))
        rows = all_rows[args.drop_warmup :]
        vision = [value for row in rows if (value := number(row.get("vision_ms", ""))) is not None]
        total = [value for row in rows if (value := number(row.get("total_ms", ""))) is not None]
        fps = [value for row in rows if (value := number(row.get("fps", ""))) is not None]
        state_counts = Counter(row.get("state", "") for row in rows)
        states_by_run[label] = state_counts
        target_frames = sum(row.get("target_visible") == "1" for row in rows)
        start_frames = sum(row.get("start_visible") == "1" for row in rows)
        cnn_frames = sum(row.get("target_source") == "cnn+opencv" for row in rows)
        fallback_frames = sum(row.get("target_source") == "opencv" for row in rows)
        stopped = [row for row in rows if row.get("state") == "STOPPED"]
        first_stop = stopped[0] if stopped else None
        outbound = next((row for row in rows if row.get("state") == "OUTBOUND"), None)
        at_target = next((row for row in rows if row.get("state") == "AT_TARGET"), None)
        home_complete = next((row for row in rows if row.get("state") == "HOME_COMPLETE"), None)
        outbound_time = number(outbound.get("timestamp", "")) if outbound else None
        target_time = number(at_target.get("timestamp", "")) if at_target else None
        home_time = number(home_complete.get("timestamp", "")) if home_complete else None
        summary = {"run": label, "frames_total": len(all_rows), "frames_analyzed": len(rows)}
        summary.update(describe(vision, "vision_ms"))
        summary.update(describe(total, "total_ms"))
        summary.update(describe(fps, "fps"))
        summary.update(
            {
                "target_frame_rate": target_frames / max(len(rows), 1),
                "start_frame_rate": start_frames / max(len(rows), 1),
                "cnn_source_rate": cnn_frames / max(target_frames, 1),
                "opencv_source_rate": fallback_frames / max(target_frames, 1),
                "first_stop_frame": int(first_stop["frame"]) if first_stop else None,
                "first_stop_distance_m": number(first_stop.get("distance_m", "")) if first_stop else None,
                "first_stop_side_px": number(first_stop.get("side_px", "")) if first_stop else None,
                "first_target_frame": int(at_target["frame"]) if at_target else None,
                "first_target_distance_m": number(at_target.get("distance_m", "")) if at_target else None,
                "first_target_side_px": number(at_target.get("side_px", "")) if at_target else None,
                "home_complete_frame": int(home_complete["frame"]) if home_complete else None,
                "home_distance_m": number(home_complete.get("start_distance_m", "")) if home_complete else None,
                "home_side_px": number(home_complete.get("start_side_px", "")) if home_complete else None,
                "time_to_target_s": target_time - outbound_time if target_time is not None and outbound_time is not None else None,
                "return_time_s": home_time - target_time if home_time is not None and target_time is not None else None,
                "mission_duration_s": home_time - outbound_time if home_time is not None and outbound_time is not None else None,
                "mission_completed": int(first_stop is not None or home_complete is not None),
            }
        )
        summaries.append(summary)
        stop_rows.append(
            {
                "run": label,
                "stopped": int(first_stop is not None or at_target is not None),
                "target_reached": int(first_stop is not None or at_target is not None),
                "home_reached": int(home_complete is not None),
                "first_stop_frame": summary["first_stop_frame"],
                "first_stop_distance_m": summary["first_stop_distance_m"],
                "first_stop_side_px": summary["first_stop_side_px"],
                "first_target_frame": summary["first_target_frame"],
                "first_target_distance_m": summary["first_target_distance_m"],
                "first_target_side_px": summary["first_target_side_px"],
                "home_complete_frame": summary["home_complete_frame"],
                "home_distance_m": summary["home_distance_m"],
                "home_side_px": summary["home_side_px"],
                "time_to_target_s": summary["time_to_target_s"],
                "return_time_s": summary["return_time_s"],
                "mission_duration_s": summary["mission_duration_s"],
                "mission_completed": summary["mission_completed"],
            }
        )
        columns[f"{label}_vision_ms"] = vision
        columns[f"{label}_total_ms"] = total
        columns[f"{label}_fps"] = fps
        for row in rows:
            raw_rows.append({"run": label, **row})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["runs"].append({"label": label, "file": str(path), "sha256": digest, "rows": len(all_rows)})

    write_csv(output / "poc_summary.csv", summaries)
    write_csv(output / "poc_raw_long.csv", raw_rows)
    write_csv(output / "stop_trials.csv", stop_rows)
    observed_states = {state for counts in states_by_run.values() for state in counts if state}
    ordered_states = list(STATES) + sorted(observed_states.difference(STATES))
    state_rows = [{"state": state, **{label: states_by_run[label].get(state, 0) for label in states_by_run}} for state in ordered_states]
    write_csv(output / "state_counts.csv", state_rows)
    max_length = max((len(values) for values in columns.values()), default=0)
    wide_rows = [{name: values[index] if index < len(values) else "" for name, values in columns.items()} for index in range(max_length)]
    write_csv(output / "latency_fps_wide.csv", wide_rows, list(columns))
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {len(summaries)} runs to {output}")


if __name__ == "__main__":
    main()
