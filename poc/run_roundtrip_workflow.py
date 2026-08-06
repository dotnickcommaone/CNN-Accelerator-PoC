from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATES = (
    "WAITING_FOR_START",
    "OUTBOUND",
    "AT_TARGET",
    "TURNING_HOME",
    "RETURNING",
    "HOME_COMPLETE",
)


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def first_state_sequence(rows: list[dict[str, str]]) -> list[str]:
    sequence: list[str] = []
    for row in rows:
        state = row["state"]
        if not sequence or sequence[-1] != state:
            sequence.append(state)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate, run, validate and export the offline round-trip PoC"
    )
    parser.add_argument("--output-dir", default="artifacts/webcam_poc/roundtrip_workflow")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=1)
    args = parser.parse_args()
    if args.start_id == args.target_id:
        parser.error("--start-id and --target-id must be different")

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    input_video = output / "roundtrip_input.avi"
    result_video = output / "roundtrip_result.avi"
    metrics = output / "roundtrip_metrics.csv"
    tables = output / "thesis_tables"

    run(
        [
            sys.executable,
            "poc/make_roundtrip_demo_video.py",
            "--start-id",
            str(args.start_id),
            "--target-id",
            str(args.target_id),
            "--output",
            str(input_video),
        ]
    )
    run(
        [
            sys.executable,
            "poc/live_webcam_demo.py",
            "--source",
            str(input_video),
            "--mode",
            "classical",
            "--mission",
            "roundtrip",
            "--start-id",
            str(args.start_id),
            "--target-id",
            str(args.target_id),
            "--headless",
            "--output-video",
            str(result_video),
            "--csv",
            str(metrics),
        ]
    )

    with metrics.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sequence = first_state_sequence(rows)
    if sequence != list(EXPECTED_STATES):
        raise RuntimeError(
            f"Round-trip validation failed: expected {EXPECTED_STATES}, received {sequence}"
        )
    if not rows or rows[-1].get("mission_complete") != "1":
        raise RuntimeError("Round-trip validation failed: final mission_complete is not 1")

    run(
        [
            sys.executable,
            "analysis/export_thesis_tables.py",
            "--run",
            f"roundtrip={metrics}",
            "--drop-warmup",
            "0",
            "--output",
            str(tables),
        ]
    )
    first_target = next(row for row in rows if row["state"] == "AT_TARGET")
    first_home = next(row for row in rows if row["state"] == "HOME_COMPLETE")
    summary = {
        "passed": True,
        "start_id": args.start_id,
        "target_id": args.target_id,
        "frames": len(rows),
        "state_sequence": sequence,
        "state_counts": dict(Counter(row["state"] for row in rows)),
        "first_target_frame": int(first_target["frame"]),
        "home_complete_frame": int(first_home["frame"]),
        "metrics_csv": str(metrics),
        "result_video": str(result_video),
        "thesis_tables": str(tables),
    }
    summary_path = output / "workflow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"PASS: complete round trip; summary={summary_path}")


if __name__ == "__main__":
    main()
