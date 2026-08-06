from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentLogger:
    def __init__(self, experiment_dir: Path) -> None:
        self.log_path = experiment_dir / "runner.log"
        self.command_path = experiment_dir / "commands.log"

    def message(self, message: str) -> None:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def run(
        self,
        command: list[str],
        cwd: Path = REPO_ROOT,
        extra_log: Path | None = None,
    ) -> None:
        rendered = subprocess.list2cmdline(command)
        self.message(f"RUN {rendered}")
        with self.command_path.open("a", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
        trial_stream = extra_log.open("a", encoding="utf-8") if extra_log else None
        try:
            log_stream = self.log_path.open("a", encoding="utf-8")
            with log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log_stream.write(line)
                    if trial_stream is not None:
                        trial_stream.write(line)
                return_code = process.wait()
        finally:
            if trial_stream is not None:
                trial_stream.close()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.")
    return result or "poc"


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manual_templates(experiment: Path, trials: int) -> None:
    manual_fields = [
        "trial",
        "condition",
        "lighting",
        "lux",
        "actual_target_stop_distance_m",
        "target_stop_error_cm",
        "actual_home_stop_distance_m",
        "home_stop_error_cm",
        "turn_angle_error_deg",
        "average_power_w",
        "peak_power_w",
        "energy_wh",
        "measurement_instrument",
        "operator_success",
        "false_stop_count",
        "marker_loss_count",
        "notes",
    ]
    with (experiment / "manual_trial_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=manual_fields)
        writer.writeheader()
        for index in range(1, trials + 1):
            writer.writerow({"trial": f"trial_{index:03d}"})

    hardware_fields = [
        "backend",
        "board",
        "clock_mhz",
        "precision",
        "lut",
        "ff",
        "bram",
        "dsp",
        "estimated_power_w",
        "measured_idle_power_w",
        "measured_active_power_w",
        "notes",
    ]
    with (experiment / "hardware_measurements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=hardware_fields)
        writer.writeheader()
        writer.writerow({"backend": "CPU_PoC", "board": "not_applicable"})
        writer.writerow({"backend": "FPGA", "board": "TBD_when_board_available"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeatable PoC trials and store all evidence in one directory"
    )
    parser.add_argument(
        "--source",
        default="synthetic",
        help="synthetic, camera index (0), or video path",
    )
    parser.add_argument("--name", default="roundtrip_poc")
    parser.add_argument("--output-root", default="artifacts/experiments")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--mode", choices=("classical", "cnn", "hybrid"), default="classical")
    parser.add_argument("--mission", choices=("stop", "roundtrip"), default="roundtrip")
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument("--config", default="model/configs/mobilenetv2_035.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default="dataset/aruco")
    parser.add_argument("--audit-dataset", action="store_true")
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--evaluate-model", action="store_true")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="test")
    parser.add_argument("--camera-calibration", default=None)
    parser.add_argument("--marker-size-m", type=float, default=0.10)
    parser.add_argument("--focal-length-px", type=float, default=None)
    parser.add_argument("--stop-distance-m", type=float, default=0.45)
    parser.add_argument("--slow-distance-m", type=float, default=0.90)
    parser.add_argument("--stop-side-px", type=float, default=180.0)
    parser.add_argument("--slow-side-px", type=float, default=100.0)
    parser.add_argument("--confirm-frames", type=int, default=3)
    parser.add_argument("--start-confirm-frames", type=int, default=3)
    parser.add_argument("--target-dwell-frames", type=int, default=15)
    parser.add_argument("--turn-frames", type=int, default=30)
    parser.add_argument("--turn-angular-speed", type=float, default=0.50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--drop-warmup", type=int, default=5)
    parser.add_argument("--headless", action="store_true", help="Use for video files, not interactive webcam setup")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--no-exit-on-complete", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    if args.start_id == args.target_id and args.mission == "roundtrip":
        parser.error("--start-id and --target-id must be different")
    if args.mode != "classical" and not args.checkpoint:
        parser.error("--checkpoint is required for cnn/hybrid mode")
    if args.evaluate_model and not args.checkpoint:
        parser.error("--checkpoint is required with --evaluate-model")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    experiment = output_root / f"{stamp}_{safe_name(args.name)}"
    experiment.mkdir(parents=True, exist_ok=False)
    (experiment / "trials").mkdir()
    logger = ExperimentLogger(experiment)
    write_manual_templates(experiment, args.trials)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_dir": str(experiment),
        "arguments": vars(args),
        "source_kind": (
            "synthetic_video"
            if args.source == "synthetic"
            else "webcam"
            if args.source.isdigit()
            else "video_file"
        ),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status_short": git_value("status", "--short"),
        },
        "trials": [],
    }
    manifest_path = experiment / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (experiment / "README.txt").write_text(
        "Raw evidence is under trials/. Derived tables are under analysis/.\n"
        "Fill manual_trial_measurements.csv after physical trials; leave unknown values blank.\n"
        "Do not edit per-frame metrics.csv files. Re-run summarize_experiment.py instead.\n",
        encoding="utf-8",
    )
    logger.message(f"Experiment directory: {experiment}")

    try:
        for index in range(1, args.trials + 1):
            trial_name = f"trial_{index:03d}"
            trial_dir = experiment / "trials" / trial_name
            trial_dir.mkdir(parents=True)
            logger.message(f"Starting {trial_name}/{args.trials}")
            if args.source != "synthetic" and not args.no_prompt:
                input(
                    f"Prepare {trial_name}. Press Enter to start; use Q/Esc to end an incomplete trial..."
                )

            source = args.source
            if args.source == "synthetic":
                input_video = trial_dir / "input.avi"
                if args.mission == "roundtrip":
                    logger.run(
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
                else:
                    logger.run(
                        [
                            sys.executable,
                            "poc/make_demo_video.py",
                            "--marker-id",
                            str(args.target_id),
                            "--output",
                            str(input_video),
                        ]
                    )
                source = str(input_video)

            metrics = trial_dir / "metrics.csv"
            annotated = trial_dir / "annotated.avi"
            command = [
                sys.executable,
                "poc/live_webcam_demo.py",
                "--source",
                source,
                "--mode",
                args.mode,
                "--mission",
                args.mission,
                "--start-id",
                str(args.start_id),
                "--target-id",
                str(args.target_id),
                "--config",
                args.config,
                "--marker-size-m",
                str(args.marker_size_m),
                "--stop-distance-m",
                str(args.stop_distance_m),
                "--slow-distance-m",
                str(args.slow_distance_m),
                "--stop-side-px",
                str(args.stop_side_px),
                "--slow-side-px",
                str(args.slow_side_px),
                "--confirm-frames",
                str(args.confirm_frames),
                "--start-confirm-frames",
                str(args.start_confirm_frames),
                "--target-dwell-frames",
                str(args.target_dwell_frames),
                "--turn-frames",
                str(args.turn_frames),
                "--turn-angular-speed",
                str(args.turn_angular_speed),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
                "--camera-fps",
                str(args.camera_fps),
                "--max-frames",
                str(args.max_frames),
                "--csv",
                str(metrics),
                "--snapshot-dir",
                str(trial_dir / "snapshots"),
            ]
            if args.checkpoint:
                command.extend(["--checkpoint", args.checkpoint])
            if args.camera_calibration:
                command.extend(["--camera-calibration", args.camera_calibration])
            if args.focal_length_px is not None:
                command.extend(["--focal-length-px", str(args.focal_length_px)])
            if not args.no_video:
                command.extend(["--output-video", str(annotated)])
            if args.source == "synthetic" or args.headless:
                command.append("--headless")
            if args.source != "synthetic" and not args.no_exit_on_complete:
                command.append("--exit-on-complete")

            logger.run(command, extra_log=trial_dir / "console.log")
            trial_record = {
                "trial": trial_name,
                "metrics": str(metrics),
                "metadata": str(metrics.with_suffix(".meta.json")),
                "annotated_video": str(annotated) if not args.no_video else None,
            }
            manifest["trials"].append(trial_record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            if args.mission == "roundtrip" and not args.no_video:
                figure_dir = trial_dir / "figures"
                try:
                    logger.run(
                        [
                            sys.executable,
                            "poc/extract_thesis_figures.py",
                            "--video",
                            str(annotated),
                            "--csv",
                            str(metrics),
                            "--output-dir",
                            str(figure_dir),
                        ]
                    )
                except subprocess.CalledProcessError:
                    logger.message(
                        f"WARNING {trial_name}: figures skipped because the mission did not contain all states"
                    )

        if args.audit_dataset:
            audit_dir = experiment / "dataset_audit"
            audit_dir.mkdir()
            audit_command = [
                sys.executable,
                "model/scripts/audit_aruco_dataset.py",
                "--root",
                args.dataset_root,
                "--output",
                str(audit_dir),
            ]
            if args.hash_images:
                audit_command.append("--hash-images")
            logger.run(audit_command, extra_log=audit_dir / "console.log")

        if args.evaluate_model:
            evaluation_dir = experiment / "model_evaluation"
            evaluation_dir.mkdir()
            logger.run(
                [
                    sys.executable,
                    "model/evaluate.py",
                    "--config",
                    args.config,
                    "--checkpoint",
                    args.checkpoint,
                    "--dataset-root",
                    args.dataset_root,
                    "--split",
                    args.evaluation_split,
                    "--output-json",
                    str(evaluation_dir / "model_summary.json"),
                    "--predictions-csv",
                    str(evaluation_dir / "model_predictions.csv"),
                    "--pr-curve-csv",
                    str(evaluation_dir / "pr_curve.csv"),
                ],
                extra_log=evaluation_dir / "console.log",
            )

        logger.run(
            [
                sys.executable,
                "experiments/summarize_experiment.py",
                str(experiment),
                "--drop-warmup",
                str(args.drop_warmup),
            ]
        )
        manifest["status"] = "COMPLETE"
        logger.message("Experiment completed")
    except BaseException as error:
        manifest["status"] = "FAILED"
        manifest["error"] = f"{type(error).__name__}: {error}"
        logger.message(f"Experiment failed: {manifest['error']}")
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"RESULTS={experiment}")


if __name__ == "__main__":
    main()
