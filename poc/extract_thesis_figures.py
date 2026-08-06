from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


STATE_ORDER = (
    "WAITING_FOR_START",
    "OUTBOUND",
    "AT_TARGET",
    "TURNING_HOME",
    "RETURNING",
    "HOME_COMPLETE",
)

STATE_COLORS = {
    "WAITING_FOR_START": (170, 110, 40),
    "OUTBOUND": (220, 150, 40),
    "AT_TARGET": (40, 150, 230),
    "TURNING_HOME": (180, 80, 180),
    "RETURNING": (70, 170, 80),
    "HOME_COMPLETE": (40, 130, 40),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def representative_frames(rows: list[dict[str, str]]) -> dict[str, int]:
    frames_by_state: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        frames_by_state[row["state"]].append(int(row["frame"]))
    missing = [state for state in STATE_ORDER if not frames_by_state[state]]
    if missing:
        raise ValueError(f"CSV is missing required states: {', '.join(missing)}")

    selected: dict[str, int] = {}
    for state in STATE_ORDER:
        state_frames = frames_by_state[state]
        if state in {"WAITING_FOR_START", "AT_TARGET", "HOME_COMPLETE"}:
            selected[state] = state_frames[0]
        else:
            selected[state] = state_frames[len(state_frames) // 2]
    return selected


def read_video_frames(video_path: Path, requested: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while requested.difference(frames):
            ok, frame = capture.read()
            if not ok:
                break
            if index in requested:
                frames[index] = frame
            index += 1
    finally:
        capture.release()
    missing = requested.difference(frames)
    if missing:
        raise RuntimeError(f"Video is missing requested frames: {sorted(missing)}")
    return frames


def labeled_cell(frame: np.ndarray, state: str, frame_index: int) -> np.ndarray:
    # Remove the runtime overlay from publication composites. In a headless video
    # run its FPS is offline processing throughput, not real-time webcam FPS.
    content = frame[86:, :] if frame.shape[0] > 86 else frame
    image = cv2.resize(content, (640, 394), interpolation=cv2.INTER_AREA)
    footer = np.full((50, 640, 3), 248, dtype=np.uint8)
    cv2.rectangle(footer, (0, 0), (10, 49), STATE_COLORS[state], -1)
    cv2.putText(
        footer,
        f"{state}  |  frame {frame_index}",
        (24, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((image, footer))


def make_sequence_figure(
    selected: dict[str, int], frames: dict[int, np.ndarray]
) -> np.ndarray:
    cells = [labeled_cell(frames[selected[state]], state, selected[state]) for state in STATE_ORDER]
    first_row = np.hstack(cells[:3])
    second_row = np.hstack(cells[3:])
    return np.vstack((first_row, second_row))


def make_target_home_figure(
    selected: dict[str, int], frames: dict[int, np.ndarray]
) -> np.ndarray:
    target = labeled_cell(
        frames[selected["AT_TARGET"]], "AT_TARGET", selected["AT_TARGET"]
    )
    home = labeled_cell(
        frames[selected["HOME_COMPLETE"]],
        "HOME_COMPLETE",
        selected["HOME_COMPLETE"],
    )
    return np.hstack((target, home))


def state_ranges(rows: list[dict[str, str]]) -> list[tuple[str, int, int]]:
    ranges: list[tuple[str, int, int]] = []
    start = int(rows[0]["frame"])
    current = rows[0]["state"]
    previous = start
    for row in rows[1:]:
        frame = int(row["frame"])
        if row["state"] != current:
            ranges.append((current, start, previous))
            current = row["state"]
            start = frame
        previous = frame
    ranges.append((current, start, previous))
    return ranges


def make_timeline_figure(rows: list[dict[str, str]]) -> np.ndarray:
    width, height = 1800, 650
    left, right, top, bottom = 300, 70, 90, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    total_frames = max(int(row["frame"]) for row in rows) + 1

    cv2.putText(
        image,
        "Round-trip mission state timeline",
        (left, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    row_height = plot_height / len(STATE_ORDER)
    ranges = state_ranges(rows)
    for state_index, state in enumerate(STATE_ORDER):
        y_center = int(top + (state_index + 0.5) * row_height)
        cv2.putText(
            image,
            state,
            (18, y_center + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        cv2.line(image, (left, y_center), (width - right, y_center), (225, 225, 225), 2)

    for state, start, end in ranges:
        state_index = STATE_ORDER.index(state)
        y_center = int(top + (state_index + 0.5) * row_height)
        x1 = left + int(start / max(total_frames - 1, 1) * plot_width)
        x2 = left + int(end / max(total_frames - 1, 1) * plot_width)
        cv2.rectangle(
            image,
            (x1, y_center - 19),
            (max(x2, x1 + 4), y_center + 19),
            STATE_COLORS[state],
            -1,
        )
        cv2.putText(
            image,
            f"{end - start + 1} frames",
            (x1 + 7, y_center + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    axis_y = height - bottom + 20
    cv2.line(image, (left, axis_y), (width - right, axis_y), (50, 50, 50), 2)
    for tick in range(0, total_frames + 1, 20):
        x = left + int(tick / max(total_frames - 1, 1) * plot_width)
        cv2.line(image, (x, axis_y - 7), (x, axis_y + 7), (50, 50, 50), 2)
        cv2.putText(
            image,
            str(tick),
            (x - 15, axis_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        "Frame index",
        (left + plot_width // 2 - 55, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (45, 45, 45),
        1,
        cv2.LINE_AA,
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract thesis figures from a round-trip PoC run")
    parser.add_argument(
        "--video",
        default="artifacts/webcam_poc/roundtrip_workflow/roundtrip_result.avi",
    )
    parser.add_argument(
        "--csv",
        default="artifacts/webcam_poc/roundtrip_workflow/roundtrip_metrics.csv",
    )
    parser.add_argument("--output-dir", default="artifacts/thesis_figures/roundtrip")
    args = parser.parse_args()

    video_path = Path(args.video)
    csv_path = Path(args.csv)
    output = Path(args.output_dir)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    output.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV contains no rows: {csv_path}")
    selected = representative_frames(rows)
    frames = read_video_frames(video_path, set(selected.values()))

    for number, state in enumerate(STATE_ORDER, start=1):
        frame_index = selected[state]
        filename = f"{number:02d}_{state.lower()}_f{frame_index:03d}.png"
        cv2.imwrite(str(output / filename), frames[frame_index])

    cv2.imwrite(
        str(output / "figure_roundtrip_state_sequence.png"),
        make_sequence_figure(selected, frames),
    )
    cv2.imwrite(
        str(output / "figure_target_and_home_comparison.png"),
        make_target_home_figure(selected, frames),
    )
    cv2.imwrite(
        str(output / "figure_roundtrip_state_timeline.png"),
        make_timeline_figure(rows),
    )
    print(f"Extracted {len(selected)} frames and 3 composite figures to {output}")


if __name__ == "__main__":
    main()
