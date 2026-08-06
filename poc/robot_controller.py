from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class RobotState(str, Enum):
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    SLOWING = "SLOWING"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class RobotCommand:
    state: RobotState
    speed: float
    reason: str
    filtered_distance_m: float | None = None
    filtered_side_px: float | None = None


class StopController:
    """Debounced stop controller; output speed is normalized to [0, 1]."""

    def __init__(
        self,
        target_id: int,
        stop_distance_m: float = 0.45,
        slow_distance_m: float = 0.90,
        stop_side_px: float = 180.0,
        slow_side_px: float = 100.0,
        confirm_frames: int = 3,
        lost_tolerance_frames: int = 5,
        smoothing_window: int = 5,
    ) -> None:
        if confirm_frames <= 0 or lost_tolerance_frames < 0 or smoothing_window <= 0:
            raise ValueError("Invalid controller frame/window configuration")
        if not 0 < stop_distance_m < slow_distance_m:
            raise ValueError("Distance thresholds must satisfy 0 < stop < slow")
        if not 0 < slow_side_px < stop_side_px:
            raise ValueError("Pixel thresholds must satisfy 0 < slow < stop")
        self.target_id = target_id
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = slow_distance_m
        self.stop_side_px = stop_side_px
        self.slow_side_px = slow_side_px
        self.confirm_frames = confirm_frames
        self.lost_tolerance_frames = lost_tolerance_frames
        self.distance_history: deque[float] = deque(maxlen=smoothing_window)
        self.side_history: deque[float] = deque(maxlen=smoothing_window)
        self.close_frames = 0
        self.lost_frames = 0
        self.state = RobotState.SEARCHING

    def reset(self) -> None:
        self.distance_history.clear()
        self.side_history.clear()
        self.close_frames = 0
        self.lost_frames = 0
        self.state = RobotState.SEARCHING

    def update(
        self,
        marker_visible: bool,
        distance_m: float | None = None,
        side_px: float | None = None,
    ) -> RobotCommand:
        if self.state == RobotState.STOPPED:
            return self._command(0.0, "stop latched; press R to reset")

        if not marker_visible:
            self.lost_frames += 1
            self.close_frames = 0
            if self.lost_frames <= self.lost_tolerance_frames and (
                self.distance_history or self.side_history
            ):
                self.state = RobotState.SLOWING
                return self._command(0.20, "target temporarily lost")
            self.state = RobotState.SEARCHING
            return self._command(0.25, "searching for target marker")

        self.lost_frames = 0
        if distance_m is not None:
            self.distance_history.append(distance_m)
        if side_px is not None:
            self.side_history.append(side_px)
        filtered_distance = (
            median(self.distance_history) if self.distance_history else None
        )
        filtered_side = median(self.side_history) if self.side_history else None

        close = (
            filtered_distance is not None
            and filtered_distance <= self.stop_distance_m
        ) or (
            filtered_distance is None
            and filtered_side is not None
            and filtered_side >= self.stop_side_px
        )
        slow = (
            filtered_distance is not None
            and filtered_distance <= self.slow_distance_m
        ) or (
            filtered_distance is None
            and filtered_side is not None
            and filtered_side >= self.slow_side_px
        )

        if close:
            self.close_frames += 1
            if self.close_frames >= self.confirm_frames:
                self.state = RobotState.STOPPED
                return self._command(0.0, "target distance confirmed")
            self.state = RobotState.SLOWING
            return self._command(
                0.10, f"confirming stop {self.close_frames}/{self.confirm_frames}"
            )

        self.close_frames = 0
        if slow:
            self.state = RobotState.SLOWING
            return self._command(0.25, "inside slow zone")
        self.state = RobotState.APPROACHING
        return self._command(0.60, "target acquired")

    def _command(self, speed: float, reason: str) -> RobotCommand:
        return RobotCommand(
            state=self.state,
            speed=speed,
            reason=reason,
            filtered_distance_m=(
                median(self.distance_history) if self.distance_history else None
            ),
            filtered_side_px=median(self.side_history) if self.side_history else None,
        )
