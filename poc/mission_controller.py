from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class MissionState(str, Enum):
    WAITING_FOR_START = "WAITING_FOR_START"
    OUTBOUND = "OUTBOUND"
    AT_TARGET = "AT_TARGET"
    TURNING_HOME = "TURNING_HOME"
    RETURNING = "RETURNING"
    HOME_COMPLETE = "HOME_COMPLETE"


@dataclass(frozen=True)
class MarkerObservation:
    visible: bool
    distance_m: float | None = None
    side_px: float | None = None


@dataclass(frozen=True)
class MissionCommand:
    state: MissionState
    linear_speed: float
    angular_speed: float
    reason: str
    active_marker_id: int
    target_arrived: bool = False
    mission_complete: bool = False

    @property
    def speed(self) -> float:
        """Compatibility alias used by the original overlay/logger."""
        return self.linear_speed


class RoundTripController:
    """Two-marker mission: start → target → turn → start → complete."""

    def __init__(
        self,
        start_id: int,
        target_id: int,
        stop_distance_m: float = 0.45,
        slow_distance_m: float = 0.90,
        stop_side_px: float = 180.0,
        slow_side_px: float = 100.0,
        confirm_frames: int = 3,
        start_confirm_frames: int = 3,
        target_dwell_frames: int = 15,
        turn_frames: int = 30,
        turn_angular_speed: float = 0.50,
        smoothing_window: int = 5,
    ) -> None:
        if start_id == target_id:
            raise ValueError("start_id and target_id must be different")
        if min(confirm_frames, start_confirm_frames, target_dwell_frames, turn_frames, smoothing_window) <= 0:
            raise ValueError("Frame counts and smoothing_window must be positive")
        if not 0 < stop_distance_m < slow_distance_m:
            raise ValueError("Distance thresholds must satisfy 0 < stop < slow")
        if not 0 < slow_side_px < stop_side_px:
            raise ValueError("Pixel thresholds must satisfy 0 < slow < stop")
        if not 0 < turn_angular_speed <= 1:
            raise ValueError("turn_angular_speed must be in (0, 1]")
        self.start_id = start_id
        self.target_id = target_id
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = slow_distance_m
        self.stop_side_px = stop_side_px
        self.slow_side_px = slow_side_px
        self.confirm_frames = confirm_frames
        self.start_confirm_frames = start_confirm_frames
        self.target_dwell_frames = target_dwell_frames
        self.turn_frames = turn_frames
        self.turn_angular_speed = turn_angular_speed
        self.target_distances: deque[float] = deque(maxlen=smoothing_window)
        self.target_sides: deque[float] = deque(maxlen=smoothing_window)
        self.home_distances: deque[float] = deque(maxlen=smoothing_window)
        self.home_sides: deque[float] = deque(maxlen=smoothing_window)
        self.reset()

    def reset(self) -> None:
        self.state = MissionState.WAITING_FOR_START
        self.start_seen_frames = 0
        self.close_frames = 0
        self.phase_frames = 0
        self.target_distances.clear()
        self.target_sides.clear()
        self.home_distances.clear()
        self.home_sides.clear()

    @staticmethod
    def _append(observation: MarkerObservation, distances: deque[float], sides: deque[float]) -> None:
        if observation.distance_m is not None:
            distances.append(observation.distance_m)
        if observation.side_px is not None:
            sides.append(observation.side_px)

    def _zone(self, distances: deque[float], sides: deque[float]) -> tuple[bool, bool]:
        distance = median(distances) if distances else None
        side = median(sides) if sides else None
        close = (distance is not None and distance <= self.stop_distance_m) or (
            distance is None and side is not None and side >= self.stop_side_px
        )
        slow = (distance is not None and distance <= self.slow_distance_m) or (
            distance is None and side is not None and side >= self.slow_side_px
        )
        return close, slow

    def update(
        self,
        start: MarkerObservation,
        target: MarkerObservation,
    ) -> MissionCommand:
        if self.state == MissionState.WAITING_FOR_START:
            self.start_seen_frames = self.start_seen_frames + 1 if start.visible else 0
            if self.start_seen_frames >= self.start_confirm_frames:
                self.state = MissionState.OUTBOUND
                self.phase_frames = 0
                return self._command(0.35, 0.0, "start marker confirmed; outbound")
            return self._command(0.0, 0.0, f"confirming start {self.start_seen_frames}/{self.start_confirm_frames}")

        if self.state == MissionState.OUTBOUND:
            if not target.visible:
                self.close_frames = 0
                return self._command(0.35, 0.0, "outbound; searching target")
            self._append(target, self.target_distances, self.target_sides)
            close, slow = self._zone(self.target_distances, self.target_sides)
            if close:
                self.close_frames += 1
                if self.close_frames >= self.confirm_frames:
                    self.state = MissionState.AT_TARGET
                    self.phase_frames = 0
                    return self._command(0.0, 0.0, "target reached", target_arrived=True)
                return self._command(0.10, 0.0, f"confirming target {self.close_frames}/{self.confirm_frames}")
            self.close_frames = 0
            return self._command(0.22 if slow else 0.60, 0.0, "approaching target")

        if self.state == MissionState.AT_TARGET:
            self.phase_frames += 1
            if self.phase_frames >= self.target_dwell_frames:
                self.state = MissionState.TURNING_HOME
                self.phase_frames = 0
                return self._command(
                    0.0,
                    self.turn_angular_speed,
                    "begin 180-degree turn",
                    target_arrived=True,
                )
            return self._command(0.0, 0.0, f"target dwell {self.phase_frames}/{self.target_dwell_frames}", target_arrived=True)

        if self.state == MissionState.TURNING_HOME:
            self.phase_frames += 1
            if self.phase_frames >= self.turn_frames:
                self.state = MissionState.RETURNING
                self.close_frames = 0
                return self._command(0.30, 0.0, "turn complete; searching start marker", target_arrived=True)
            return self._command(
                0.0,
                self.turn_angular_speed,
                f"turning home {self.phase_frames}/{self.turn_frames}",
                target_arrived=True,
            )

        if self.state == MissionState.RETURNING:
            if not start.visible:
                self.close_frames = 0
                return self._command(0.30, 0.0, "returning; searching start marker", target_arrived=True)
            self._append(start, self.home_distances, self.home_sides)
            close, slow = self._zone(self.home_distances, self.home_sides)
            if close:
                self.close_frames += 1
                if self.close_frames >= self.confirm_frames:
                    self.state = MissionState.HOME_COMPLETE
                    return self._command(0.0, 0.0, "home marker confirmed; mission complete", target_arrived=True, mission_complete=True)
                return self._command(0.10, 0.0, f"confirming home {self.close_frames}/{self.confirm_frames}", target_arrived=True)
            self.close_frames = 0
            return self._command(0.22 if slow else 0.55, 0.0, "approaching home", target_arrived=True)

        return self._command(0.0, 0.0, "mission complete; reset required", target_arrived=True, mission_complete=True)

    def _command(
        self,
        linear: float,
        angular: float,
        reason: str,
        target_arrived: bool = False,
        mission_complete: bool = False,
    ) -> MissionCommand:
        active_id = self.start_id if self.state in {
            MissionState.WAITING_FOR_START,
            MissionState.TURNING_HOME,
            MissionState.RETURNING,
            MissionState.HOME_COMPLETE,
        } else self.target_id
        return MissionCommand(
            state=self.state,
            linear_speed=linear,
            angular_speed=angular,
            reason=reason,
            active_marker_id=active_id,
            target_arrived=target_arrived,
            mission_complete=mission_complete,
        )
