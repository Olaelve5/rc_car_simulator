"""Arena homography + car marker tracking, as a per-frame tracker and a live service.

``ArenaTracker`` processes single frames. ``LiveTracker`` runs it on a camera
in a background thread so a simulator or control loop can poll the newest car
pose without blocking on capture or ArUco detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Optional, Tuple, Union

import numpy as np

from config.settings import ArenaConfig, CameraConfig, DEFAULT_SETTINGS
from .camera_stream import CameraStream
from .homography import HomographyTransformer
from .marker_tracker import Detections, MarkerTracker, VehiclePose


class ArenaTracker:
    """Keeps the homography up to date from corner markers and tracks the car.

    The homography is recomputed every frame where all four corner markers are
    visible; if some are occluded (e.g. by the car), the last one is reused.
    """

    def __init__(
        self,
        arena_width_m: float,
        arena_height_m: float,
        corner_ids: Tuple[int, int, int, int],
        vehicle_marker_id: int,
        dictionary: Union[int, str],
        flip_camera: bool = True,
    ) -> None:
        self.corner_ids = corner_ids
        self.transformer = HomographyTransformer(arena_width_m, arena_height_m)
        self.marker_tracker = MarkerTracker(
            dictionary_id=dictionary, vehicle_marker_id=vehicle_marker_id
        )
        self.flip_camera = flip_camera
        self.locked = False
        self.visible_corners: Tuple[int, ...] = ()

    @classmethod
    def from_config(
        cls,
        arena: ArenaConfig,
        flip_camera: bool = True,
        arena_width_m: Optional[float] = None,
        arena_height_m: Optional[float] = None,
    ) -> "ArenaTracker":
        """Builds a tracker from ArenaConfig, optionally overriding the size."""
        return cls(
            arena_width_m=arena_width_m or arena.arena_width_m,
            arena_height_m=arena_height_m or arena.arena_height_m,
            corner_ids=arena.corner_marker_ids,
            vehicle_marker_id=arena.vehicle_marker_id,
            dictionary=arena.aruco_dict_name,
            flip_camera=flip_camera,
        )

    def update(
        self, frame: np.ndarray, timestamp: Optional[float] = None
    ) -> Tuple[Optional[VehiclePose], Detections]:
        """Processes one frame. Returns the car pose in meters (or None)."""
        detections = self.marker_tracker.detect_markers(frame)
        corners = self.marker_tracker.corner_markers_from_detections(
            detections, self.corner_ids
        )
        self.visible_corners = tuple(corners)
        if not self.locked and len(corners) == 4:
            pixel_corners = np.array([corners[i] for i in self.corner_ids])
            self.transformer.compute_matrix(pixel_corners)

        if not self.transformer.is_calibrated:
            return None, detections
        pose = self.marker_tracker.vehicle_pose_from_detections(
            detections, self.transformer, timestamp
        )
        return pose, detections


@dataclass
class TrackedVehicle:
    """Latest tracked car state in the arena frame.

    Attributes:
        x: Position along arena X-axis in meters.
        y: Position along arena Y-axis in meters.
        yaw: Heading in radians [-pi, pi], counter-clockwise from +X.
        velocity: Smoothed signed speed along the heading in m/s (+forward).
        timestamp: Monotonic time of the camera frame the pose came from.
    """

    x: float
    y: float
    yaw: float
    velocity: float
    timestamp: float

    @property
    def age_s(self) -> float:
        """Seconds since the frame this pose was measured in."""
        return time.monotonic() - self.timestamp


class LiveTracker:
    """Tracks the real car from a camera in a background thread.

    Attributes:
        tracker: The underlying per-frame ArenaTracker.
        camera: The threaded camera grabber.
        stale_after_s: ``latest()`` returns None once a pose is older than this.
    """

    def __init__(
        self,
        source: Union[int, str, None] = None,
        arena: Optional[ArenaConfig] = None,
        camera: Optional[CameraConfig] = None,
        flip_camera: Optional[bool] = None,
        arena_width_m: Optional[float] = None,
        arena_height_m: Optional[float] = None,
        velocity_smoothing: float = 0.3,
        stale_after_s: float = 0.5,
    ) -> None:
        """Configures camera and tracker; call ``start()`` to begin tracking."""
        arena = arena or DEFAULT_SETTINGS.arena
        cam = camera or DEFAULT_SETTINGS.camera
        flip = cam.flip_video if flip_camera is None else flip_camera

        self.tracker = ArenaTracker.from_config(
            arena, flip, arena_width_m, arena_height_m
        )
        self.camera = CameraStream(
            source=cam.camera_id if source is None else source,
            resolution=cam.resolution,
            fps=cam.fps,
            fourcc=cam.fourcc,
            flip_video=flip,
        )
        self.velocity_smoothing = float(velocity_smoothing)
        self.stale_after_s = float(stale_after_s)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._vehicle: Optional[TrackedVehicle] = None
        self._frame: Optional[np.ndarray] = None
        self._detections: Detections = ([], None, [])
        self._frame_id = 0

    def start(self) -> "LiveTracker":
        """Opens the camera and starts the tracking thread.

        Raises:
            RuntimeError: If the camera or video file cannot be opened.
        """
        self.camera.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._track_loop, daemon=True, name="LiveTrackerThread"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stops tracking and releases the camera."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.camera.stop()

    def latest(self) -> Optional[TrackedVehicle]:
        """Returns the newest car state, or None if not seen recently."""
        with self._lock:
            vehicle = self._vehicle
        if vehicle is None or vehicle.age_s > self.stale_after_s:
            return None
        return vehicle

    def latest_frame(self) -> Tuple[int, Optional[np.ndarray], Detections]:
        """Returns (frame_id, last processed frame, its detections) for previews."""
        with self._lock:
            return self._frame_id, self._frame, self._detections

    @property
    def status(self) -> str:
        """Short human-readable tracking state."""
        if not self.tracker.transformer.is_calibrated:
            return f"CALIBRATING {len(self.tracker.visible_corners)}/4 corners"
        if self.latest() is None:
            return "SEARCHING CAR"
        return "LOCKED" if self.tracker.locked else "TRACKING"

    def toggle_lock(self) -> bool:
        """Freezes / unfreezes the homography; returns the new locked state."""
        self.tracker.locked = not self.tracker.locked
        return self.tracker.locked

    def _track_loop(self) -> None:
        last_frame_id = -1
        while self._running:
            is_new, frame, frame_id = self.camera.read_latest(last_frame_id)
            if not is_new or frame is None:
                time.sleep(0.002)
                continue
            last_frame_id = frame_id

            stamp = time.monotonic()
            pose, detections = self.tracker.update(frame, stamp)
            with self._lock:
                if pose is not None:
                    self._vehicle = self._next_vehicle(pose)
                self._frame = frame
                self._detections = detections
                self._frame_id = frame_id

    def _next_vehicle(self, pose: VehiclePose) -> TrackedVehicle:
        """Builds the new state, estimating speed by finite differences."""
        prev = self._vehicle
        velocity = 0.0
        if prev is not None:
            dt = pose.timestamp - prev.timestamp
            if 0.0 < dt < self.stale_after_s:
                dx, dy = pose.x - prev.x, pose.y - prev.y
                raw = (dx * math.cos(pose.yaw) + dy * math.sin(pose.yaw)) / dt
                a = self.velocity_smoothing
                velocity = a * raw + (1.0 - a) * prev.velocity
        return TrackedVehicle(
            x=pose.x,
            y=pose.y,
            yaw=pose.yaw,
            velocity=velocity,
            timestamp=pose.timestamp,
        )

    def __enter__(self) -> "LiveTracker":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
