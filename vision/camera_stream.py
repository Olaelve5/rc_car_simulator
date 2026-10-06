"""Threaded UVC camera capture stream with hardware negotiation and FPS tracking."""

from __future__ import annotations

import collections
import threading
import time
from typing import Deque, Optional, Tuple, Union
import cv2
import numpy as np


class CameraStream:
    """Threaded camera grabber to decouple frame ingestion from control loops.

    Runs hardware frame acquisition in a dedicated daemon thread to prevent
    blocking calls (`cap.read()`) from throttling downstream processing and
    rendering pipelines. Tracks real-time acquisition FPS and hardware properties.

    Attributes:
        source: OS camera device index (int or str) or video file path.
        resolution: Requested (width, height) capture resolution in pixels.
        fps: Target capture frame rate requested from hardware driver.
        fourcc: Video codec FourCC code (default 'MJPG' for high-framerate UVC).
        flip_video: Whether to rotate frame 180 degrees for overhead mounting.
        buffer_size: Driver queue buffer size (1 minimizes latency).
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 60,
        fourcc: str = "MJPG",
        flip_video: bool = True,
        buffer_size: int = 1,
    ) -> None:
        """Initializes stream configuration without immediately acquiring device."""
        if isinstance(source, str) and source.isdigit():
            self.source: Union[int, str] = int(source)
        else:
            self.source = source

        self.resolution = resolution
        self.fps = fps
        self.fourcc = fourcc
        self.flip_video = flip_video
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()

        # Frame buffers and tracking metrics
        self._latest_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.measured_fps: float = 0.0
        self._frame_times: Deque[float] = collections.deque(maxlen=30)

        # Hardware reported properties
        self.actual_width: int = 0
        self.actual_height: int = 0
        self.actual_fps: float = 0.0

    def start(self) -> CameraStream:
        """Opens capture device, configures hardware, and launches background thread.

        Returns:
            Self instance for fluent chaining.

        Raises:
            RuntimeError: If video device fails to open.
        """
        if isinstance(self.source, int):
            # Prefer AVFoundation on macOS for low-latency UVC access
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        if isinstance(self.source, int):
            # Request MJPG codec first to enable higher framerates on USB 2.0
            if self.fourcc:
                self._cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*self.fourcc),
                )
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Disable autofocus if supported
            self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # Query what hardware driver accepted
            self.actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS))

            req_desc = (
                f"{self.resolution[0]}x{self.resolution[1]} "
                f"@ {self.fps}fps ({self.fourcc})"
            )
            hw_desc = (
                f"{self.actual_width}x{self.actual_height} "
                f"@ {self.actual_fps:.1f}fps"
            )
            print(
                f"[CameraStream] Source: {self.source} | "
                f"Req: {req_desc} | HW: {hw_desc}"
            )

            # Flush sensor initialization frames
            time.sleep(0.05)
            for _ in range(3):
                self._cap.read()

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="CameraCaptureThread",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signals worker thread to terminate and releases hardware device."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the most recent frame in a thread-safe copy.

        Returns:
            Tuple of (success_flag, latest_frame_bgr).
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def read_latest(
        self, last_frame_id: int = -1
    ) -> Tuple[bool, Optional[np.ndarray], int]:
        """Non-blocking check for new camera frame since last queried ID.

        Args:
            last_frame_id: Frame sequence ID processed in previous iteration.

        Returns:
            Tuple of (is_new_frame, frame_reference, current_frame_id).
            Returns (False, latest_frame, last_frame_id) if no new frame has arrived.
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, last_frame_id
            is_new = self.frame_count != last_frame_id
            return is_new, self._latest_frame, self.frame_count

    def is_opened(self) -> bool:
        """Checks if capture device is open and acquisition thread is running."""
        return self._cap is not None and self._cap.isOpened() and self._running

    def _capture_loop(self) -> None:
        """Background worker thread continuously grabbing frames from hardware."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            self._frame_times.append(now)
            if len(self._frame_times) >= 2:
                dt = self._frame_times[-1] - self._frame_times[0]
                if dt > 0.0:
                    self.measured_fps = (len(self._frame_times) - 1) / dt

            if self.flip_video:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            with self._lock:
                self._latest_frame = frame
                self.frame_count += 1

    def __enter__(self) -> CameraStream:
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
