"""UVC camera capture stream running in an independent acquisition thread."""

from __future__ import annotations

import threading
from typing import Optional, Tuple
import cv2
import numpy as np


class CameraStream:
    """Threaded camera grabber to decouple frame ingestion from control loops.

    Attributes:
        camera_id: OS video device index.
        resolution: Requested (width, height) capture resolution.
        fps: Target frame rate.
    """

    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        flip_video: bool = True,
    ) -> None:
        """Initializes camera parameters without immediately acquiring the device."""
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.flip_video = flip_video
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

    def start(self) -> CameraStream:
        """Opens capture device and launches background polling thread.

        Returns:
            Self instance for method chaining.

        Raises:
            RuntimeError: If video device fails to open.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Signals thread termination, waits for join, and releases hardware handle."""
        pass

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the most recent frame in a thread-safe manner.

        Returns:
            A tuple of (success_flag, latest_frame_bgr).
        """
        raise NotImplementedError

    def is_opened(self) -> bool:
        """Checks if underlying capture device is open and active."""
        raise NotImplementedError

    def _capture_loop(self) -> None:
        """Internal worker continuously grabbing frames from VideoCapture."""
        pass
