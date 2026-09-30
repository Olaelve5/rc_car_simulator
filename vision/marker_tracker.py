"""ArUco marker detection and 2D vehicle pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from .homography import HomographyTransformer


@dataclass
class VehiclePose:
    """Estimated state of vehicle within arena reference frame.

    Attributes:
        x: Position along arena X-axis in meters.
        y: Position along arena Y-axis in meters.
        yaw: Orientation angle in radians [-pi, pi].
        timestamp: Monotonic epoch timestamp of detection.
        confidence: Detection confidence or corner reprojection metric.
    """

    x: float
    y: float
    yaw: float
    timestamp: float
    confidence: float = 1.0


class MarkerTracker:
    """Detects planar ArUco markers and resolves vehicle 2D ground pose.

    Attributes:
        dictionary_id: OpenCV ArUco predefined dictionary identifier.
        vehicle_marker_id: Target marker ID affixed to vehicle roof.
        marker_size_m: Physical width/length of marker square in meters.
    """

    def __init__(
        self,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
        vehicle_marker_id: int = 0,
        marker_size_m: float = 0.05,
    ) -> None:
        """Initializes detector parameters and dictionary."""
        self.dictionary_id = dictionary_id
        self.vehicle_marker_id = vehicle_marker_id
        self.marker_size_m = marker_size_m

    def detect_markers(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
        """Scans frame for ArUco markers and returns detected corners and IDs.

        Args:
            frame: Input BGR image array.

        Returns:
            Tuple of (corners, ids, rejected_candidates).
        """
        raise NotImplementedError

    def get_vehicle_pose(
        self,
        frame: np.ndarray,
        transformer: Optional[HomographyTransformer] = None,
    ) -> Optional[VehiclePose]:
        """Extracts 2D vehicle pose (x, y, yaw) from detected vehicle marker.

        Args:
            frame: Raw BGR camera image.
            transformer: Optional homography transformer for world coordinate mapping.

        Returns:
            VehiclePose instance if marker is detected, else None.
        """
        raise NotImplementedError

    def get_corner_markers(
        self,
        frame: np.ndarray,
        corner_ids: Tuple[int, int, int, int],
    ) -> Dict[int, Tuple[float, float]]:
        """Locates calibration corner markers defining the arena perimeter.

        Args:
            frame: Raw camera frame.
            corner_ids: 4-tuple of expected marker IDs for arena corners.

        Returns:
            Dictionary mapping marker ID to center pixel coordinates (u, v).
        """
        raise NotImplementedError
