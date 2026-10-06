"""ArUco marker detection and 2D vehicle pose estimation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
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


Detections = Tuple[List[np.ndarray], Optional[np.ndarray], List[np.ndarray]]


class MarkerTracker:
    """Detects planar ArUco markers and resolves vehicle 2D ground pose.

    Vehicle yaw is the direction of the marker's top edge (ArUco corners 0->1
    are its top-left/top-right), so mount the marker with its top facing the
    front of the car, or compensate with ``yaw_offset_rad``.

    Attributes:
        dictionary_id: OpenCV ArUco predefined dictionary identifier.
        vehicle_marker_id: Target marker ID affixed to vehicle roof.
        marker_size_m: Physical width/length of marker square in meters.
        yaw_offset_rad: Added to the measured marker heading to get car heading.
    """

    def __init__(
        self,
        dictionary_id: Union[int, str] = cv2.aruco.DICT_4X4_50,
        vehicle_marker_id: int = 0,
        marker_size_m: float = 0.05,
        yaw_offset_rad: float = 0.0,
    ) -> None:
        """Initializes detector parameters and dictionary."""
        if isinstance(dictionary_id, str):
            dictionary_id = getattr(cv2.aruco, dictionary_id)
        self.dictionary_id = dictionary_id
        self.vehicle_marker_id = vehicle_marker_id
        self.marker_size_m = marker_size_m
        self.yaw_offset_rad = yaw_offset_rad

        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect_markers(
        self,
        frame: np.ndarray,
    ) -> Detections:
        """Scans frame for ArUco markers and returns detected corners and IDs.

        Args:
            frame: Input BGR image array.

        Returns:
            Tuple of (corners, ids, rejected_candidates).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, rejected = self._detector.detectMarkers(gray)
        return list(corners), ids, list(rejected)

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
        return self.vehicle_pose_from_detections(
            self.detect_markers(frame), transformer
        )

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
        return self.corner_markers_from_detections(
            self.detect_markers(frame), corner_ids
        )

    def vehicle_pose_from_detections(
        self,
        detections: Detections,
        transformer: Optional[HomographyTransformer] = None,
        timestamp: Optional[float] = None,
    ) -> Optional[VehiclePose]:
        """Same as get_vehicle_pose, but reuses an existing detect_markers result.

        Without a calibrated transformer the pose is returned in pixel
        coordinates (x=u, y=v) with yaw in the image frame.
        """
        marker = self._find_marker(detections, self.vehicle_marker_id)
        if marker is None:
            return None
        if timestamp is None:
            timestamp = time.monotonic()

        center = marker.mean(axis=0)
        front = (marker[0] + marker[1]) / 2.0

        if transformer is not None and transformer.is_calibrated:
            cx, cy = transformer.pixel_to_world((center[0], center[1]))
            fx, fy = transformer.pixel_to_world((front[0], front[1]))
        else:
            cx, cy = float(center[0]), float(center[1])
            fx, fy = float(front[0]), float(front[1])

        yaw = math.atan2(fy - cy, fx - cx) + self.yaw_offset_rad
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        return VehiclePose(x=cx, y=cy, yaw=yaw, timestamp=timestamp)

    @staticmethod
    def corner_markers_from_detections(
        detections: Detections,
        corner_ids: Tuple[int, int, int, int],
    ) -> Dict[int, Tuple[float, float]]:
        """Same as get_corner_markers, but reuses an existing detect_markers result."""
        found: Dict[int, Tuple[float, float]] = {}
        for marker_id in corner_ids:
            marker = MarkerTracker._find_marker(detections, marker_id)
            if marker is not None:
                center = marker.mean(axis=0)
                found[marker_id] = (float(center[0]), float(center[1]))
        return found

    @staticmethod
    def _find_marker(detections: Detections, marker_id: int) -> Optional[np.ndarray]:
        """Returns the (4, 2) corner array of the given marker ID, if detected."""
        corners, ids, _ = detections
        if ids is None:
            return None
        matches = np.flatnonzero(ids.ravel() == marker_id)
        if matches.size == 0:
            return None
        return corners[matches[0]].reshape(4, 2)
