"""Perspective transformation and 4-point homography mapping for arena calibration."""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class HomographyTransformer:
    """Manages geometric transformations between image pixel and arena coordinates.

    World frame convention (matches the bicycle model): origin at the first
    corner marker, X to the right along the arena width, Y along the arena
    height, yaw measured counter-clockwise from +X. Default world corners are
    ordered (0, 0), (W, 0), (W, H), (0, H), i.e. counter-clockwise seen from
    above.

    Attributes:
        arena_width_m: Physical width of rectangular arena in meters.
        arena_height_m: Physical height of rectangular arena in meters.
    """

    def __init__(
        self,
        arena_width_m: float = 2.0,
        arena_height_m: float = 1.5,
    ) -> None:
        """Initializes arena dimensions and placeholder transformation matrices."""
        self.arena_width_m = arena_width_m
        self.arena_height_m = arena_height_m
        self._matrix: Optional[np.ndarray] = None
        self._inv_matrix: Optional[np.ndarray] = None

    @property
    def is_calibrated(self) -> bool:
        """True once a homography matrix has been computed."""
        return self._matrix is not None

    @property
    def matrix(self) -> Optional[np.ndarray]:
        """The current pixel -> world 3x3 homography, or None if uncalibrated."""
        return self._matrix

    def default_world_corners(self) -> np.ndarray:
        """Returns arena corners (0,0), (W,0), (W,H), (0,H) as a (4, 2) array."""
        w, h = self.arena_width_m, self.arena_height_m
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float32)

    def compute_matrix(
        self,
        pixel_corners: np.ndarray,
        world_corners: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes 3x3 homography matrix from 4 arena corner coordinates.

        Args:
            pixel_corners: Array of shape (4, 2) in image pixel coordinates (u, v).
            world_corners: Optional array of shape (4, 2) in world coordinates (x, y).
                           If omitted, defaults to standard rectangular arena bounds.

        Returns:
            The calculated 3x3 homography matrix.
        """
        src = np.asarray(pixel_corners, dtype=np.float32).reshape(4, 2)
        if world_corners is None:
            dst = self.default_world_corners()
        else:
            dst = np.asarray(world_corners, dtype=np.float32).reshape(4, 2)

        self._matrix = cv2.getPerspectiveTransform(src, dst)
        self._inv_matrix = np.linalg.inv(self._matrix)
        return self._matrix

    def pixel_to_world(self, pixel_coord: Tuple[float, float]) -> Tuple[float, float]:
        """Maps single pixel coordinate (u, v) into arena metric coordinates (x, y).

        Args:
            pixel_coord: Point in pixel coordinates (u, v).

        Returns:
            Point in arena world coordinates (x, y) in meters.
        """
        if self._matrix is None:
            raise RuntimeError("Homography not computed; call compute_matrix first.")
        x, y = self._apply(self._matrix, pixel_coord)
        return float(x), float(y)

    def world_to_pixel(self, world_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Maps metric world coordinate (x, y) into camera pixel coordinates (u, v).

        Args:
            world_coord: Point in arena metric coordinates (x, y).

        Returns:
            Nearest integer pixel coordinate (u, v).
        """
        if self._inv_matrix is None:
            raise RuntimeError("Homography not computed; call compute_matrix first.")
        u, v = self._apply(self._inv_matrix, world_coord)
        return int(round(u)), int(round(v))

    def warp_image(
        self,
        image: np.ndarray,
        output_resolution: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Applies bird's-eye perspective warp to raw camera frame.

        The output is drawn with world Y pointing up, so it looks like a top-down
        map of the arena with the origin in the bottom-left corner.

        Args:
            image: Raw input image from camera (H, W, C).
            output_resolution: Target output resolution (width, height).

        Returns:
            Rectified bird's-eye view image.
        """
        if self._matrix is None:
            raise RuntimeError("Homography not computed; call compute_matrix first.")
        if output_resolution is None:
            px_per_m = 400.0
            output_resolution = (
                int(round(self.arena_width_m * px_per_m)),
                int(round(self.arena_height_m * px_per_m)),
            )
        out_w, out_h = output_resolution
        sx = out_w / self.arena_width_m
        sy = out_h / self.arena_height_m
        # World (meters, Y up) -> output image pixels (Y down).
        world_to_out = np.array(
            [[sx, 0.0, 0.0], [0.0, -sy, out_h], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        return cv2.warpPerspective(image, world_to_out @ self._matrix, (out_w, out_h))

    @staticmethod
    def _apply(matrix: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
        """Applies a 3x3 projective transform to a single 2D point."""
        p = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
        return p[0] / p[2], p[1] / p[2]
