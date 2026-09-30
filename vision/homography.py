"""Perspective transformation and 4-point homography mapping for arena calibration."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class HomographyTransformer:
    """Manages geometric transformations between image pixel and arena coordinates.

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
        raise NotImplementedError

    def pixel_to_world(self, pixel_coord: Tuple[float, float]) -> Tuple[float, float]:
        """Maps single pixel coordinate (u, v) into arena metric coordinates (x, y).

        Args:
            pixel_coord: Point in pixel coordinates (u, v).

        Returns:
            Point in arena world coordinates (x, y) in meters.
        """
        raise NotImplementedError

    def world_to_pixel(self, world_coord: Tuple[float, float]) -> Tuple[int, int]:
        """Maps metric world coordinate (x, y) into camera pixel coordinates (u, v).

        Args:
            world_coord: Point in arena metric coordinates (x, y).

        Returns:
            Nearest integer pixel coordinate (u, v).
        """
        raise NotImplementedError

    def warp_image(
        self,
        image: np.ndarray,
        output_resolution: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Applies bird's-eye perspective warp to raw camera frame.

        Args:
            image: Raw input image from camera (H, W, C).
            output_resolution: Target output resolution (width, height).

        Returns:
            Rectified bird's-eye view image.
        """
        raise NotImplementedError
