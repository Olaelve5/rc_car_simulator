"""Utilities for generating reference waypoints and tracks (circle, figure-8, etc.)."""

from __future__ import annotations

from typing import Tuple
import numpy as np


class TrackGenerator:
    """Generates geometric reference trajectories for tracking and RL benchmarks."""

    @staticmethod
    def generate_circle(
        radius_m: float = 0.6,
        center: Tuple[float, float] = (1.0, 0.75),
        num_points: int = 100,
    ) -> np.ndarray:
        """Generates waypoints for a circular reference path.

        Args:
            radius_m: Circle radius in meters.
            center: Center coordinate (x, y) in arena frame.
            num_points: Number of discrete waypoint samples.

        Returns:
            Array of waypoints of shape (num_points, 2).
        """
        raise NotImplementedError

    @staticmethod
    def generate_figure_eight(
        length_m: float = 1.4,
        width_m: float = 0.8,
        center: Tuple[float, float] = (1.0, 0.75),
        num_points: int = 200,
    ) -> np.ndarray:
        """Generates waypoints for a figure-8 (lemniscate) reference path.

        Args:
            length_m: Total length along major axis in meters.
            width_m: Total width along minor axis in meters.
            center: Center coordinate (x, y) in arena frame.
            num_points: Number of discrete waypoint samples.

        Returns:
            Array of waypoints of shape (num_points, 2).
        """
        raise NotImplementedError

    @staticmethod
    def generate_oval(
        length_m: float = 1.4,
        width_m: float = 0.8,
        center: Tuple[float, float] = (1.0, 0.75),
        num_points: int = 150,
    ) -> np.ndarray:
        """Generates waypoints for an oval / rounded rectangle track.

        Args:
            length_m: Straight segment span in meters.
            width_m: Turn diameter in meters.
            center: Center coordinate (x, y) in arena frame.
            num_points: Number of discrete waypoint samples.

        Returns:
            Array of waypoints of shape (num_points, 2).
        """
        raise NotImplementedError
