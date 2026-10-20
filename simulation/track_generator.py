"""Utilities for generating reference trajectories and computing tracking errors."""

from __future__ import annotations

import math
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
        thetas = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        xs = center[0] + radius_m * np.cos(thetas)
        ys = center[1] + radius_m * np.sin(thetas)
        return np.column_stack((xs, ys)).astype(np.float64)

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
        ts = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)
        # Gerono lemniscate formulation centered at (cx, cy)
        xs = center[0] + (length_m / 2.0) * np.sin(ts)
        ys = center[1] + (width_m / 4.0) * np.sin(2.0 * ts)
        return np.column_stack((xs, ys)).astype(np.float64)

    @staticmethod
    def generate_oval(
        length_m: float = 1.4,
        width_m: float = 0.8,
        center: Tuple[float, float] = (1.0, 0.75),
        num_points: int = 150,
    ) -> np.ndarray:
        """Generates waypoints for an oval / rounded rectangle track.

        Args:
            length_m: Total outer length along X in meters.
            width_m: Total outer width along Y in meters.
            center: Center coordinate (x, y) in arena frame.
            num_points: Number of discrete waypoint samples.

        Returns:
            Array of waypoints of shape (num_points, 2).
        """
        radius = width_m / 2.0
        straight = max(length_m - 2.0 * radius, 0.1)

        # Distribute points proportionally to path lengths
        arc_len = np.pi * radius
        total_perim = 2.0 * straight + 2.0 * arc_len
        n_straight = max(int(round(num_points * (straight / total_perim))), 2)
        n_arc = max(int(round(num_points * (arc_len / total_perim))), 4)

        cx, cy = center
        half_s = straight / 2.0

        # 1. Bottom straight (moving right: +X)
        pts_bottom = np.column_stack(
            (
                np.linspace(cx - half_s, cx + half_s, n_straight, endpoint=False),
                np.full(n_straight, cy - radius),
            )
        )

        # 2. Right semicircle (counter-clockwise: -pi/2 to +pi/2)
        theta_right = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_arc, endpoint=False)
        pts_right = np.column_stack(
            (
                (cx + half_s) + radius * np.cos(theta_right),
                cy + radius * np.sin(theta_right),
            )
        )

        # 3. Top straight (moving left: -X)
        pts_top = np.column_stack(
            (
                np.linspace(cx + half_s, cx - half_s, n_straight, endpoint=False),
                np.full(n_straight, cy + radius),
            )
        )

        # 4. Left semicircle (counter-clockwise: pi/2 to 3pi/2)
        theta_left = np.linspace(
            np.pi / 2.0, 3.0 * np.pi / 2.0, n_arc, endpoint=False
        )
        pts_left = np.column_stack(
            (
                (cx - half_s) + radius * np.cos(theta_left),
                cy + radius * np.sin(theta_left),
            )
        )

        waypoints = np.vstack((pts_bottom, pts_right, pts_top, pts_left))
        return waypoints.astype(np.float64)

    @staticmethod
    def compute_tracking_errors(
        waypoints: np.ndarray,
        x: float,
        y: float,
        yaw: float,
    ) -> Tuple[float, float, int]:
        """Computes signed cross-track error (CTE) and heading error against path.

        Args:
            waypoints: Array of shape (N, 2) containing path (x, y) coordinates.
            x: Vehicle position X in arena frame (meters).
            y: Vehicle position Y in arena frame (meters).
            yaw: Vehicle heading orientation in radians [-pi, pi].

        Returns:
            Tuple of:
                - cross_track_error (signed float, meters: positive = left of track)
                - heading_error (signed float, radians: [-pi, pi])
                - closest_waypoint_index (int)
        """
        if len(waypoints) < 2:
            return 0.0, 0.0, 0

        # Vectorized Euclidean distance from vehicle to all waypoints
        diffs = waypoints - np.array([x, y])
        dists_sq = np.sum(diffs**2, axis=1)
        idx = int(np.argmin(dists_sq))

        # Forward tangent vector from closest waypoint to next waypoint
        next_idx = (idx + 1) % len(waypoints)
        tangent = waypoints[next_idx] - waypoints[idx]
        norm = np.linalg.norm(tangent)
        if norm < 1e-6:
            tangent = np.array([1.0, 0.0])
        else:
            tangent = tangent / norm

        # Path normal vector (rotated 90 deg counter-clockwise: [-t_y, t_x])
        normal = np.array([-tangent[1], tangent[0]])

        # Vector from closest waypoint to vehicle
        vec_to_car = np.array([x, y]) - waypoints[idx]

        # Signed cross-track error: projection onto path normal
        cross_track_error = float(np.dot(vec_to_car, normal))

        # Path tangent heading
        path_yaw = math.atan2(tangent[1], tangent[0])

        # Heading error wrapped into [-pi, pi]
        raw_heading_err = yaw - path_yaw
        heading_error = float(
            (raw_heading_err + math.pi) % (2.0 * math.pi) - math.pi
        )

        return cross_track_error, heading_error, idx
