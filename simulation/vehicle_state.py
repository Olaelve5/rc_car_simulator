"""Vehicle state representation for 2D kinematic and dynamic simulations."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class VehicleState:
    """Dynamic state vector for 2D planar vehicle motion.

    Attributes:
        x: Position along arena X-axis in meters.
        y: Position along arena Y-axis in meters.
        yaw: Heading orientation angle in radians [-pi, pi] relative to X-axis.
        velocity: Forward linear velocity in m/s (positive = forward,
            negative = reverse).
        steering_angle: Front wheel steering angle in radians (positive = left).
        v_x: Longitudinal velocity in vehicle body frame (m/s).
        v_y: Lateral sliding velocity in vehicle body frame (m/s).
        yaw_rate: Yaw rotational velocity r = d(yaw)/dt in rad/s.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    velocity: float = 0.0
    steering_angle: float = 0.0
    v_x: float = 0.0
    v_y: float = 0.0
    yaw_rate: float = 0.0

    @property
    def drift_angle_rad(self) -> float:
        """Returns body sideslip/drift angle beta = atan2(v_y, |v_x|) in radians."""
        if abs(self.v_x) < 0.05 and abs(self.v_y) < 0.05:
            return 0.0
        return float(math.atan2(self.v_y, max(abs(self.v_x), 0.01)))

    def copy(self) -> VehicleState:
        """Returns an independent deep copy of this state."""
        return VehicleState(
            x=self.x,
            y=self.y,
            yaw=self.yaw,
            velocity=self.velocity,
            steering_angle=self.steering_angle,
            v_x=self.v_x,
            v_y=self.v_y,
            yaw_rate=self.yaw_rate,
        )

    def to_numpy(self) -> np.ndarray:
        """Exports state as a 1D NumPy array [x, y, yaw, velocity]."""
        return np.array([self.x, self.y, self.yaw, self.velocity], dtype=np.float64)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, steering_angle: float = 0.0) -> VehicleState:
        """Constructs VehicleState from a 1D array [x, y, yaw, velocity]."""
        v = float(arr[3])
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            yaw=float(arr[2]),
            velocity=v,
            steering_angle=float(steering_angle),
            v_x=v,
            v_y=0.0,
            yaw_rate=0.0,
        )
