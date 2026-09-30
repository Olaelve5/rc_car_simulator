"""Proportional-Integral-Derivative (PID) tracking controller stub."""

from __future__ import annotations

import numpy as np

from .base_controller import BaseController


class PIDController(BaseController):
    """Classical PID controller for decoupled lateral (cross-track)
    and longitudinal (speed) tracking.

    Attributes:
        kp_steer: Proportional gain for lateral error.
        ki_steer: Integral gain for lateral error.
        kd_steer: Derivative gain for lateral error.
        kp_speed: Proportional gain for speed tracking.
        ki_speed: Integral gain for speed tracking.
        kd_speed: Derivative gain for speed tracking.
    """

    def __init__(
        self,
        kp_steer: float = 1.0,
        ki_steer: float = 0.0,
        kd_steer: float = 0.1,
        kp_speed: float = 1.0,
        ki_speed: float = 0.1,
        kd_speed: float = 0.0,
        dt: float = 0.02,
    ) -> None:
        """Initializes gains, sampling time, and error integrators."""
        super().__init__(name="PIDController")
        self.kp_steer = kp_steer
        self.ki_steer = ki_steer
        self.kd_steer = kd_steer
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.dt = dt

    def compute_action(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Computes PID control action demands for steering and throttle.

        Args:
            state: Current state vector [x, y, yaw, velocity].
            reference: Target reference waypoint [ref_x, ref_y, ref_yaw, ref_v].

        Returns:
            Control action array [steering, throttle] in normalized range [-1.0, 1.0].
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clears integral error accumulators and previous error derivatives."""
        pass
