"""Stanley non-linear path tracking controller stub."""

from __future__ import annotations

import numpy as np

from .base_controller import BaseController


class StanleyController(BaseController):
    """Stanley formulation for front-axle path tracking.

    Combines heading alignment error and non-linear cross-track error damping.

    Attributes:
        k_path: Cross-track error gain.
        k_soft: Softening constant to prevent singularity at zero velocity.
        wheelbase_m: Wheelbase of 1:28 vehicle chassis in meters.
    """

    def __init__(
        self,
        k_path: float = 1.5,
        k_soft: float = 0.5,
        wheelbase_m: float = 0.098,
        kp_speed: float = 1.0,
    ) -> None:
        """Initializes Stanley geometry parameters and controller gains."""
        super().__init__(name="StanleyController")
        self.k_path = k_path
        self.k_soft = k_soft
        self.wheelbase_m = wheelbase_m
        self.kp_speed = kp_speed

    def compute_action(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Computes Stanley steering angle and longitudinal throttle command.

        Args:
            state: Current state vector [x, y, yaw, velocity].
            reference: Target path reference [ref_x, ref_y, ref_yaw, ref_v].

        Returns:
            Control action array [steering, throttle] in normalized range [-1.0, 1.0].
        """
        pass

    def reset(self) -> None:
        """Resets internal controller states."""
        pass
