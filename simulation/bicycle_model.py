"""Discrete-time 2D kinematic bicycle model for 1:28 scale vehicle simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VehicleState:
    """Dynamic state vector for 2D planar vehicle motion.

    Attributes:
        x: Position in X-axis (meters).
        y: Position in Y-axis (meters).
        yaw: Heading angle in radians relative to X-axis.
        velocity: Forward linear velocity (m/s).
        steering_angle: Front wheel steer angle in radians.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    velocity: float = 0.0
    steering_angle: float = 0.0


@dataclass
class VehicleDynamicsParams:
    """Physical parameters representing a 1:28 scale chassis.

    Attributes:
        wheelbase_m: Distance between front and rear axle centers (meters).
        max_steer_rad: Maximum physical front wheel steering angle (radians).
        max_velocity_mps: Maximum forward speed limit (m/s).
        max_accel_mps2: Maximum linear acceleration limit (m/s^2).
        dt: Integration time step (seconds).
    """

    wheelbase_m: float = 0.098  # ~98mm standard 1:28 wheelbase
    max_steer_rad: float = 0.5236  # ~30 deg
    max_velocity_mps: float = 1.50
    max_accel_mps2: float = 3.00
    dt: float = 0.02  # 50 Hz


class KinematicBicycleModel:
    """Discrete kinematic bicycle model interface for vehicle state propagation.

    Attributes:
        params: Physical vehicle parameter specifications.
        state: Current dynamic state of the simulated vehicle.
    """

    def __init__(
        self,
        params: Optional[VehicleDynamicsParams] = None,
        initial_state: Optional[VehicleState] = None,
    ) -> None:
        """Initializes vehicle physical parameters and initial state."""
        self.params = params or VehicleDynamicsParams()
        self.state = initial_state or VehicleState()

    def reset(self, state: Optional[VehicleState] = None) -> VehicleState:
        """Resets the model to a specified or origin state.

        Args:
            state: Optional custom state to reset to.

        Returns:
            The newly reset VehicleState.
        """
        raise NotImplementedError

    def step(
        self,
        steering_target: float,
        throttle_input: float,
        dt: Optional[float] = None,
    ) -> VehicleState:
        """Performs a single discrete forward integration time step.

        Args:
            steering_target: Desired steering angle or normalized steer demand.
            throttle_input: Acceleration / throttle command.
            dt: Optional custom integration time step override.

        Returns:
            Updated VehicleState after time dt.
        """
        raise NotImplementedError

    def get_state(self) -> VehicleState:
        """Returns the current state snapshot.

        Returns:
            VehicleState instance.
        """
        raise NotImplementedError
