"""Discrete-time 2D kinematic bicycle model for 1:28 scale vehicle simulation.

This module provides the geometry-based vehicle model with Center-of-Gravity (CoG)
reference. Also re-exports VehicleState, VehicleDynamicsParams, and the model factory
for backward compatibility across the simulation subsystem.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union
import numpy as np

from config.settings import (
    DEFAULT_SETTINGS,
    VehicleDynamicsConfig,
    VehicleDynamicsParams,
)
from .dynamic_bicycle_model import DynamicBicycleModel
from .vehicle_state import VehicleState

__all__ = [
    "DynamicBicycleModel",
    "KinematicBicycleModel",
    "VehicleDynamicsConfig",
    "VehicleDynamicsParams",
    "VehicleState",
    "create_vehicle_model",
]


class KinematicBicycleModel:
    """Discrete kinematic bicycle model with Center-of-Gravity (CoG) reference.

    Translates steering commands and throttle demands into updated vehicle poses,
    assuming pure rolling without lateral tire slip (ideal geometric tracking).
    """

    def __init__(
        self,
        params: Optional[VehicleDynamicsConfig] = None,
        initial_state: Optional[VehicleState] = None,
    ) -> None:
        """Initializes vehicle physical parameters and dynamic state."""
        self.params = params or DEFAULT_SETTINGS.vehicle
        self.state = initial_state.copy() if initial_state else VehicleState()
        self.model_name = "Kinematic"

    def reset(self, state: Optional[VehicleState] = None) -> VehicleState:
        """Resets the vehicle state to a specified pose or clean origin."""
        if state is not None:
            self.state = state.copy()
        else:
            self.state = VehicleState()
        return self.state.copy()

    def get_state(self) -> VehicleState:
        """Returns a snapshot copy of the current vehicle state."""
        return self.state.copy()

    def step(
        self,
        steering_target: float,
        throttle_input: float,
        dt: Optional[float] = None,
    ) -> VehicleState:
        """Advances vehicle dynamics forward by time dt given control inputs."""
        sub_dt = dt if dt is not None else self.params.dt

        # 1. Steering actuator dynamics
        self.state.steering_angle = self._update_steering(steering_target, sub_dt)

        # 2. Longitudinal acceleration calculation
        accel = self._compute_acceleration(
            throttle_input, self.state.velocity, sub_dt
        )

        # 3. Numerical state integration (RK4)
        self._integrate_kinematics(sub_dt, self.state.steering_angle, accel)

        # 4. Heading normalization
        self.state.yaw = self._normalize_angle(self.state.yaw)

        # Update body velocities for consistent telemetry
        self.state.v_x = self.state.velocity
        self.state.v_y = 0.0
        beta = self._compute_sideslip_angle(self.state.steering_angle)
        self.state.yaw_rate = float(
            (self.state.velocity / self.params.cog_to_rear_m) * math.sin(beta)
        )

        return self.state.copy()

    def _update_steering(self, steering_target: float, dt: float) -> float:
        """Applies physical bounds and servo slew-rate limits to steering angle."""
        target = float(
            np.clip(
                steering_target,
                -self.params.max_steer_rad,
                self.params.max_steer_rad,
            )
        )
        max_delta = self.params.max_steer_rate_radps * dt
        delta = float(
            np.clip(target - self.state.steering_angle, -max_delta, max_delta)
        )
        return self.state.steering_angle + delta

    def _compute_acceleration(
        self,
        throttle_input: float,
        current_velocity: float,
        dt: float,
    ) -> float:
        """Computes net linear acceleration accounting for drive,
        braking, and friction.
        """
        throttle = float(np.clip(throttle_input, -1.0, 1.0))

        if abs(throttle) < self.params.min_throttle_deadband:
            throttle = 0.0

        if throttle > 0.0:
            if current_velocity >= -0.05:
                a_motor = throttle * self.params.max_accel_mps2
            else:
                a_motor = throttle * self.params.max_decel_mps2
        elif throttle < 0.0:
            if current_velocity > 0.05:
                a_motor = throttle * self.params.max_decel_mps2
            else:
                a_motor = throttle * (self.params.max_accel_mps2 * 0.6)
        else:
            a_motor = 0.0

        if a_motor == 0.0:
            if abs(current_velocity) < 0.02:
                return -current_velocity / max(dt, 1e-4)

        if abs(current_velocity) > 1e-4:
            a_roll = -self.params.rolling_resistance_coeff * np.sign(current_velocity)
        else:
            a_roll = 0.0

        a_drag = -self.params.drag_coeff * current_velocity

        return float(a_motor + a_roll + a_drag)

    def _compute_sideslip_angle(self, steering_angle: float) -> float:
        """Calculates vehicle body sideslip angle (beta) at Center of Gravity."""
        tan_delta = math.tan(steering_angle)
        ratio = self.params.cog_to_rear_m / self.params.wheelbase_m
        return math.atan(ratio * tan_delta)

    def _compute_derivatives(
        self,
        velocity: float,
        yaw: float,
        steering_angle: float,
        acceleration: float,
    ) -> Tuple[float, float, float, float]:
        """Evaluates state time derivatives [x_dot, y_dot, yaw_dot, v_dot]."""
        beta = self._compute_sideslip_angle(steering_angle)
        x_dot = velocity * math.cos(yaw + beta)
        y_dot = velocity * math.sin(yaw + beta)
        yaw_dot = (velocity / self.params.cog_to_rear_m) * math.sin(beta)
        v_dot = acceleration
        return x_dot, y_dot, yaw_dot, v_dot

    def _integrate_kinematics(
        self,
        dt: float,
        steering_angle: float,
        acceleration: float,
    ) -> None:
        """Integrates vehicle state over dt using Runge-Kutta 4th order (RK4)."""
        x = self.state.x
        y = self.state.y
        yaw = self.state.yaw
        v = self.state.velocity

        dx1, dy1, dyaw1, dv1 = self._compute_derivatives(
            v, yaw, steering_angle, acceleration
        )
        v2 = v + 0.5 * dt * dv1
        yaw2 = yaw + 0.5 * dt * dyaw1
        dx2, dy2, dyaw2, dv2 = self._compute_derivatives(
            v2, yaw2, steering_angle, acceleration
        )
        v3 = v + 0.5 * dt * dv2
        yaw3 = yaw + 0.5 * dt * dyaw2
        dx3, dy3, dyaw3, dv3 = self._compute_derivatives(
            v3, yaw3, steering_angle, acceleration
        )
        v4 = v + dt * dv3
        yaw4 = yaw + dt * dyaw3
        dx4, dy4, dyaw4, dv4 = self._compute_derivatives(
            v4, yaw4, steering_angle, acceleration
        )

        self.state.x = x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        self.state.y = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
        self.state.yaw = yaw + (dt / 6.0) * (
            dyaw1 + 2.0 * dyaw2 + 2.0 * dyaw3 + dyaw4
        )

        new_v = v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        self.state.velocity = float(
            np.clip(
                new_v,
                -self.params.max_reverse_velocity_mps,
                self.params.max_velocity_mps,
            )
        )
        if abs(self.state.velocity) < 1e-4:
            self.state.velocity = 0.0

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wraps any angle in radians to the interval [-pi, pi]."""
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def create_vehicle_model(
    model_type: str = "dynamic",
    params: Optional[VehicleDynamicsConfig] = None,
    initial_state: Optional[VehicleState] = None,
) -> Union[KinematicBicycleModel, DynamicBicycleModel]:
    """Factory helper to instantiate either Kinematic or Dynamic vehicle models.

    Args:
        model_type: Either 'dynamic' or 'kinematic' (case-insensitive).
        params: Optional physical vehicle parameters.
        initial_state: Optional initial vehicle pose.

    Returns:
        Configured vehicle dynamics model instance.
    """
    kind = model_type.strip().lower()
    if kind == "dynamic":
        return DynamicBicycleModel(params=params, initial_state=initial_state)
    elif kind == "kinematic":
        return KinematicBicycleModel(params=params, initial_state=initial_state)
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected 'dynamic' or 'kinematic'."
        )
