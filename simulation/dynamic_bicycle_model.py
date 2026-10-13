"""Discrete-time 2D dynamic bicycle model with non-linear lateral tire slip.

This module models lateral tire forces, slip angles (alpha), yaw acceleration,
weight distribution, and tire saturation (tanh friction limit).
Allows drifting, sliding, and realistic countersteering while blending
to kinematic motion at crawling speeds to prevent low-speed singularities.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from config.settings import DEFAULT_SETTINGS, VehicleDynamicsConfig
from .vehicle_state import VehicleState


class DynamicBicycleModel:
    """Discrete 2D dynamic bicycle model with non-linear lateral tire slip.

    Attributes:
        params: Physical and tire dynamics parameters.
        state: Dynamic vehicle state vector [x, y, yaw, v_x, v_y, yaw_rate].
    """

    def __init__(
        self,
        params: Optional[VehicleDynamicsConfig] = None,
        initial_state: Optional[VehicleState] = None,
    ) -> None:
        """Initializes dynamic vehicle parameters and internal states."""
        self.params = params or DEFAULT_SETTINGS.vehicle
        self.state = initial_state.copy() if initial_state else VehicleState()
        if self.state.v_x == 0.0 and self.state.velocity != 0.0:
            self.state.v_x = self.state.velocity
        self.model_name = "Dynamic"

    def reset(self, state: Optional[VehicleState] = None) -> VehicleState:
        """Resets the dynamic vehicle state."""
        if state is not None:
            self.state = state.copy()
            if self.state.v_x == 0.0 and self.state.velocity != 0.0:
                self.state.v_x = self.state.velocity
        else:
            self.state = VehicleState()
        return self.state.copy()

    def get_state(self) -> VehicleState:
        """Returns a snapshot copy of current state."""
        return self.state.copy()

    def step(
        self,
        steering_target: float,
        throttle_input: float,
        dt: Optional[float] = None,
    ) -> VehicleState:
        """Advances dynamic vehicle physics forward by time dt."""
        sub_dt = dt if dt is not None else self.params.dt
        self.state.steering_angle = self._update_steering(steering_target, sub_dt)
        a_long = self._compute_acceleration(
            throttle_input, self.state.v_x, sub_dt
        )
        self._integrate_dynamics(sub_dt, self.state.steering_angle, a_long)
        self.state.yaw = self._normalize_angle(self.state.yaw)
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
        """Computes longitudinal acceleration command."""
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

    def _compute_derivatives(
        self,
        v_x: float,
        v_y: float,
        yaw: float,
        r: float,
        delta: float,
        a_long: float,
        dt: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """Evaluates dynamic state derivatives [dx, dy, dyaw, dvx, dvy, dr]."""
        m = self.params.mass_kg
        iz = self.params.iz_kgm2
        lf = self.params.cog_to_front_m
        lr = self.params.cog_to_rear_m
        wheelbase = self.params.wheelbase_m
        mu = self.params.tire_friction_mu
        gravity = 9.81

        # Vertical axle loads
        f_zf = m * gravity * (lr / wheelbase)
        f_zr = m * gravity * (lf / wheelbase)

        # Longitudinal drive force
        f_x = m * a_long

        # Slip angles
        v_eff = max(abs(v_x), 0.06)
        if v_x >= 0.0:
            alpha_f = delta - math.atan2(v_y + lf * r, v_eff)
            alpha_r = -math.atan2(v_y - lr * r, v_eff)
        else:
            alpha_f = delta - math.atan2(v_y - lf * r, v_eff)
            alpha_r = -math.atan2(v_y + lr * r, v_eff)

        # Non-linear tire lateral forces with smooth tanh saturation
        f_yf_max = max(mu * f_zf, 1e-3)
        f_yr_max = max(mu * f_zr, 1e-3)

        f_yf = f_yf_max * math.tanh(
            (self.params.cornering_stiffness_front * alpha_f) / f_yf_max
        )
        f_yr = f_yr_max * math.tanh(
            (self.params.cornering_stiffness_rear * alpha_r) / f_yr_max
        )

        # Equations of motion in vehicle body frame
        dv_x = ((f_x - f_yf * math.sin(delta)) / m) + v_y * r
        dv_y = ((f_yf * math.cos(delta) + f_yr) / m) - v_x * r
        dr = (lf * f_yf * math.cos(delta) - lr * f_yr) / iz

        # Low-speed kinematic blending to eliminate standstill singularities
        v_blend = self.params.kinematic_blend_speed_mps
        if abs(v_x) < v_blend:
            tan_delta = math.tan(delta)
            beta_kin = math.atan((lr / wheelbase) * tan_delta)
            r_kin = (v_x / lr) * math.sin(beta_kin)
            vy_kin = v_x * math.sin(beta_kin)

            blend_w = min(1.0, max(0.0, abs(v_x) / max(v_blend, 1e-4)))
            dv_y = (1.0 - blend_w) * ((vy_kin - v_y) / max(dt, 1e-4)) + blend_w * dv_y
            dr = (1.0 - blend_w) * ((r_kin - r) / max(dt, 1e-4)) + blend_w * dr

        # World coordinates derivatives
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        dx = v_x * cos_yaw - v_y * sin_yaw
        dy = v_x * sin_yaw + v_y * cos_yaw
        dyaw = r

        return dx, dy, dyaw, dv_x, dv_y, dr

    def _integrate_dynamics(
        self,
        dt: float,
        steering_angle: float,
        a_long: float,
    ) -> None:
        """RK4 integration of the 6D dynamic state."""
        x = self.state.x
        y = self.state.y
        yaw = self.state.yaw
        vx = self.state.v_x
        vy = self.state.v_y
        r = self.state.yaw_rate

        dx1, dy1, dyaw1, dvx1, dvy1, dr1 = self._compute_derivatives(
            vx, vy, yaw, r, steering_angle, a_long, dt
        )
        vx2 = vx + 0.5 * dt * dvx1
        vy2 = vy + 0.5 * dt * dvy1
        yaw2 = yaw + 0.5 * dt * dyaw1
        r2 = r + 0.5 * dt * dr1
        dx2, dy2, dyaw2, dvx2, dvy2, dr2 = self._compute_derivatives(
            vx2, vy2, yaw2, r2, steering_angle, a_long, dt
        )
        vx3 = vx + 0.5 * dt * dvx2
        vy3 = vy + 0.5 * dt * dvy2
        yaw3 = yaw + 0.5 * dt * dyaw2
        r3 = r + 0.5 * dt * dr2
        dx3, dy3, dyaw3, dvx3, dvy3, dr3 = self._compute_derivatives(
            vx3, vy3, yaw3, r3, steering_angle, a_long, dt
        )
        vx4 = vx + dt * dvx3
        vy4 = vy + dt * dvy3
        yaw4 = yaw + dt * dyaw3
        r4 = r + dt * dr3
        dx4, dy4, dyaw4, dvx4, dvy4, dr4 = self._compute_derivatives(
            vx4, vy4, yaw4, r4, steering_angle, a_long, dt
        )

        self.state.x = x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4)
        self.state.y = y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4)
        self.state.yaw = yaw + (dt / 6.0) * (dyaw1 + 2.0 * dyaw2 + 2.0 * dyaw3 + dy4)
        new_vx = vx + (dt / 6.0) * (dvx1 + 2.0 * dvx2 + 2.0 * dvx3 + dvx4)
        new_vy = vy + (dt / 6.0) * (dvy1 + 2.0 * dvy2 + 2.0 * dvy3 + dvy4)
        new_r = r + (dt / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)

        self.state.v_x = float(
            np.clip(
                new_vx,
                -self.params.max_reverse_velocity_mps,
                self.params.max_velocity_mps,
            )
        )
        self.state.v_y = float(new_vy)
        self.state.yaw_rate = float(new_r)

        if abs(self.state.v_x) < 0.02 and abs(a_long) < 0.05:
            self.state.v_x = 0.0
            self.state.v_y = 0.0
            self.state.yaw_rate = 0.0

        spd = math.hypot(self.state.v_x, self.state.v_y)
        self.state.velocity = float(spd if self.state.v_x >= 0.0 else -spd)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Wraps any angle in radians to the interval [-pi, pi]."""
        return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
