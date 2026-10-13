"""Proportional-Integral-Derivative (PID) tracking controller for 1:28 RC testbed.

Implements decoupled lateral (cross-track error + heading alignment) and
longitudinal (linear velocity) PID control loops with anti-windup clamping
and derivative filtering.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config.settings import DEFAULT_SETTINGS  # noqa: E402
from simulation.track_generator import TrackGenerator  # noqa: E402
from simulation.vehicle_state import VehicleState  # noqa: E402
from .base_controller import BaseController  # noqa: E402

__all__ = ["PIDController"]


class PIDController(BaseController):
    """Classical PID controller for decoupled lateral and longitudinal tracking.

    Lateral Control:
        error_lat = e_ct + k_head * e_psi
        delta = -(kp * error_lat + ki * int(error_lat) + kd * d(error_lat)/dt)

    Longitudinal Control:
        error_speed = v_target - v
        throttle = kp * error_speed + ki * int(error_speed) + kd * d_speed + ff

    Attributes:
        kp_steer: Proportional gain for lateral tracking error.
        ki_steer: Integral gain for lateral tracking error.
        kd_steer: Derivative gain for lateral tracking error damping.
        k_head: Weighting gain for heading alignment error.
        kp_speed: Proportional gain for longitudinal speed regulation.
        ki_speed: Integral gain for longitudinal speed regulation.
        kd_speed: Derivative gain for longitudinal speed regulation.
        target_velocity: Cruising forward speed demand (m/s).
        max_steer_rad: Maximum physical front-wheel steering angle in radians.
        dt: Control loop sampling time (seconds).
        waypoints: Optional pre-loaded reference path waypoints of shape (N, 2).
    """

    def __init__(
        self,
        kp_steer: float = 1.8,
        ki_steer: float = 0.05,
        kd_steer: float = 0.20,
        k_head: float = 0.90,
        kp_speed: float = 1.8,
        ki_speed: float = 0.20,
        kd_speed: float = 0.05,
        target_velocity: float = 0.55,
        max_steer_rad: Optional[float] = None,
        waypoints: Optional[np.ndarray] = None,
        dt: float = 0.02,
        adapt_speed_on_curvature: bool = True,
    ) -> None:
        """Initializes gains, sampling time, and error integrators."""
        super().__init__(name="PIDController")
        veh_cfg = DEFAULT_SETTINGS.vehicle

        # Lateral steering gains
        self.kp_steer = float(kp_steer)
        self.ki_steer = float(ki_steer)
        self.kd_steer = float(kd_steer)
        self.k_head = float(k_head)

        # Longitudinal velocity gains
        self.kp_speed = float(kp_speed)
        self.ki_speed = float(ki_speed)
        self.kd_speed = float(kd_speed)

        self.target_velocity = float(target_velocity)
        self.max_steer_rad = (
            float(max_steer_rad)
            if max_steer_rad is not None
            else veh_cfg.max_steer_rad
        )
        self.dt = float(dt)
        self.adapt_speed_on_curvature = adapt_speed_on_curvature

        self.waypoints: Optional[np.ndarray] = None
        if waypoints is not None:
            self.set_waypoints(waypoints)

        # Integrator accumulators and previous errors
        self._lat_integral: float = 0.0
        self._lat_prev_error: Optional[float] = None

        self._speed_integral: float = 0.0
        self._speed_prev_error: Optional[float] = None

        self._last_diagnostics: Dict[str, Any] = {}

    def set_waypoints(self, waypoints: np.ndarray) -> None:
        """Sets reference path waypoints of shape (N, 2)."""
        if len(waypoints) < 2:
            raise ValueError("Waypoints array must contain at least 2 points.")
        self.waypoints = np.asarray(waypoints, dtype=np.float64)

    def reset(self) -> None:
        """Clears integral error accumulators and previous error derivatives."""
        self._lat_integral = 0.0
        self._lat_prev_error = None
        self._speed_integral = 0.0
        self._speed_prev_error = None
        self._last_diagnostics.clear()

    def compute_action(
        self,
        state: Union[np.ndarray, VehicleState],
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes PID control action demands for steering and throttle.

        Args:
            state: Current state vector [x, y, yaw, velocity] (or Box(6) Gym obs,
                or VehicleState instance).
            reference: Target path reference waypoints (N, 2), or single target
                slice [ref_x, ref_y, ref_yaw, (ref_v)]. If None, uses pre-configured
                self.waypoints.

        Returns:
            Control action array [steering, throttle] in normalized range [-1.0, 1.0].
        """
        # 1. Extract vehicle state
        x, y, yaw, v = self._extract_state(state)

        # 2. Resolve reference waypoints
        pts = self._resolve_waypoints(reference)
        if pts is None:
            raise ValueError(
                "PIDController requires reference path waypoints. "
                "Pass waypoints in compute_action() or initialize with waypoints."
            )

        # 3. Compute tracking errors via TrackGenerator
        cte, heading_err, closest_idx = TrackGenerator.compute_tracking_errors(
            pts, x, y, yaw
        )

        # 4. Lateral PID Control Law
        # Composite lateral tracking error (cross-track deviation + heading error)
        error_lat = cte + self.k_head * heading_err

        # Error derivative (finite difference)
        if self._lat_prev_error is None:
            d_lat = 0.0
        else:
            d_lat = (error_lat - self._lat_prev_error) / max(self.dt, 1e-4)
        self._lat_prev_error = error_lat

        # Error integral with anti-windup clamping
        self._lat_integral += error_lat * self.dt
        max_i_lat = 0.20 / max(self.ki_steer, 1e-4)
        self._lat_integral = float(np.clip(self._lat_integral, -max_i_lat, max_i_lat))

        # Steering command (negative sign counters positive displacement)
        p_steer = self.kp_steer * error_lat
        i_steer = self.ki_steer * self._lat_integral
        d_steer = self.kd_steer * d_lat
        delta_raw = -(p_steer + i_steer + d_steer)

        # Physical steering angle saturation
        delta_target = float(
            np.clip(delta_raw, -self.max_steer_rad, self.max_steer_rad)
        )
        normalized_steer = delta_target / self.max_steer_rad

        # 5. Longitudinal Velocity PID Control Law
        target_v = self.target_velocity
        if reference is not None and reference.ndim == 1 and len(reference) >= 4:
            target_v = float(reference[3])

        if self.adapt_speed_on_curvature:
            turn_attenuation = max(0.60, math.cos(delta_target))
            effective_target_v = target_v * turn_attenuation
        else:
            effective_target_v = target_v

        v_error = effective_target_v - v

        # Speed error derivative
        if self._speed_prev_error is None:
            d_speed = 0.0
        else:
            d_speed = (v_error - self._speed_prev_error) / max(self.dt, 1e-4)
        self._speed_prev_error = v_error

        # Speed error integral with anti-windup clamping
        self._speed_integral += v_error * self.dt
        self._speed_integral = float(np.clip(self._speed_integral, -1.0, 1.0))

        # Feedforward term (~0.25 compensates for indoor rolling resistance at 0.55 m/s)
        feedforward = 0.25 if effective_target_v > 0.05 else 0.0
        p_speed = self.kp_speed * v_error
        i_speed = self.ki_speed * self._speed_integral
        d_speed_term = self.kd_speed * d_speed

        throttle_raw = p_speed + i_speed + d_speed_term + feedforward
        normalized_throttle = float(np.clip(throttle_raw, -1.0, 1.0))

        # 6. Store telemetry diagnostics
        self._last_diagnostics = {
            "cross_track_error": cte,
            "heading_error": heading_err,
            "error_lat": error_lat,
            "delta_steer_rad": delta_target,
            "target_velocity": effective_target_v,
            "speed_error": v_error,
            "p_steer": p_steer,
            "i_steer": i_steer,
            "d_steer": d_steer,
            "p_speed": p_speed,
            "i_speed": i_speed,
            "d_speed": d_speed_term,
            "closest_waypoint_idx": closest_idx,
            "normalized_steer": normalized_steer,
            "normalized_throttle": normalized_throttle,
        }

        return np.array([normalized_steer, normalized_throttle], dtype=np.float32)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns telemetry dictionary for logging and performance metrics."""
        return self._last_diagnostics.copy()

    def _extract_state(
        self, state: Union[np.ndarray, VehicleState]
    ) -> Tuple[float, float, float, float]:
        """Extracts (x, y, yaw, velocity) from array or VehicleState."""
        if isinstance(state, VehicleState):
            return state.x, state.y, state.yaw, state.velocity
        arr = np.asarray(state, dtype=np.float64)
        return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])

    def _resolve_waypoints(
        self, reference: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Resolves reference path waypoints from arguments or internal state."""
        if reference is not None and reference.ndim == 2 and reference.shape[1] >= 2:
            return reference[:, :2]
        if self.waypoints is not None:
            return self.waypoints
        return None


def run_pid_demo(
    track_type: str = "oval",
    model_type: str = "dynamic",
    target_velocity: float = 0.55,
    max_steps: int = 1500,
) -> None:
    """Executes a real-time autonomous simulation demonstrating PID control."""
    from simulation.arena_env import ArenaEnv

    print("\n" + "=" * 65)
    print("  1:28 RC CAR CYBER-PHYSICAL TESTBED: PID AUTONOMOUS TRACKING")
    print("=" * 65)
    print(f"Physics Engine  : {model_type.upper()}")
    print(f"Reference Track : {track_type.upper()}")
    print(f"Target Velocity : {target_velocity:.2f} m/s")
    print("Window Controls : Press 'ESC' or 'Q' to quit | 'C' to clear trail")
    print("=" * 65 + "\n")

    env = ArenaEnv(
        render_mode="human",
        model_type=model_type,
        track_type=track_type,
        max_episode_steps=max_steps,
    )

    controller = PIDController(
        target_velocity=target_velocity,
        waypoints=env.waypoints,
    )

    obs, _ = env.reset()
    last_log_time = time.monotonic()
    step_count = 0
    cte_history = []

    try:
        while True:
            # 1. Compute control actions via PID algorithm
            action = controller.compute_action(obs)

            # 2. Advance simulation environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            cte_history.append(abs(info["cross_track_error"]))

            # 3. Render visual display
            env.render()

            # 4. Periodic terminal logging (every 1 second)
            now = time.monotonic()
            if now - last_log_time >= 1.0:
                last_log_time = now
                diag = controller.get_diagnostics()
                mean_cte = float(np.mean(cte_history[-50:]))
                print(
                    f"Step {step_count:4d} | "
                    f"v={info['velocity']:+.2f}m/s | "
                    f"CTE={info['cross_track_error']:+.3f}m (avg: {mean_cte:.3f}m) | "
                    f"HeadErr={math.degrees(info['heading_error']):+5.1f}° | "
                    f"Steer={math.degrees(diag['delta_steer_rad']):+5.1f}° | "
                    f"WP={diag['closest_waypoint_idx']:2d}/{len(env.waypoints)}"
                )

            if terminated:
                if info.get("out_of_bounds", False):
                    print("\n[!] Collision detected with arena wall!")
                else:
                    print("\n[!] Cross-track error threshold exceeded!")
                obs, _ = env.reset()
                controller.reset()
                cte_history.clear()

            if truncated:
                print(f"\n[OK] Completed full {max_steps} steps benchmark run.")
                break

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        if cte_history:
            print("\n" + "-" * 65)
            print("PERFORMANCE BENCHMARK SUMMARY (PID):")
            print(f"  Total Steps Completed : {step_count}")
            print(f"  Mean Cross-Track Error: {np.mean(cte_history)*100:.2f} cm")
            print(f"  Max Cross-Track Error : {np.max(cte_history)*100:.2f} cm")
            print("-" * 65 + "\n")


def main() -> None:
    """CLI entry point for running the PID controller autonomous demo."""
    parser = argparse.ArgumentParser(
        description="Autonomous path tracking using decoupled lateral/longitudinal PID."
    )
    parser.add_argument(
        "--track",
        type=str,
        choices=["oval", "circle", "figure_eight"],
        default="oval",
        help="Reference track trajectory (default: oval).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["dynamic", "kinematic"],
        default="dynamic",
        help="Physics model engine: 'dynamic' or 'kinematic' (default: dynamic).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.55,
        help="Target forward velocity in m/s (default: 0.55).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum simulation steps to run (default: 2000).",
    )

    args = parser.parse_args()

    run_pid_demo(
        track_type=args.track,
        model_type=args.model,
        target_velocity=args.speed,
        max_steps=args.steps,
    )


if __name__ == "__main__":
    main()
