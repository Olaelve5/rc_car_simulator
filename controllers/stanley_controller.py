"""Stanley non-linear path tracking controller for autonomous 1:28 RC testbed.

Combines path heading alignment and non-linear cross-track error correction
evaluated at the vehicle's front axle center:
    delta = (psi_path - psi) - arctan2(k_path * e_fa, |v| + k_soft)
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

__all__ = ["StanleyController"]


class StanleyController(BaseController):
    """Stanley formulation for front-axle trajectory tracking.

    The Stanley algorithm combines path heading alignment and non-linear
    cross-track error correction evaluated at the front axle center:
        delta = (psi_path - psi) - arctan2(k_path * e_fa, |v| + k_soft)

    Attributes:
        k_path: Cross-track error gain (higher = more aggressive lateral correction).
        k_soft: Softening parameter (m/s) to prevent zero-velocity division singularity.
        wheelbase_m: Wheelbase of 1:28 vehicle chassis in meters (L).
        cog_to_front_m: Distance from CoG to front axle center (l_f).
        max_steer_rad: Maximum physical front-wheel steering angle in radians.
        target_velocity: Cruising forward speed demand (m/s).
        kp_speed: Proportional gain for longitudinal speed control.
        ki_speed: Integral gain for longitudinal speed control.
        waypoints: Optional pre-loaded reference path waypoints of shape (N, 2).
    """

    def __init__(
        self,
        k_path: float = 2.5,
        k_soft: float = 0.35,
        wheelbase_m: Optional[float] = None,
        cog_to_front_m: Optional[float] = None,
        max_steer_rad: Optional[float] = None,
        target_velocity: float = 0.60,
        kp_speed: float = 1.8,
        ki_speed: float = 0.20,
        waypoints: Optional[np.ndarray] = None,
        dt: float = 0.02,
        adapt_speed_on_curvature: bool = True,
    ) -> None:
        """Initializes Stanley geometry parameters, gains, and error integrators."""
        super().__init__(name="StanleyController")
        veh_cfg = DEFAULT_SETTINGS.vehicle
        self.k_path = float(k_path)
        self.k_soft = float(k_soft)
        self.wheelbase_m = (
            float(wheelbase_m) if wheelbase_m is not None else veh_cfg.wheelbase_m
        )
        self.cog_to_front_m = (
            float(cog_to_front_m)
            if cog_to_front_m is not None
            else veh_cfg.cog_to_front_m
        )
        self.max_steer_rad = (
            float(max_steer_rad)
            if max_steer_rad is not None
            else veh_cfg.max_steer_rad
        )
        self.target_velocity = float(target_velocity)
        self.kp_speed = float(kp_speed)
        self.ki_speed = float(ki_speed)
        self.dt = float(dt)
        self.adapt_speed_on_curvature = adapt_speed_on_curvature

        self.waypoints: Optional[np.ndarray] = None
        if waypoints is not None:
            self.set_waypoints(waypoints)

        self._speed_integral: float = 0.0
        self._last_diagnostics: Dict[str, Any] = {}

    def set_waypoints(self, waypoints: np.ndarray) -> None:
        """Sets reference path waypoints of shape (N, 2)."""
        if len(waypoints) < 2:
            raise ValueError("Waypoints array must contain at least 2 points.")
        self.waypoints = np.asarray(waypoints, dtype=np.float64)

    def reset(self) -> None:
        """Resets internal controller integrators and diagnostic cache."""
        self._speed_integral = 0.0
        self._last_diagnostics.clear()

    def compute_action(
        self,
        state: Union[np.ndarray, VehicleState],
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes Stanley steering angle and longitudinal throttle command.

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
                "StanleyController requires reference path waypoints. "
                "Pass waypoints in compute_action() or initialize with waypoints."
            )

        # 3. Project state to front axle center
        x_f = x + self.cog_to_front_m * math.cos(yaw)
        y_f = y + self.cog_to_front_m * math.sin(yaw)

        # 4. Compute tracking errors at front axle via TrackGenerator
        cte_f, heading_err, closest_idx = TrackGenerator.compute_tracking_errors(
            pts, x_f, y_f, yaw
        )

        # 5. Stanley steering control law
        # Lateral error damping term
        cte_correction = math.atan2(
            -self.k_path * cte_f,
            abs(v) + self.k_soft,
        )
        # Combined steering demand (heading alignment + cross-track damping)
        delta_raw = -heading_err + cte_correction

        # Normalize delta_raw into [-pi, pi]
        delta_raw = (delta_raw + math.pi) % (2.0 * math.pi) - math.pi

        # Apply physical steering limits
        delta_target = float(
            np.clip(delta_raw, -self.max_steer_rad, self.max_steer_rad)
        )
        normalized_steer = delta_target / self.max_steer_rad

        # 6. Longitudinal speed control
        target_v = self.target_velocity
        if reference is not None and reference.ndim == 1 and len(reference) >= 4:
            target_v = float(reference[3])

        if self.adapt_speed_on_curvature:
            turn_attenuation = max(0.60, math.cos(delta_target))
            effective_target_v = target_v * turn_attenuation
        else:
            effective_target_v = target_v

        v_error = effective_target_v - v
        self._speed_integral += v_error * self.dt
        self._speed_integral = float(np.clip(self._speed_integral, -1.0, 1.0))

        # Feedforward (~0.25 compensates for indoor rolling resistance at 0.6 m/s)
        feedforward = 0.25 if effective_target_v > 0.05 else 0.0
        throttle_raw = (
            self.kp_speed * v_error
            + self.ki_speed * self._speed_integral
            + feedforward
        )
        normalized_throttle = float(np.clip(throttle_raw, -1.0, 1.0))

        # 7. Store telemetry diagnostics
        self._last_diagnostics = {
            "front_x": x_f,
            "front_y": y_f,
            "cross_track_error": cte_f,
            "heading_error": heading_err,
            "delta_steer_rad": delta_target,
            "target_velocity": effective_target_v,
            "speed_error": v_error,
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


def run_stanley_demo(
    track_type: str = "oval",
    model_type: str = "dynamic",
    target_velocity: float = 0.60,
    k_path: float = 2.5,
    max_steps: int = 1500,
) -> None:
    """Executes a real-time autonomous simulation demonstrating Stanley control."""
    from simulation.arena_env import ArenaEnv

    print("\n" + "=" * 65)
    print("  1:28 RC CAR CYBER-PHYSICAL TESTBED: STANLEY AUTONOMOUS PILOT")
    print("=" * 65)
    print(f"Physics Engine  : {model_type.upper()}")
    print(f"Reference Track : {track_type.upper()}")
    print(f"Target Velocity : {target_velocity:.2f} m/s")
    print(f"Cross-Track Gain: k_path = {k_path:.2f}")
    print("Window Controls : Press 'ESC' or 'Q' to quit | 'C' to clear trail")
    print("=" * 65 + "\n")

    env = ArenaEnv(
        render_mode="human",
        model_type=model_type,
        track_type=track_type,
        max_episode_steps=max_steps,
    )

    controller = StanleyController(
        k_path=k_path,
        target_velocity=target_velocity,
        waypoints=env.waypoints,
    )

    obs, _ = env.reset()
    last_log_time = time.monotonic()
    step_count = 0
    cte_history = []

    try:
        while True:
            # 1. Compute control actions via Stanley algorithm
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
            print("PERFORMANCE BENCHMARK SUMMARY:")
            print(f"  Total Steps Completed : {step_count}")
            print(f"  Mean Cross-Track Error: {np.mean(cte_history)*100:.2f} cm")
            print(f"  Max Cross-Track Error : {np.max(cte_history)*100:.2f} cm")
            print("-" * 65 + "\n")


def main() -> None:
    """CLI entry point for running the Stanley controller autonomous demo."""
    parser = argparse.ArgumentParser(
        description="Autonomous path tracking using the non-linear Stanley controller."
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
        default=0.60,
        help="Target forward velocity in m/s (default: 0.60).",
    )
    parser.add_argument(
        "--k-path",
        type=float,
        default=2.5,
        help="Stanley cross-track gain (default: 2.5).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum simulation steps to run (default: 2000).",
    )

    args = parser.parse_args()

    run_stanley_demo(
        track_type=args.track,
        model_type=args.model,
        target_velocity=args.speed,
        k_path=args.k_path,
        max_steps=args.steps,
    )


if __name__ == "__main__":
    main()
