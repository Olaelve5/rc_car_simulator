"""Gymnasium interface for cyber-physical RC car simulation and sim-to-real transfer."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config.settings import (
    DEFAULT_SETTINGS,
    ArenaConfig,
    VehicleDynamicsConfig,
    VehicleDynamicsParams,
)
from .arena_visualizer import ArenaVisualizer
from .bicycle_model import KinematicBicycleModel, create_vehicle_model
from .dynamic_bicycle_model import DynamicBicycleModel
from .track_generator import TrackGenerator
from .vehicle_state import VehicleState

__all__ = ["ArenaEnv", "VehicleDynamicsConfig", "VehicleDynamicsParams"]


class ArenaEnv(gym.Env):
    """Gymnasium environment representing the 2D planar testbed arena.

    Observation Space:
        Box(6): [x, y, yaw, velocity, cross_track_error, heading_error]
        - x: Vehicle X position in arena frame (meters)
        - y: Vehicle Y position in arena frame (meters)
        - yaw: Vehicle heading orientation [-pi, pi] (radians)
        - velocity: Vehicle longitudinal linear velocity (m/s)
        - cross_track_error: Signed lateral deviation from reference path (meters)
        - heading_error: Signed angular deviation from path tangent [-pi, pi] (radians)

    Action Space:
        Box(2): [steering_command, throttle_command] normalized to [-1.0, 1.0]
        - action[0]: Normalized steering demand mapped to physical steering angle
        - action[1]: Normalized throttle/brake demand mapped to [-1.0, 1.0]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dynamics_params: Optional[VehicleDynamicsConfig] = None,
        arena_config: Optional[ArenaConfig] = None,
        model_type: str = "dynamic",
        track_type: str = "oval",
        waypoints: Optional[np.ndarray] = None,
        max_episode_steps: int = 500,
        max_cross_track_error: float = 0.8,
        boundary_clearance_m: float = 0.08,
        collision_penalty: float = 50.0,
        w_progress: float = 1.0,
        w_cte: float = 2.0,
        w_heading: float = 0.5,
        w_action: float = 0.05,
    ) -> None:
        """Initializes observation/action spaces, vehicle model, and track."""
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{render_mode}'. "
                f"Supported modes: {self.metadata['render_modes']}"
            )
        self.render_mode = render_mode

        self.dynamics_params = dynamics_params or DEFAULT_SETTINGS.vehicle
        self.arena_config = arena_config or DEFAULT_SETTINGS.arena
        self.track_type = track_type
        self.max_episode_steps = max_episode_steps
        self.max_cross_track_error = max_cross_track_error
        self.boundary_clearance_m = boundary_clearance_m
        self.collision_penalty = collision_penalty

        # Reward formulation weights
        self.w_progress = w_progress
        self.w_cte = w_cte
        self.w_heading = w_heading
        self.w_action = w_action

        # Reference trajectory path waypoints (N, 2)
        if waypoints is not None:
            self.waypoints = np.asarray(waypoints, dtype=np.float64)
        else:
            self.waypoints = self._generate_default_waypoints(
                track_type=self.track_type,
                arena=self.arena_config,
            )

        # Vehicle physics simulation model (Dynamic or Kinematic)
        self.model: Union[KinematicBicycleModel, DynamicBicycleModel] = (
            create_vehicle_model(
                model_type=model_type,
                params=self.dynamics_params,
            )
        )

        # Action: [steering (-1.0 to 1.0), throttle (-1.0 to 1.0)]
        self.action_space: spaces.Box = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [x, y, yaw, velocity, cte, heading_error]
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([-5.0, -5.0, -np.pi, -5.0, -5.0, -np.pi], dtype=np.float32),
            high=np.array([5.0, 5.0, np.pi, 5.0, 5.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

        self._step_count: int = 0
        self._visualizer: Optional[ArenaVisualizer] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional configuration dictionary. Can contain:
                - 'initial_state': VehicleState instance
                - 'initial_pose': Tuple of (x, y, yaw)
                - 'waypoint_idx': int index along reference path to start at
                - 'add_noise': bool, whether to add small random perturbations

        Returns:
            Tuple of (initial_observation, info_dictionary).
        """
        super().reset(seed=seed)
        self._step_count = 0
        opts = options or {}

        # 1. Resolve initial vehicle state
        if "initial_state" in opts:
            init_state = opts["initial_state"].copy()
        elif "initial_pose" in opts:
            ix, iy, iyaw = opts["initial_pose"]
            init_state = VehicleState(x=ix, y=iy, yaw=iyaw)
        else:
            wp_idx = int(opts.get("waypoint_idx", 0)) % len(self.waypoints)
            next_idx = (wp_idx + 1) % len(self.waypoints)
            x0 = float(self.waypoints[wp_idx, 0])
            y0 = float(self.waypoints[wp_idx, 1])
            dx = float(self.waypoints[next_idx, 0] - x0)
            dy = float(self.waypoints[next_idx, 1] - y0)
            yaw0 = math.atan2(dy, dx)

            if opts.get("add_noise", False):
                x0 += float(self.np_random.uniform(-0.03, 0.03))
                y0 += float(self.np_random.uniform(-0.03, 0.03))
                yaw0 += float(self.np_random.uniform(-0.08, 0.08))

            init_state = VehicleState(x=x0, y=y0, yaw=yaw0)

        # 2. Reset vehicle model and visualizer trail
        self.model.reset(init_state)
        if self._visualizer is not None:
            self._visualizer.clear_trail()

        # 3. Compute initial tracking errors
        state = self.model.get_state()
        cte, heading_err, closest_idx = TrackGenerator.compute_tracking_errors(
            self.waypoints, state.x, state.y, state.yaw
        )

        obs = self._get_obs(state, cte, heading_err)
        info = self._get_info(
            state=state,
            cte=cte,
            heading_err=heading_err,
            closest_idx=closest_idx,
        )

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies actuation action and advances environment by one simulation step.

        Args:
            action: Array containing [steering, throttle] normalized in [-1.0, 1.0].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # 1. Action validation and scaling to physical demands
        act = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        steer_target = float(act[0]) * self.dynamics_params.max_steer_rad
        throttle_input = float(act[1])

        # 2. Advance vehicle physics
        state = self.model.step(
            steering_target=steer_target,
            throttle_input=throttle_input,
        )
        self._step_count += 1

        # 3. Track deviation analysis
        cte, heading_err, closest_idx = TrackGenerator.compute_tracking_errors(
            self.waypoints, state.x, state.y, state.yaw
        )

        # 4. Termination checks (wall contact or catastrophic off-track)
        out_of_bounds = self._check_boundary_collision(state)
        cte_exceeded = abs(cte) > self.max_cross_track_error
        terminated = bool(out_of_bounds or cte_exceeded)

        # 5. Truncation check (max episode duration reached)
        truncated = bool(self._step_count >= self.max_episode_steps)

        # 6. Reward computation
        reward = self._compute_reward(
            state=state,
            cte=cte,
            heading_err=heading_err,
            action=act,
            out_of_bounds=out_of_bounds,
            cte_exceeded=cte_exceeded,
        )

        obs = self._get_obs(state, cte, heading_err)
        info = self._get_info(
            state=state,
            cte=cte,
            heading_err=heading_err,
            closest_idx=closest_idx,
            out_of_bounds=out_of_bounds,
            cte_exceeded=cte_exceeded,
        )

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment visually according to configured render_mode.

        Returns:
            RGB array of shape (H, W, 3) if render_mode is 'rgb_array', otherwise None.
        """
        if self.render_mode is None:
            return None

        if self._visualizer is None:
            self._visualizer = ArenaVisualizer(
                arena_config=self.arena_config,
                render_fps=self.metadata["render_fps"],
                render_mode=self.render_mode,
                window_title=f"ArenaEnv ({self.model.model_name})",
                waypoints=self.waypoints,
            )

        state = self.model.get_state()
        cte, _, _ = TrackGenerator.compute_tracking_errors(
            self.waypoints, state.x, state.y, state.yaw
        )
        status = (
            f"[{self.model.model_name.upper()}] Step: {self._step_count} | "
            f"v: {state.velocity:+.2f}m/s | CTE: {cte:+.3f}m"
        )

        return self._visualizer.render(
            vehicle_pose=(state.x, state.y, state.yaw),
            steering_angle=state.steering_angle,
            status_text=status,
        )

    def close(self) -> None:
        """Cleans up any allocated simulation or rendering resources."""
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None

    def set_model(self, model_type: str) -> None:
        """Switches between dynamic and kinematic physics models on the fly."""
        curr_state = self.model.get_state()
        self.model = create_vehicle_model(
            model_type=model_type,
            params=self.dynamics_params,
            initial_state=curr_state,
        )

    def set_waypoints(self, waypoints: np.ndarray) -> None:
        """Updates reference path waypoints and notifies visualizer if active."""
        if len(waypoints) < 2:
            raise ValueError("Waypoints array must contain at least 2 points.")
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        if self._visualizer is not None:
            self._visualizer.set_waypoints(self.waypoints)

    def _generate_default_waypoints(
        self,
        track_type: str,
        arena: ArenaConfig,
    ) -> np.ndarray:
        """Generates reference path waypoints via TrackGenerator."""
        cx = arena.arena_width_m / 2.0
        cy = arena.arena_height_m / 2.0
        kind = track_type.strip().lower()

        if kind == "circle":
            radius = min(arena.arena_width_m, arena.arena_height_m) * 0.35
            return TrackGenerator.generate_circle(radius_m=radius, center=(cx, cy))
        elif kind == "figure_eight":
            return TrackGenerator.generate_figure_eight(
                length_m=arena.arena_width_m * 0.70,
                width_m=arena.arena_height_m * 0.60,
                center=(cx, cy),
            )
        elif kind == "oval":
            return TrackGenerator.generate_oval(
                length_m=arena.arena_width_m * 0.70,
                width_m=arena.arena_height_m * 0.60,
                center=(cx, cy),
            )
        else:
            raise ValueError(
                f"Unknown track_type '{track_type}'. "
                "Expected 'oval', 'circle', or 'figure_eight'."
            )

    def _check_boundary_collision(self, state: VehicleState) -> bool:
        """Checks if vehicle position exceeds arena wall clearances."""
        b = self.boundary_clearance_m
        return bool(
            state.x < b
            or state.x > (self.arena_config.arena_width_m - b)
            or state.y < b
            or state.y > (self.arena_config.arena_height_m - b)
        )

    def _compute_reward(
        self,
        state: VehicleState,
        cte: float,
        heading_err: float,
        action: np.ndarray,
        out_of_bounds: bool,
        cte_exceeded: bool,
    ) -> float:
        """Computes scalar reward balancing path progress, tracking, and smoothness."""
        if out_of_bounds:
            return -self.collision_penalty
        if cte_exceeded:
            return -self.collision_penalty * 0.5

        # Forward progress projected onto path tangent
        r_progress = self.w_progress * state.velocity * math.cos(heading_err)

        # Penalties for lateral and heading deviations
        r_cte = -self.w_cte * (cte**2)
        r_heading = -self.w_heading * (heading_err**2)

        # Control effort penalty to encourage smooth commands
        r_action = -self.w_action * float(np.sum(action**2))

        return float(r_progress + r_cte + r_heading + r_action)

    def _get_obs(
        self, state: VehicleState, cte: float, heading_error: float
    ) -> np.ndarray:
        """Constructs 6D observation vector matching observation_space."""
        return np.array(
            [
                state.x,
                state.y,
                state.yaw,
                state.velocity,
                cte,
                heading_error,
            ],
            dtype=np.float32,
        )

    def _get_info(
        self,
        state: VehicleState,
        cte: float,
        heading_err: float,
        closest_idx: int,
        out_of_bounds: bool = False,
        cte_exceeded: bool = False,
    ) -> Dict[str, Any]:
        """Assembles telemetry and diagnostic dictionary."""
        return {
            "step": self._step_count,
            "closest_waypoint_idx": closest_idx,
            "cross_track_error": cte,
            "heading_error": heading_err,
            "velocity": state.velocity,
            "steering_angle": state.steering_angle,
            "yaw": state.yaw,
            "drift_angle_rad": state.drift_angle_rad,
            "out_of_bounds": out_of_bounds,
            "cte_exceeded": cte_exceeded,
            "model_type": self.model.model_name,
        }
