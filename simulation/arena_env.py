"""Gymnasium interface for cyber-physical RC car simulation and sim-to-real transfer."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .bicycle_model import KinematicBicycleModel, VehicleDynamicsParams


class ArenaEnv(gym.Env):
    """Gymnasium environment representing the 2D planar testbed arena.

    Observation Space:
        Box(6): [x, y, yaw, velocity, cross_track_error, heading_error]

    Action Space:
        Box(2): [steering_command, throttle_command] normalized to [-1.0, 1.0]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        dynamics_params: Optional[VehicleDynamicsParams] = None,
    ) -> None:
        """Initializes observation/action spaces and simulation handles."""
        super().__init__()
        self.render_mode = render_mode
        self.dynamics_params = dynamics_params or VehicleDynamicsParams()
        self.model = KinematicBicycleModel(params=self.dynamics_params)

        # Action: [steering (-1.0 to 1.0), throttle (-1.0 to 1.0)]
        self.action_space: spaces.Box = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: [x, y, yaw, velocity, cte, heading_error]
        self.observation_space: spaces.Box = spaces.Box(
            low=np.array([-5.0, -5.0, -np.pi, 0.0, -5.0, -np.pi], dtype=np.float32),
            high=np.array([5.0, 5.0, np.pi, 5.0, 5.0, np.pi], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state.

        Args:
            seed: Optional random seed for reproducibility.
            options: Optional configuration dictionary.

        Returns:
            Tuple of (initial_observation, info_dictionary).
        """
        raise NotImplementedError

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies actuation action and advances environment by one simulation step.

        Args:
            action: Array containing [steering, throttle].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        raise NotImplementedError

    def render(self) -> Optional[np.ndarray]:
        """Renders the environment visually according to configured render_mode.

        Returns:
            RGB array if render_mode is 'rgb_array', otherwise None.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Cleans up any allocated simulation or rendering resources."""
        pass
