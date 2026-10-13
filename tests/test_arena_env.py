"""Unit tests for Gymnasium ArenaEnv reinforcement learning environment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

# Ensure repository root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulation.arena_env import ArenaEnv  # noqa: E402
from simulation.dynamic_bicycle_model import DynamicBicycleModel  # noqa: E402
from simulation.bicycle_model import KinematicBicycleModel  # noqa: E402
from simulation.vehicle_state import VehicleState  # noqa: E402


def test_arena_env_initialization() -> None:
    """Verifies default spaces, dynamics, and track configuration."""
    env = ArenaEnv(model_type="dynamic", track_type="oval")

    # Space specifications
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (6,)
    assert np.allclose(env.action_space.low, [-1.0, -1.0])
    assert np.allclose(env.action_space.high, [1.0, 1.0])

    # Model and track
    assert isinstance(env.model, DynamicBicycleModel)
    assert env.waypoints.shape[0] > 10
    assert env.waypoints.shape[1] == 2

    env.close()


def test_arena_env_reset() -> None:
    """Verifies environment reset structure and deterministic options."""
    env = ArenaEnv(model_type="dynamic")

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    # Info verification
    assert "cross_track_error" in info
    assert "heading_error" in info
    assert "closest_waypoint_idx" in info
    assert info["step"] == 0
    assert info["model_type"] == "Dynamic"

    # Reset with specific waypoint
    obs2, info2 = env.reset(options={"waypoint_idx": 5})
    assert info2["closest_waypoint_idx"] == 5

    # Reset with explicit initial pose
    obs3, info3 = env.reset(options={"initial_pose": (1.0, 0.75, math.pi / 4)})
    assert math.isclose(obs3[0], 1.0, abs_tol=1e-4)
    assert math.isclose(obs3[1], 0.75, abs_tol=1e-4)

    # Reset with explicit VehicleState
    custom_state = VehicleState(x=1.2, y=0.8, yaw=0.1)
    obs4, _ = env.reset(options={"initial_state": custom_state})
    assert math.isclose(obs4[0], 1.2, abs_tol=1e-4)
    assert math.isclose(obs4[1], 0.8, abs_tol=1e-4)

    env.close()


def test_arena_env_step_execution() -> None:
    """Verifies forward stepping advances state, calculates rewards, and tracks."""
    env = ArenaEnv(model_type="dynamic")
    env.reset(seed=123)

    action = np.array([0.0, 0.5], dtype=np.float32)  # Go straight half throttle
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert info["step"] == 1
    assert obs[3] > 0.0  # Velocity increased

    env.close()


def test_arena_env_model_switching() -> None:
    """Verifies switching between kinematic and dynamic physics engines."""
    env = ArenaEnv(model_type="kinematic")
    assert isinstance(env.model, KinematicBicycleModel)

    env.reset()
    action = np.array([0.2, 0.4], dtype=np.float32)
    env.step(action)

    # Switch to dynamic model
    env.set_model("dynamic")
    assert isinstance(env.model, DynamicBicycleModel)

    # Step again with dynamic model
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["model_type"] == "Dynamic"

    env.close()


def test_arena_env_track_types() -> None:
    """Verifies track generation for circle, figure_eight, and oval."""
    for track_name in ["oval", "circle", "figure_eight"]:
        env = ArenaEnv(track_type=track_name)
        assert len(env.waypoints) > 20
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        env.close()


def test_arena_env_boundary_termination() -> None:
    """Verifies episode terminates when car crashes into arena walls."""
    env = ArenaEnv(
        boundary_clearance_m=0.10,
        collision_penalty=100.0,
    )
    # Place car at the wall boundary
    env.reset(options={"initial_pose": (0.05, 0.75, math.pi)})

    # Drive into the left wall
    action = np.array([0.0, 1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert info["out_of_bounds"] is True
    assert reward == -100.0

    env.close()


def test_arena_env_truncation() -> None:
    """Verifies episode truncates when reaching max_episode_steps."""
    max_steps = 5
    env = ArenaEnv(max_episode_steps=max_steps)
    env.reset()

    truncated = False
    for _ in range(max_steps):
        _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        if terminated:
            break

    assert truncated is True
    env.close()


def test_arena_env_rgb_array_render() -> None:
    """Verifies headless visualizer rendering returns valid (H, W, 3) frame."""
    env = ArenaEnv(render_mode="rgb_array")
    env.reset()

    frame = env.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8

    env.close()


def test_arena_env_custom_waypoints() -> None:
    """Verifies passing custom trajectory coordinates."""
    custom_pts = np.array(
        [
            [0.5, 0.5],
            [1.5, 0.5],
            [1.5, 1.0],
            [0.5, 1.0],
        ],
        dtype=np.float64,
    )
    env = ArenaEnv(waypoints=custom_pts)
    assert np.allclose(env.waypoints, custom_pts)

    obs, info = env.reset()
    assert info["closest_waypoint_idx"] == 0
    env.close()


def main() -> None:
    """Direct execution runner."""
    print("=" * 60)
    print("Running Gymnasium ArenaEnv Tests")
    print("=" * 60)
    test_arena_env_initialization()
    print("[OK] test_arena_env_initialization passed")
    test_arena_env_reset()
    print("[OK] test_arena_env_reset passed")
    test_arena_env_step_execution()
    print("[OK] test_arena_env_step_execution passed")
    test_arena_env_model_switching()
    print("[OK] test_arena_env_model_switching passed")
    test_arena_env_track_types()
    print("[OK] test_arena_env_track_types passed")
    test_arena_env_boundary_termination()
    print("[OK] test_arena_env_boundary_termination passed")
    test_arena_env_truncation()
    print("[OK] test_arena_env_truncation passed")
    test_arena_env_rgb_array_render()
    print("[OK] test_arena_env_rgb_array_render passed")
    test_arena_env_custom_waypoints()
    print("[OK] test_arena_env_custom_waypoints passed")
    print("=" * 60)
    print("ALL TESTS PASSED! ✓")


if __name__ == "__main__":
    main()
