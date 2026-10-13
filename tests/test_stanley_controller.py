"""Unit tests for the Stanley non-linear path tracking controller."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

# Ensure repository root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from controllers.stanley_controller import StanleyController  # noqa: E402
from simulation.arena_env import ArenaEnv  # noqa: E402
from simulation.track_generator import TrackGenerator  # noqa: E402
from simulation.vehicle_state import VehicleState  # noqa: E402


def test_stanley_initialization() -> None:
    """Verifies default parameters and geometry configuration."""
    controller = StanleyController(k_path=2.5, target_velocity=0.60)
    assert controller.k_path == 2.5
    assert controller.target_velocity == 0.60
    assert controller.wheelbase_m > 0.05
    assert controller.cog_to_front_m > 0.02


def test_stanley_lateral_correction_direction() -> None:
    """Verifies steering direction counters lateral displacement (left vs right)."""
    # Straight path along +X (y = 0.75)
    waypoints = np.array(
        [
            [0.0, 0.75],
            [1.0, 0.75],
            [2.0, 0.75],
        ],
        dtype=np.float64,
    )
    controller = StanleyController(
        k_path=2.0,
        waypoints=waypoints,
        target_velocity=0.50,
    )

    # 1. Car on centerline facing +X: steering should be near zero
    state_center = np.array([0.5, 0.75, 0.0, 0.5], dtype=np.float32)
    act_center = controller.compute_action(state_center)
    assert abs(act_center[0]) < 1e-3, f"Expected 0 steer, got {act_center[0]}"

    # 2. Car is to the LEFT of path (+Y offset, y = 0.85):
    # Stanley must steer RIGHT (negative steer angle)
    state_left = np.array([0.5, 0.85, 0.0, 0.5], dtype=np.float32)
    act_left = controller.compute_action(state_left)
    assert (
        act_left[0] < -0.05
    ), f"Expected negative steering to turn right, got {act_left[0]}"

    # 3. Car is to the RIGHT of path (-Y offset, y = 0.65):
    # Stanley must steer LEFT (positive steer angle)
    state_right = np.array([0.5, 0.65, 0.0, 0.5], dtype=np.float32)
    act_right = controller.compute_action(state_right)
    assert (
        act_right[0] > 0.05
    ), f"Expected positive steering to turn left, got {act_right[0]}"


def test_stanley_heading_alignment() -> None:
    """Verifies steering aligns vehicle heading with path tangent."""
    waypoints = np.array([[0.0, 0.75], [1.0, 0.75], [2.0, 0.75]])
    controller = StanleyController(waypoints=waypoints, target_velocity=0.50)

    # Car on path but yaw pointed +20 deg left: must steer right (negative)
    state_yaw_left = np.array(
        [0.5, 0.75, math.radians(20.0), 0.5], dtype=np.float32
    )
    act_yaw_left = controller.compute_action(state_yaw_left)
    assert act_yaw_left[0] < -0.10

    # Car on path but yaw pointed -20 deg right: must steer left (positive)
    state_yaw_right = np.array(
        [0.5, 0.75, math.radians(-20.0), 0.5], dtype=np.float32
    )
    act_yaw_right = controller.compute_action(state_yaw_right)
    assert act_yaw_right[0] > 0.10


def test_stanley_speed_control() -> None:
    """Verifies longitudinal speed control demands appropriate throttle."""
    waypoints = TrackGenerator.generate_oval()
    controller = StanleyController(
        target_velocity=0.60,
        waypoints=waypoints,
        kp_speed=2.0,
    )

    # 1. Car stopped: should request positive forward throttle
    act_stopped = controller.compute_action(np.array([1.0, 0.75, 0.0, 0.0]))
    assert act_stopped[1] > 0.30

    # 2. Car moving faster than target (0.90 m/s > 0.60 m/s):
    # throttle demand should be low or negative (braking)
    act_overspeed = controller.compute_action(np.array([1.0, 0.75, 0.0, 0.90]))
    assert act_overspeed[1] < act_stopped[1]


def test_stanley_vehicle_state_input() -> None:
    """Verifies compute_action accepts VehicleState objects directly."""
    waypoints = TrackGenerator.generate_oval()
    controller = StanleyController(waypoints=waypoints)

    state = VehicleState(x=1.0, y=0.5, yaw=0.0, velocity=0.4)
    action = controller.compute_action(state)

    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert -1.0 <= action[0] <= 1.0
    assert -1.0 <= action[1] <= 1.0


def test_stanley_closed_loop_tracking() -> None:
    """Verifies closed-loop path tracking in ArenaEnv maintains low CTE."""
    env = ArenaEnv(
        model_type="dynamic",
        track_type="oval",
        max_episode_steps=200,
    )
    controller = StanleyController(
        k_path=2.5,
        target_velocity=0.55,
        waypoints=env.waypoints,
    )

    obs, _ = env.reset(seed=42)
    cte_records = []

    for _ in range(120):
        action = controller.compute_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cte_records.append(abs(info["cross_track_error"]))

        assert not terminated, "Vehicle crashed during closed-loop Stanley tracking!"
        if truncated:
            break

    env.close()

    mean_cte = float(np.mean(cte_records))
    max_cte = float(np.max(cte_records))

    # Average tracking error on 1:28 oval should be well within 8cm
    assert mean_cte < 0.08, f"Mean CTE too high: {mean_cte:.4f}m"
    assert max_cte < 0.20, f"Max CTE too high: {max_cte:.4f}m"


def main() -> None:
    """Direct test runner."""
    print("=" * 60)
    print("Running Stanley Controller Test Suite")
    print("=" * 60)
    test_stanley_initialization()
    print("[OK] test_stanley_initialization passed")
    test_stanley_lateral_correction_direction()
    print("[OK] test_stanley_lateral_correction_direction passed")
    test_stanley_heading_alignment()
    print("[OK] test_stanley_heading_alignment passed")
    test_stanley_speed_control()
    print("[OK] test_stanley_speed_control passed")
    test_stanley_vehicle_state_input()
    print("[OK] test_stanley_vehicle_state_input passed")
    test_stanley_closed_loop_tracking()
    print("[OK] test_stanley_closed_loop_tracking passed")
    print("=" * 60)
    print("ALL STANLEY CONTROLLER TESTS PASSED! ✓")


if __name__ == "__main__":
    main()
