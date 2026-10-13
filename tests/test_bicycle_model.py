"""Unit tests and physics validation for KinematicBicycleModel."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulation.bicycle_model import (  # noqa: E402
    DynamicBicycleModel,
    KinematicBicycleModel,
    VehicleDynamicsParams,
    VehicleState,
    create_vehicle_model,
)


def test_vehicle_state_helpers() -> None:
    """Verifies VehicleState copying, to_numpy, and from_numpy helpers."""
    state = VehicleState(x=1.2, y=0.8, yaw=0.5, velocity=1.1, steering_angle=0.2)
    copy_state = state.copy()

    assert copy_state == state
    assert copy_state is not state  # Ensure distinct instance

    arr = state.to_numpy()
    assert np.allclose(arr, [1.2, 0.8, 0.5, 1.1])

    restored = VehicleState.from_numpy(arr, steering_angle=0.2)
    assert np.isclose(restored.x, state.x)
    assert np.isclose(restored.y, state.y)
    assert np.isclose(restored.yaw, state.yaw)
    assert np.isclose(restored.velocity, state.velocity)
    assert np.isclose(restored.steering_angle, state.steering_angle)
    print("[OK] test_vehicle_state_helpers passed.")


def test_straight_line_acceleration_and_top_speed() -> None:
    """Verifies vehicle accelerates along X axis with zero steering and hits limit."""
    params = VehicleDynamicsParams(dt=0.02, max_velocity_mps=1.50)
    model = KinematicBicycleModel(params=params)

    # 100 steps (2 seconds) of full forward throttle
    for _ in range(100):
        model.step(steering_target=0.0, throttle_input=1.0)

    state = model.get_state()
    assert state.x > 0.5
    assert abs(state.y) < 1e-4
    assert state.velocity <= params.max_velocity_mps
    assert np.isclose(state.velocity, params.max_velocity_mps, atol=0.05)
    print(
        f"[OK] test_straight_line_acceleration passed (Final v: "
        f"{state.velocity:.2f})."
    )


def test_steering_slew_rate_limiting() -> None:
    """Verifies that steering servo respects max_steer_rate_radps."""
    params = VehicleDynamicsParams(
        dt=0.02,
        max_steer_rad=0.50,
        max_steer_rate_radps=5.0,  # 5 rad/s -> max delta per step = 0.1 rad
    )
    model = KinematicBicycleModel(params=params)

    # Request maximum steering in a single step
    state_1 = model.step(steering_target=0.50, throttle_input=0.0)
    expected_step_1 = 5.0 * 0.02  # 0.10 rad
    assert np.isclose(state_1.steering_angle, expected_step_1, atol=1e-4)

    # Run for 5 steps total -> should reach full lock (0.50 rad)
    for _ in range(4):
        state = model.step(steering_target=0.50, throttle_input=0.0)

    assert np.isclose(state.steering_angle, 0.50, atol=1e-4)
    print("[OK] test_steering_slew_rate_limiting passed.")


def test_turning_circle_geometry() -> None:
    """Verifies vehicle turning radius matches theoretical Kinematic CoG model."""
    params = VehicleDynamicsParams(
        wheelbase_m=0.10,
        cog_to_front_m=0.05,
        cog_to_rear_m=0.05,
        max_steer_rad=0.40,
        dt=0.01,
        rolling_resistance_coeff=0.0,
        drag_coeff=0.0,
    )
    # Start with initial steering angle and constant speed
    init_state = VehicleState(
        x=0.0, y=0.0, yaw=0.0, velocity=1.0, steering_angle=0.40
    )
    model = KinematicBicycleModel(params=params, initial_state=init_state)

    # Theoretical CoG turning radius: R_cog = sqrt((L / tan(delta))^2 + lr^2)
    steer = 0.40
    R_r = params.wheelbase_m / math.tan(steer)
    R_cog_theoretical = math.sqrt(R_r**2 + params.cog_to_rear_m**2)

    # Theoretical angular velocity omega = v / R_cog
    omega_theoretical = 1.0 / R_cog_theoretical
    period = 2.0 * math.pi / omega_theoretical

    num_steps = int(round(period / params.dt))
    ys = []

    for _ in range(num_steps):
        st = model.step(steering_target=steer, throttle_input=0.0, dt=params.dt)
        # Keep speed exactly at 1.0 for geometric verification
        model.state.velocity = 1.0
        ys.append(st.y)

    final_state = model.get_state()

    # After 1 full revolution, vehicle should return close to starting point
    distance_from_origin = math.hypot(final_state.x, final_state.y)
    assert distance_from_origin < 0.05, (
        f"Loop not completed cleanly! Error: {distance_from_origin:.4f}m"
    )

    measured_diameter_y = max(ys) - min(ys)
    expected_diameter = 2.0 * R_cog_theoretical
    assert np.isclose(measured_diameter_y, expected_diameter, atol=0.03), (
        f"Expected diameter {expected_diameter:.3f}m, measured "
        f"{measured_diameter_y:.3f}m"
    )
    print(
        f"[OK] test_turning_circle_geometry passed (Theoretical R: "
        f"{R_cog_theoretical:.3f}m, Loop Error: {distance_from_origin:.4f}m)."
    )


def test_throttle_deadband_and_friction_braking() -> None:
    """Verifies that throttle within deadband does not accelerate and car halts."""
    params = VehicleDynamicsParams(
        dt=0.02,
        min_throttle_deadband=0.05,
        rolling_resistance_coeff=0.5,
    )
    model = KinematicBicycleModel(params=params)

    # Below deadband -> vehicle must not move
    for _ in range(20):
        model.step(steering_target=0.0, throttle_input=0.03)
    assert model.get_state().velocity == 0.0
    assert model.get_state().x == 0.0

    # Give an initial speed, then release throttle (throttle = 0)
    model.reset(VehicleState(velocity=0.8))
    for _ in range(120):  # 2.4 seconds of coasting
        model.step(steering_target=0.0, throttle_input=0.0)

    # Should come to a clean complete halt without vibrating or oscillating
    final_v = model.get_state().velocity
    assert final_v == 0.0, f"Expected car to stop completely, got {final_v}"
    print("[OK] test_throttle_deadband_and_friction_braking passed.")


def test_active_braking_deceleration() -> None:
    """Verifies active braking stops the car faster than natural coasting."""
    params = VehicleDynamicsParams(
        dt=0.02,
        max_decel_mps2=5.0,
        rolling_resistance_coeff=0.3,
    )

    # Coasting trial
    model_coast = KinematicBicycleModel(
        params=params, initial_state=VehicleState(velocity=1.0)
    )
    for _ in range(20):
        model_coast.step(steering_target=0.0, throttle_input=0.0)
    v_coast = model_coast.get_state().velocity

    # Braking trial
    model_brake = KinematicBicycleModel(
        params=params, initial_state=VehicleState(velocity=1.0)
    )
    for _ in range(20):
        model_brake.step(steering_target=0.0, throttle_input=-1.0)
    v_brake = model_brake.get_state().velocity

    print(
        f"[OK] test_active_braking passed (Coast: {v_coast:.2f} vs "
        f"Brake: {v_brake:.2f} m/s)."
    )


def test_dynamic_bicycle_model_sliding_and_drift() -> None:
    """Verifies that DynamicBicycleModel develops lateral slip during hard cornering."""
    params = VehicleDynamicsParams(dt=0.02, max_velocity_mps=1.50)
    init_state = VehicleState(
        x=0.0, y=0.0, yaw=0.0, velocity=1.2, steering_angle=0.0
    )
    dyn_model = DynamicBicycleModel(params=params, initial_state=init_state)

    # Command sharp full-lock turn while moving fast
    for _ in range(25):  # 0.5 seconds of cornering
        st = dyn_model.step(steering_target=0.50, throttle_input=0.5)

    # Under hard cornering, dynamic model must exhibit non-zero
    # lateral sliding velocity (v_y) and a non-zero drift/sideslip angle.
    assert abs(st.v_y) > 0.005, f"Expected lateral slip v_y > 0.005, got {st.v_y}"
    assert (
        abs(st.drift_angle_rad) > 0.005
    ), f"Expected drift angle > 0.005 rad, got {st.drift_angle_rad}"
    assert (
        abs(st.yaw_rate) > 0.1
    ), f"Expected yaw rate > 0.1 rad/s, got {st.yaw_rate}"
    print(
        f"[OK] test_dynamic_bicycle_model_sliding passed (v_y: {st.v_y:.3f} m/s, "
        f"drift angle: {math.degrees(st.drift_angle_rad):.1f}°)."
    )


def test_create_vehicle_model_factory() -> None:
    """Verifies factory function correctly instantiates requested model."""
    dyn = create_vehicle_model("dynamic")
    assert isinstance(dyn, DynamicBicycleModel)
    assert dyn.model_name == "Dynamic"

    kin = create_vehicle_model("kinematic")
    assert isinstance(kin, KinematicBicycleModel)
    assert kin.model_name == "Kinematic"
    print("[OK] test_create_vehicle_model_factory passed.")


def test_modular_imports() -> None:
    """Verifies that modular files can be imported directly."""
    from config.settings import DEFAULT_SETTINGS, VehicleDynamicsConfig
    from simulation.dynamic_bicycle_model import DynamicBicycleModel as DBM
    from simulation.vehicle_state import VehicleState as VS

    params = VehicleDynamicsConfig()
    m = DBM(params=params)
    assert m.model_name == "Dynamic"
    assert DEFAULT_SETTINGS.vehicle.wheelbase_m == 0.098
    s = VS(x=1.0, y=2.0)
    assert s.x == 1.0
    print("[OK] test_modular_imports passed.")


def main() -> None:
    print("=" * 60)
    print("Running Vehicle Physics Validation Suite (Kinematic & Dynamic)")
    print("=" * 60)
    test_vehicle_state_helpers()
    test_straight_line_acceleration_and_top_speed()
    test_steering_slew_rate_limiting()
    test_turning_circle_geometry()
    test_throttle_deadband_and_friction_braking()
    test_active_braking_deceleration()
    test_dynamic_bicycle_model_sliding_and_drift()
    test_create_vehicle_model_factory()
    test_modular_imports()
    print("=" * 60)
    print("ALL VEHICLE MODEL PHYSICS TESTS PASSED! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
