"""Centralized configuration schemas and default parameters for the testbed.

This module defines strongly-typed configuration dataclasses for vision,
networking, vehicle safety boundaries, arena geometry, and vehicle physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CameraConfig:
    """Configuration for overhead UVC camera capture on macOS."""

    camera_id: int = 0
    resolution: Tuple[int, int] = (1280, 720)  # (width, height) in pixels
    fps: int = 60
    fourcc: str = "MJPG"
    auto_exposure: bool = False
    exposure_value: float = -6.0
    flip_video: bool = True  # Rotate frame 180 deg for overhead camera orientation


@dataclass
class NetworkConfig:
    """Network configuration for UDP communication with onboard ESP32-C3."""

    esp32_ip: str = "192.168.1.100"
    udp_port: int = 8888
    local_port: int = 8889
    transmission_rate_hz: float = 50.0  # Control loop update rate
    socket_timeout_s: float = 0.1


@dataclass
class TransmitterConfig:
    """USB serial link to the ESP32-C3 + nRF24L01 RC transmitter."""

    port: str = ""  # e.g. /dev/cu.usbmodem1101; empty = pass on the command line
    baud: int = 115200  # Ignored on native USB CDC, kept for consistency
    send_rate_hz: float = 50.0  # Must be well above 1 / firmware failsafe (300 ms)
    max_throttle: float = 0.60  # Throttle cap for indoor testing (1.0 = full)
    invert_steering: bool = True  # Firmware +STEER turns the 284131 left


@dataclass
class SafetyConfig:
    """Hardware safety boundaries and constraints for 1:28 RC vehicle."""

    max_throttle_pct: float = 0.50  # Cap throttle at 50% for indoor testbed safety
    crawling_speed_limit_mps: float = 0.30  # Max crawling linear speed (m/s)
    emergency_stop_timeout_s: float = 0.25  # Watchdog cutoff if heartbeat is lost
    max_steering_angle_rad: float = 0.5236  # ~30 degrees in radians
    min_throttle_deadband: float = 0.05  # Lower threshold to overcome static friction


@dataclass
class ArenaConfig:
    """Physical arena dimensions and ArUco marker configuration."""

    arena_width_m: float = 2.0  # Physical width in meters
    arena_height_m: float = 1.5  # Physical height in meters
    aruco_dict_name: str = "DICT_4X4_50"
    vehicle_marker_id: int = 4
    vehicle_marker_size_m: float = 0.05
    corner_marker_ids: Tuple[int, int, int, int] = (0, 1, 2, 3)


@dataclass
class VehicleDynamicsConfig:
    """Physical, kinematic, and tire dynamics parameters for 1:28 scale vehicle.

    Calibrated for a standard ~98mm wheelbase RC car (e.g., Kyosho Mini-Z,
    WLToys 284131) running on a flat indoor surface.
    """

    # --- Geometric Parameters ---
    wheelbase_m: float = 0.098  # Distance between front and rear axle centers (L)
    cog_to_front_m: float = 0.049  # Distance from CoG to front axle (l_f)
    cog_to_rear_m: float = 0.049  # Distance from CoG to rear axle (l_r)

    # --- Steering Actuation Limits ---
    max_steer_rad: float = 0.5236  # Maximum physical front wheel angle (~30 deg)
    max_steer_rate_radps: float = 5.0  # Servo rate limit (rad/s)

    # --- Drive & Velocity Limits ---
    max_velocity_mps: float = 1.00  # Forward speed limit (m/s)
    max_reverse_velocity_mps: float = 0.50  # Reverse speed limit (m/s)
    max_accel_mps2: float = 2.00  # Maximum linear acceleration (m/s^2)
    max_decel_mps2: float = 4.50  # Maximum active braking deceleration (m/s^2)

    # --- Friction & Resistive Forces ---
    rolling_resistance_coeff: float = 0.55  # Rolling resistance (m/s^2)
    drag_coeff: float = 0.30  # Viscous drivetrain drag coefficient (1/s)
    min_throttle_deadband: float = 0.10  # Throttle threshold to overcome stiction

    # --- Dynamic Model & Tire Grip Parameters ---
    mass_kg: float = 0.180  # Vehicle mass (~180g for 1:28 chassis)
    iz_kgm2: float = 0.00035  # Yaw moment of inertia (kg*m^2)
    tire_friction_mu: float = 0.65  # Tire-floor friction coefficient (grip)
    cornering_stiffness_front: float = 8.0  # Front cornering stiffness (N/rad)
    cornering_stiffness_rear: float = 6.5  # Rear cornering stiffness (N/rad)
    kinematic_blend_speed_mps: float = 0.15  # Low-speed kinematic blend threshold (m/s)

    # --- Timing ---
    dt: float = 0.02  # Default simulation integration step (50 Hz)


# Type alias for backward compatibility
VehicleDynamicsParams = VehicleDynamicsConfig


@dataclass
class TestbedSettings:
    """Master configuration container for the testbed system."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    transmitter: TransmitterConfig = field(default_factory=TransmitterConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    vehicle: VehicleDynamicsConfig = field(default_factory=VehicleDynamicsConfig)


# Global default configuration instance
DEFAULT_SETTINGS = TestbedSettings()
