"""Centralized configuration schemas and default parameters for the testbed.

This module defines strongly-typed configuration dataclasses for vision,
networking, vehicle safety boundaries, and arena geometry.
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
class TestbedSettings:
    """Master configuration container for the testbed system."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)


# Global default configuration instance
DEFAULT_SETTINGS = TestbedSettings()
