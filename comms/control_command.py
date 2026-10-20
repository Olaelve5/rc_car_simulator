"""Hardware-independent drive command shared by controllers and transmitters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlCommand:
    """Normalized vehicle actuation command.

    Attributes:
        steering: Normalized steering in range [-1.0, 1.0] (left < 0 < right).
        throttle: Normalized throttle command in range [-1.0, 1.0] (+forward).
        brake: Emergency stop override; forces zero throttle.
        timestamp: Unix epoch timestamp of command creation.
    """

    steering: float = 0.0
    throttle: float = 0.0
    brake: bool = False
    timestamp: float = 0.0
