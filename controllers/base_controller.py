"""Abstract base class interface for testbed controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseController(ABC):
    """Abstract interface defining required methods for vehicle controllers."""

    def __init__(self, name: str = "BaseController") -> None:
        """Initializes base controller parameters."""
        self.name = name

    @abstractmethod
    def compute_action(
        self,
        state: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Computes control action demand based on current state and reference.

        Args:
            state: Current vehicle state vector [x, y, yaw, velocity].
            reference: Target reference waypoint or trajectory slice
                [ref_x, ref_y, ref_yaw, ref_v].

        Returns:
            Control action array [steering, throttle] in normalized range [-1.0, 1.0].
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets internal controller states (integrators, filters, history)."""
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns telemetry dictionary for logging and performance metrics."""
        return {}
