"""Reinforcement learning policy inference wrapper for Stable-Baselines3 models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from .base_controller import BaseController


class RLPolicyAgent(BaseController):
    """Wraps pre-trained Stable-Baselines3 policy for real-time inference.

    Attributes:
        model_path: Optional path to serialized SB3 policy (.zip).
        deterministic: Whether to evaluate actions deterministically.
        model: Loaded Stable-Baselines3 policy algorithm instance.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        deterministic: bool = True,
    ) -> None:
        """Initializes model wrapper and loads weights if path provided."""
        super().__init__(name="RLPolicyAgent")
        self.model_path = Path(model_path) if model_path else None
        self.deterministic = deterministic
        self.model: Optional[BaseAlgorithm] = None

    def load_model(
        self, model_path: Union[str, Path], algorithm_cls: Optional[type] = None
    ) -> None:
        """Loads serialized model weights from disk.

        Args:
            model_path: Path to .zip model artifact.
            algorithm_cls: Optional algorithm class (PPO, SAC, TD3).
        """
        pass

    def compute_action(
        self,
        state: np.ndarray,
        reference: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Runs forward policy inference given environment observation.

        Args:
            state: Observation vector matching environment observation_space.
            reference: Optional reference trajectory input if not encoded in state.

        Returns:
            Action vector [steering, throttle] in normalized range [-1.0, 1.0].
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Resets recurrent states (if using recurrent policies like RecurrentPPO)."""
        pass
