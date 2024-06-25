"""The main protocol class for ninetails."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Literal, Sequence

import gymnasium as gym
import numpy as np


class NinetailsVectorGymnasiumEnv:
    """NinetailsVectorGymnasiumEnv."""

    supported_observation_spaces: tuple[type[gym.Space], ...]
    supported_action_spaces: tuple[type[gym.Space], ...]

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: None | Literal["fork", "forkserver", "spawn"] = None,
        strict: bool = False,
    ):
        """__init__.

        Args:
            env_fns (list[Callable[[], gym.Env]]): env_fns
            start_method (None | Literal["fork", "forkserver", "spawn"]): start_method
            strict (bool): strict
        """
        raise NotImplementedError

    def _delete_spaces(self) -> None:
        """_delete_spaces.

        Returns:
            None:
        """
        if hasattr(self, "action_spaces"):
            del self.__dict__["action_spaces"]

    def sample_actions(self) -> np.ndarray:
        """sample_actions.

        Returns:
            np.ndarray:
        """
        return np.stack([space.sample() for space in self.action_spaces], axis=0)

    @cached_property
    def observation_spaces(self) -> list[gym.spaces.Space]:
        """observation_spaces.

        Returns:
            list[gym.spaces.Space]:
        """
        raise NotImplementedError

    @cached_property
    def action_spaces(self) -> list[gym.spaces.Space]:
        """action_spaces.

        Returns:
            list[gym.spaces.Space]:
        """
        raise NotImplementedError

    def step(
        self, actions: np.ndarray | Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """step.

        Args:
            actions (np.ndarray | Sequence[np.ndarray]): actions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """
        raise NotImplementedError

    def reset(
        self, seed: None | int = None
    ) -> tuple[np.ndarray, tuple[dict[str, Any]]]:
        """reset.

        Args:
            seed (None | int): seed

        Returns:
            tuple[np.ndarray, tuple[dict[str, Any]]]:
        """
        raise NotImplementedError

    def reset_single_env(
        self, env_idx: int, seed: None | int = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset_single_env.

        Args:
            env_idx (int): env_idx
            seed (None | int): seed

        Returns:
            tuple[np.ndarray, dict[str, Any]]:
        """
        raise NotImplementedError

    def close(self) -> None:
        """close.

        Returns:
            None:
        """
        raise NotImplementedError

    def render(self, **kwargs: Any) -> None | np.ndarray | list[np.ndarray]:
        """render.

        Args:
            kwargs (Any): kwargs

        Returns:
            None | np.ndarray | list[np.ndarray]:
        """
        raise NotImplementedError
