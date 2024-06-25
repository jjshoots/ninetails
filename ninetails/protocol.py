"""The main protocol class for ninetails."""

from __future__ import annotations
import numpy as np
from functools import cached_property
import gymnasium as gym
from typing import Any, Callable, Literal, Protocol, Sequence


class NinetailsVectorGymnasiumEnv:
    supported_observation_spaces: tuple[type[gym.Space], ...]
        supported_action_spaces: tuple[type[gym.Space], ...]

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: None | Literal["fork", "forkserver", "spawn"] = None,
        strict: bool = False,
    ):
        raise NotImplementedError

    @cached_property
    def observation_spaces(self) -> list[gym.spaces.Space]:
        raise NotImplementedError

    @cached_property
    def action_spaces(self) -> list[gym.spaces.Space]:
        raise NotImplementedError

    def step(
        self, actions: np.ndarray | Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
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

    def reset(
        self, seed: None | int
    ) -> tuple[np.ndarray, tuple[dict[str, Any]]]:
        raise NotImplementedError

    def reset_single_env(
        self, env_idx: int, seed: None | int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def render(self, **kwargs: Any) -> None | np.ndarray | list[np.ndarray]:
        raise NotImplementedError
