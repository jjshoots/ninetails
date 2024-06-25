"""The pseudo class for ninetails with only 1 env."""

from __future__ import annotations

from functools import cached_property
from typing import Any, Callable, Literal, Sequence

import gymnasium as gym
import numpy as np

from ninetails.exceptions import SubProcessVectorEnvException
from ninetails.protocol import NinetailsVectorGymnasiumEnv


class PseudoSubProcessVectorGymnasiumEnv(NinetailsVectorGymnasiumEnv):
    """PseudoSubProcessVectorGymnasiumEnv.

    This has the same signature as SubProcessVectorGymnasiumEnv, but is only for 1 environment and is much faster.
    """

    supported_observation_spaces = (gym.spaces.Box, gym.spaces.Discrete)
    supported_action_spaces = (gym.spaces.Box, gym.spaces.Discrete)

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        start_method: None | Literal["fork", "forkserver", "spawn"] = None,
        strict: bool = False,
    ):
        """__init__.

        Args:
            env_fns (list[Callable[[], gym.Env]]): Functions that return instances of Gymnasium environments.
            start_method (None | Literal["fork", "forkserver", "spawn"]): The start method to use for creating parallel envs.
            strict (bool): Whether to check certain assertations for a strict implementation. Useful for debugging.
        """
        self.strict = strict
        self.num_envs = len(env_fns)

        if not self.num_envs == 1:
            raise SubProcessVectorEnvException("This class only accepts 1 env_fn.")

        self._env = env_fns[0]()

    @cached_property
    def observation_spaces(self) -> list[gym.spaces.Space]:
        """observation_spaces.

        Returns:
            list[gym.spaces.Space]:
        """
        return [self._env.observation_space]

    @cached_property
    def action_spaces(self) -> list[gym.spaces.Space]:
        """action_spaces.

        Returns:
            list[gym.spaces.Space]:
        """
        return [self._env.action_space]

    def step(
        self, actions: np.ndarray | Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """step.

        Args:
            actions (np.ndarray | Sequence[np.ndarray]): actions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """
        if self.strict:
            action = actions[0]
            if not self.action_spaces[0].contains(action):
                SubProcessVectorEnvException(
                    f"Action space of environment 0 ({self.action_spaces[0]}) does not contain {action=}. To disable this check, set `strict=False` on init"
                )

        obs, rew, term, trunc, info = self._env.step(actions[0])

        if self.strict:
            if not self.observation_spaces[0].contains(obs):
                SubProcessVectorEnvException(
                    f"Observation space of environment 0 ({self.observation_spaces[0]}) does not contain {obs=}. To disable this check, set `strict=False` on init"
                )

        return (
            np.expand_dims(obs, axis=0),
            np.expand_dims(float(rew), axis=0),
            np.expand_dims(term, axis=0),
            np.expand_dims(trunc, axis=0),
            (info,),
        )

    def reset(
        self, seed: None | int = None
    ) -> tuple[np.ndarray, tuple[dict[str, Any]]]:
        """reset.

        Args:
            seed (None | int): seed

        Returns:
            tuple[np.ndarray, tuple[dict[str, Any]]]:
        """
        # on reset, we need to reset the observation and action spaces
        # this does not affect the spaces of envs not seeded
        self._delete_spaces()

        obs, info = self._env.reset(seed=seed)
        self._env.action_space.seed(seed=seed)

        # check the observation space
        if self.strict:
            if not self.observation_spaces[0].contains(obs):
                SubProcessVectorEnvException(
                    f"Observation space of environment 0 ({self.observation_spaces[0]}) does not contain {obs=}. To disable this check, set `strict=False` on init"
                )

        return (
            np.expand_dims(obs, axis=0),
            (info,),
        )

    def reset_single_env(
        self, env_idx: int, seed: None | int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset_single_env.

        Args:
            env_idx (int): env_idx
            seed (None | int): seed
        """
        # on reset, we need to reset the observation and action spaces
        # this does not affect the spaces of envs not seeded
        self._delete_spaces()

        assert env_idx == 0, "No other environment except index 0"
        obs, info = self._env.reset(seed=seed)
        self._env.action_space.seed(seed=seed)

        # check the observation space
        if self.strict:
            if not self.observation_spaces[0].contains(obs):
                SubProcessVectorEnvException(
                    f"Observation space of environment 0 ({self.observation_spaces[0]}) does not contain {obs=}. To disable this check, set `strict=False` on init"
                )

        return obs, info

    def close(self) -> None:
        """close.

        Returns:
            None:
        """
        self._env.close()

    def render(self, **kwargs: Any) -> None | np.ndarray | list[np.ndarray]:
        """render.

        Args:
            kwargs (Any): kwargs

        Returns:
            Sequence[np.ndarray]:
        """
        return self._env.render()
