"""The main class for ninetails."""

from __future__ import annotations

import multiprocessing
import warnings
from functools import cached_property
from typing import Any, Callable, Literal, Sequence

import gymnasium as gym
import numpy as np

from ninetails.exceptions import SubProcessVectorEnvException
from ninetails.process_utils import CloudpickleWrapper, process_worker


class SubProcessVectorGymnasiumEnv:
    """SubProcessVectorGymnasiumEnv.

    Creates a multiprocess vectorized wrapper for multiple gymnasium environments.
    This distributes each environment to its own process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    **Warning**
        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.
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

        if self.num_envs == 1:
            warnings.warn(
                "You are attempting to create a vector environment with only 1 environment.\n"
                "This is probably undesirable and will create a fairly significant slowdown.\n"
                "We advise that you use `PseudoSubProcessVectorGymnasiumEnv` instead if the desired goal is the same signature but for 1 environment:\n"
                "\n"
                "from ninetails import PseudoSubProcessVectorGymnasiumEnv"
                "env = PseudoSubProcessVectorGymnasiumEnv(env_fn)"
            )

        self.waiting = False
        self.closed = False

        # define the forking method
        if start_method is None:
            start_method = (
                "forkserver"
                if ("forkserver" in multiprocessing.get_all_start_methods())
                else "spawn"
            )

        # grab the context to use with process forking
        ctx = multiprocessing.get_context(start_method)
        assert hasattr(ctx, "Process"), "Context does not have a Process attribute"

        # define N number of local and remote connections
        # we only care about the local pipes
        self.pipes, remote_pipes = zip(
            *[ctx.Pipe(duplex=True) for _ in range(self.num_envs)]
        )

        # define N number of processes, one for each env, and spin them off
        self.processes: list[multiprocessing.Process] = []
        for local, remote, env_fn in zip(self.pipes, remote_pipes, env_fns):
            args = (remote, local, CloudpickleWrapper(env_fn))
            process = ctx.Process(  # pyright: ignore[reportAttributeAccessIssue]
                target=process_worker, args=args, daemon=True
            )
            process.start()
            self.processes.append(process)
            remote.close()

        # check that the action and observation spaces are np arrays
        for i, space in enumerate(self.observation_spaces):
            if not isinstance(space, self.supported_observation_spaces):
                raise SubProcessVectorEnvException(
                    f"Usage of SubProcVecEnv requires that observation spaces are `gymnasium.spaces.Box` type, got {space=} for environment {i}."
                )
        for i, space in enumerate(self.action_spaces):
            if not isinstance(space, self.supported_action_spaces):
                raise SubProcessVectorEnvException(
                    f"Usage of SubProcVecEnv requires that action spaces are `gymnasium.spaces.Box` type, got {space=} for environment {i}."
                )

    @cached_property
    def observation_spaces(self) -> list[gym.spaces.Space]:
        """Get a list of observation spaces of all the underlying envs.

        This effectively creates an identical copy of all the env spaces.

        Returns:
            list[gym.spaces.Space]:
        """
        for pipe in self.pipes:
            pipe.send(("get_observation_space", None))
        return [pipe.recv() for pipe in self.pipes]

    @cached_property
    def action_spaces(self) -> list[gym.spaces.Space]:
        """Get a list of action spaces of all the underlying envs.

        This effectively creates an identical copy of all the env spaces.

        Returns:
            list[gym.spaces.Space]:
        """
        for pipe in self.pipes:
            pipe.send(("get_action_space", None))
        return [pipe.recv() for pipe in self.pipes]

    def step(
        self, actions: np.ndarray | Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """step.

        Args:
            actions (np.ndarray | Sequence[np.ndarray]): actions

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """
        self._send_actions(actions)
        return self._receive_transitions()

    def _send_actions(self, actions: np.ndarray | Sequence[np.ndarray]) -> None:
        """Sends a list of actions to each environment.

        Args:
            actions (np.ndarray | Sequence[np.ndarray]): actions

        Returns:
            None:
        """
        # check the space and number of actions
        if self.strict:
            assert len(actions) == self.num_envs
            for i, (action, space) in enumerate(zip(actions, self.action_spaces)):
                if not space.contains(action):
                    SubProcessVectorEnvException(
                        f"Action space of environment {i} ({space}) does not contain {action=}. To disable this check, set `strict=False` on init"
                    )

        # pass the action to the remote through the pipe
        for pipe, action in zip(self.pipes, actions):
            pipe.send(("step", action))

        # flag that we are waiting
        self.waiting = True

    def _receive_transitions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """recv_transition.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[dict[str, Any]]]:
        """
        # collect the results from the remote
        results = [pipe.recv() for pipe in self.pipes]
        self.waiting = False

        # unpack the results
        obs, rews, terms, truncs, infos = zip(*results)
        obs = np.stack(obs, axis=0)
        rews = np.stack(rews, axis=0)
        terms = np.stack(terms, axis=0)
        truncs = np.stack(truncs, axis=0)

        # check the observation space
        if self.strict:
            for i, (ob, space) in enumerate(zip(obs, self.action_spaces)):
                if not space.contains(ob):
                    SubProcessVectorEnvException(
                        f"Observation space of environment {i} ({space}) does not contain {ob=}. To disable this check, set `strict=False` on init"
                    )

        # return things
        return obs, rews, terms, truncs, infos

    def _delete_spaces(self) -> None:
        """_delete_spaces.

        Returns:
            None:
        """
        if hasattr(self, "action_spaces"):
            del self.__dict__["action_spaces"]

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

        for pipe in self.pipes:
            pipe.send(("reset", seed))
        results = [pipe.recv() for pipe in self.pipes]

        # unpack the results
        obs, infos = zip(*results)
        obs = np.stack(obs, axis=0)

        # check the observation space
        if self.strict:
            for i, (ob, space) in enumerate(zip(obs, self.action_spaces)):
                if not space.contains(ob):
                    SubProcessVectorEnvException(
                        f"Observation space of environment {i} ({space}) does not contain {ob=}. To disable this check, set `strict=False` on init"
                    )

        return obs, infos

    def reset_single_env(
        self, env_idx: int, seed: None | int = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """reset_single_env.

        Args:
            env_idx (int): env_idx
            seed (None | int): seed
        """
        # on reset, we need to reset the observation and action spaces
        # this does not affect the spaces of envs not seeded
        self._delete_spaces()

        pipe = self.pipes[env_idx]
        pipe.send(("reset", seed))
        ob, info = pipe.recv()

        # check the observation space
        if self.strict:
            space = self.observation_spaces[env_idx]
            if not space.contains(ob):
                SubProcessVectorEnvException(
                    f"Observation space of environment {env_idx} ({space}) does not contain {ob=}. To disable this check, set `strict=False` on init"
                )

        return ob, info

    def close(self) -> None:
        """close.

        Returns:
            None:
        """
        # if already closed just return
        if self.closed:
            return

        # await all remaining processes
        if self.waiting:
            for pipe in self.pipes:
                pipe.recv()

        # send close to all pipes
        for pipe in self.pipes:
            pipe.send(("close", None))

        # await all process to close
        for process in self.processes:
            process.join()

        self.closed = True

    def render(self, **kwargs: Any) -> np.ndarray:
        """render.

        Args:
            kwargs (Any): kwargs

        Returns:
            Sequence[np.ndarray]:
        """
        for pipe in self.pipes:
            pipe.send(("render", kwargs))
        imgs = [pipe.recv() for pipe in self.pipes]

        return np.stack(imgs, axis=0)

    def sample_actions(self) -> np.ndarray:
        """sample_actions.

        Returns:
            np.ndarray:
        """
        return np.stack([space.sample() for space in self.action_spaces])
