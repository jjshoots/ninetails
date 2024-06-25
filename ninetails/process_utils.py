"""Stuff for handling a single process worker."""

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

import cloudpickle


class CloudpickleWrapper(object):
    """CloudpickleWrapper."""

    def __init__(self, var: Any):
        """__init__.

        Args:
            var (Any): var
        """
        self.var = var

    def __getstate__(self) -> Any:
        """__getstate__.

        Returns:
            Any:
        """
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs: Any) -> None:
        """__setstate__.

        Args:
            obs (Any): obs

        Returns:
            None:
        """
        self.var = cloudpickle.loads(obs)


def process_worker(
    remote: Connection, parent_remote: Connection, pickled_env: CloudpickleWrapper
) -> None:
    """process_worker.

    Args:
        remote (Connection): remote
        parent_remote (Connection): parent_remote
        pickled_env (CloudpickleWrapper): pickled_env

    Returns:
        None:
    """
    parent_remote.close()
    env = pickled_env.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                remote.send((observation, reward, terminated, truncated, info))
            elif cmd == "reset":
                observation, info = env.reset(seed=data)
                if data:
                    env.action_space.seed(data)
                remote.send((observation, info))
            elif cmd == "render":
                remote.send(env.render(**data))
            elif cmd == "close":
                env.close()
                remote.close()
            elif cmd == "get_action_space":
                remote.send(env.action_space)
            elif cmd == "get_observation_space":
                remote.send(env.observation_space)
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(
                    "`{}` is not implemented in the worker".format(cmd)
                )
        except EOFError:
            break
