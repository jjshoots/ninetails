"""Core tests."""

from __future__ import annotations

import itertools
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from ninetails import PseudoSubProcessVectorGymnasiumEnv, SubProcessVectorGymnasiumEnv
from tests.utils import all_equal

_NUM_ENV_RANGE = [1, 2, 4]

_ENV_NAMES = [
    "MountainCarContinuous-v0",
    "MountainCar-v0",
]


@pytest.mark.parametrize(
    "env_name, num_envs", itertools.product(_ENV_NAMES, _NUM_ENV_RANGE)
)
def test_core(env_name: str, num_envs: int) -> None:
    """Test that the core functionality works.

    Args:
        env_name (str): env_name
        num_envs (int): num_envs

    Returns:
        None:
    """
    # create the env
    env_fns = [lambda i=i: gym.make(env_name) for i in range(num_envs)]
    vec_env = SubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)

    # reset
    terminations, truncations = np.array([False]), np.array([False])
    observations, infos = vec_env.reset()

    # step through things
    while not np.any(terminations) and not np.any(truncations):
        actions = vec_env.sample_actions()
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)


@pytest.mark.parametrize("env_name, num_envs", itertools.product(_ENV_NAMES, [1]))
def test_seedability(env_name: str, num_envs: int) -> None:
    """Test that seeding works.

    Args:
        env_name (str): env_name
        num_envs (int): num_envs

    Returns:
        None:
    """
    # how many steps to take in the environment
    seed = 42
    num_steps = 5

    # create the env
    env_fns = [lambda i=i: gym.make(env_name) for i in range(num_envs)]
    vec_env = SubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)

    # a place we store observations
    results1: list[tuple[Any, ...]] = []
    results2: list[tuple[Any, ...]] = []

    # first pass
    vec_env.reset(seed=seed)
    for i in range(num_steps):
        actions = vec_env.sample_actions()
        results1.append(vec_env.step(actions))

    # second pass
    vec_env.reset(seed=seed)
    for i in range(num_steps):
        actions = vec_env.sample_actions()
        results2.append(vec_env.step(actions))

    assert all_equal(results1, results2)


@pytest.mark.parametrize("env_name", _ENV_NAMES)
def test_pseudo(env_name: str) -> None:
    """Test that pseudo vec env performs the same.

    Args:
        env_name (str): env_name

    Returns:
        None:
    """
    # how many steps to take in the environment
    seed = 42
    num_steps = 5

    env_fns = [lambda i=i: gym.make(env_name) for i in range(1)]
    vec_env = SubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)
    p_vec_env = PseudoSubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)

    # a place we store observations
    results1: list[tuple[Any, ...]] = []
    results2: list[tuple[Any, ...]] = []

    # first pass
    vec_env.reset(seed=seed)
    for i in range(num_steps):
        actions = vec_env.sample_actions()
        print(actions)
        results1.append(vec_env.step(actions))

    # second pass
    p_vec_env.reset(seed=seed)
    for i in range(num_steps):
        actions = p_vec_env.sample_actions()
        print(actions)
        results2.append(p_vec_env.step(actions))

    assert all_equal(results1, results2)
