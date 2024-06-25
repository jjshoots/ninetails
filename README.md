# Ninetails

A wrapper for creating vectorized gymnasium environments.

## Installation

`pip3 install ninetails`

## Usage

```py
import gymnasium as gym
import numpy as np

from ninetails import SubProcessVectorGymnasiumEnv


def main() -> None:
    """main.

    Returns:
        None:
    """
    # define your environment using a function that returns the environment here
    env_fns = [lambda i=i: gym.make("MountainCarContinuous-v0") for i in range(1)]

    # create a vectorized environment
    # `strict` is useful here for debugging
    vec_env = SubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)

    # define our initial termination and trunction arrays
    terminations, truncations = np.array([False]), np.array([False])

    # reset follows the same signature as a Gymnasium environment
    observations, infos = vec_env.reset(seed=42)

    for step_count in range(5000):
        # sample an action, this is an np.ndarray of [num_envs, *env.action_space.shape]
        actions = vec_env.sample_actions()

        # similarly, the step function follows the same signature as a Gymnasium environment with the following shapes
        # observations: np.ndarray of shape [num_envs, *env.observation_space.shape]
        # rewards: np.ndarray of shape [num_envs, 1]
        # terminations: np.ndarray of shape [num_envs, 1]
        # truncations: np.ndarray of shape [num_envs, 1]
        # infos: tuple[dict[str, Any]]
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)

        # to reset underlying environments
        done_ids = set(np.where(terminations).tolist() + np.where(truncations).tolist())
        for id in done_ids:
            # warning, you'll have to handle starting observations yourself here
            reset_obs, reset_info = vec_env.reset(id)


if __name__ == "__main__":
    main()
```
