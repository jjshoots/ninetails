"""Example usage of ninetails."""
import gymnasium as gym
import numpy as np

from ninetails import SubProcessVectorGymnasiumEnv


def main() -> None:
    """main.

    Returns:
        None:
    """
    env_fns = [lambda i=i: gym.make("MountainCarContinuous-v0") for i in range(4)]
    vec_env = SubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)

    terminations, truncations = np.array([False]), np.array([False])
    observations, infos = vec_env.reset()

    while not np.any(terminations) and not np.any(truncations):
        actions = vec_env.sample_actions()
        observations, rewards, terminations, truncations, infos = vec_env.step(actions)


if __name__ == "__main__":
    main()
