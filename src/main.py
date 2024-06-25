import gymnasium as gym

from spv_gymnasium import SubProcessVectorGymnasiumEnv


def main() -> None:
    env_fns = [lambda i=i: gym.make("CarRacing-v2") for i in range(16)]
    vec_env = SubProcessVectorGymnasiumEnv(env_fns=env_fns, strict=True)

    term, trunc = False, False
    observations, infos = vec_env.reset()

    print(observations)
    print(observations.shape)
    print(infos)


if __name__ == "__main__":
    main()
