from paws_gym.envs.base_env import BaseEnv
import numpy as np

if __name__ == "__main__":
    env = BaseEnv(gui=True, ctrl_freq=30)
    print('[INFO] Observation space:', env.observation_space)


    action = env.action_space.sample()

    while True:
        env.step(action)