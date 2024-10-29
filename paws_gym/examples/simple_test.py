from paws_gym.envs.velocity_control_env import VelocityControlEnv
import numpy as np

if __name__ == "__main__":
    env = VelocityControlEnv(pyb_freq=240, ctrl_freq=10, fixed=True, gui=True)
    print('[INFO] Observation space:', env.observation_space)

    env.reset()

    while True:
        obs, rew, done, truncated, info = env.step(env.action_space.sample())
        # print(obs)