from paws_gym.envs.velocity_control_env import VelocityControlEnv
import numpy as np

if __name__ == "__main__":
    env = VelocityControlEnv(pyb_freq=240, ctrl_freq=4, fixed=False, gui=True)
    print('[INFO] Observation space:', env.observation_space)
    print('[INFO] Action space:', env.action_space)

    env.reset()

    total_reward = 0
    count = 0
    while True:
        obs, rew, done, truncated, info = env.step(env.action_space.sample())
        total_reward += rew
        count += 1
        if truncated or done:
            break

    print(total_reward, count)