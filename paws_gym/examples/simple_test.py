from paws_gym.envs.velocity_control_env import VelocityControlEnv
import numpy as np

if __name__ == "__main__":
    env = VelocityControlEnv(pyb_freq=240, ctrl_freq=240, fixed=False, velocity_control=True, gui=True)
    print('[INFO] Observation space:', env.observation_space)
    print('[INFO] Action space:', env.action_space)

    env.reset()

    total_reward = 0
    count = 0
    while True:
        obs, rew, done, truncated, info = env.step(np.zeros(12))
        total_reward += rew
        count += 1
        if truncated:
            break

    print(total_reward, count)