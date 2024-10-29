from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from paws_gym.envs.velocity_control_env import VelocityControlEnv

if __name__ == "__main__":
    # env = VelocityControlEnv(ctrl_freq=10, fixed=False, gui=False)
    # check_env(env, warn=True)

    NAME = "PPO"
    LOG_PATH = f"./logs/"
    MODEL_PATH = f"./models/{NAME}"

    train_env = make_vec_env(VelocityControlEnv, n_envs=1, env_kwargs={"pyb_freq": 240, "ctrl_freq": 120, "fixed": False, "gui": False})
    model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_PATH)

    model.learn(total_timesteps=1000000, progress_bar=True)

    model.save(MODEL_PATH)