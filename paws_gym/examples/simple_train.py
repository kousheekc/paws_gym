from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from paws_gym.envs.base_env import BaseEnv

if __name__ == "__main__":
    env = BaseEnv(ctrl_freq=2, fixed=False, gui=False)
    check_env(env, warn=True)

    # NAME = "RAMP_PPO_MLP_PD_RL_FF_jump_countdown"
    # LOG_PATH = f"./logs/{NAME}/"
    # MODEL_PATH = f"./models/{NAME}"

    # train_env = make_vec_env(BaseEnv, n_envs=64, env_kwargs={"ctrl_freq": 10, "fixed": False, "gui": False})
    # train_logger = configure(LOG_PATH, ["stdout", "csv", "tensorboard"])
    # model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=LOG_PATH)
    # model.set_logger(train_logger)

    # model.learn(total_timesteps=3000000, progress_bar=True)

    # model.save(MODEL_PATH)