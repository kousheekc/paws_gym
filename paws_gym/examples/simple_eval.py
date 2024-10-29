from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from paws_gym.envs.velocity_control_env import VelocityControlEnv

if __name__ == "__main__":
    # env = BaseMassSpringDamperEnv("step")
    # check_env(env, warn=True)

    NAME = "PPO"

    model = PPO.load(f"./models/{NAME}")

    # eval_env = make_vec_env(BaseM assSpringDamperEnv, n_envs=16, env_kwargs={"randomise": True, "setpoint_type": "ramp", "pd_control": True, "rl_control": True, "ff_obs": False})
    # mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    env = VelocityControlEnv(pyb_freq=240, ctrl_freq=120, fixed=False, gui=True)
    obs, _ = env.reset()

    for _ in range(10000):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Reward: {reward}")
