from paws_gym.envs.base_env import BaseEnv
import numpy as np

class VelocityControlEnv(BaseEnv):
    def __init__(self, pyb_freq: int = 240, ctrl_freq: int = 240, fixed=False, gui=False):
        super().__init__(pyb_freq=pyb_freq, ctrl_freq=ctrl_freq, fixed=fixed, gui=gui)
        self.desired_velocity = [0.0, -0.5, 0.0] # vx, vy, w

    def _compute_reward(self):
        pos, quat = self._pybullet_client.getBasePositionAndOrientation(self._bot._bot_id)
        rpy = self._pybullet_client.getEulerFromQuaternion(quat)
        vel, ang_vel = self._pybullet_client.getBaseVelocity(self._bot._bot_id)

        desired_velocity_reward_term = -np.abs(self.desired_velocity[0] - vel[0]) - np.abs(self.desired_velocity[1] - vel[1]) - np.abs(self.desired_velocity[2] - ang_vel[2])
        pitch_roll_flat_reward_term = -np.exp(rpy[0]**2 + rpy[1]**2)+1
        change_action_reward_term = -np.sum((self.act_buffer[-1] - self.act_buffer[-2])**2)

        # print(desired_velocity_reward_term, pitch_roll_flat_reward_term, change_action_reward_term)

        reward = desired_velocity_reward_term + pitch_roll_flat_reward_term + change_action_reward_term
        return reward
    
    def _compute_terminated(self):
        return False
    
    def _compute_truncated(self):
        if (self.elapsed_time > 20):
            return True
        else:
            return False