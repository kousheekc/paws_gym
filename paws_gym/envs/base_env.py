import pybullet
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from paws_gym.envs.bot_env import Bot

class BaseEnv(gym.Env):
    def __init__(self, pyb_freq: int = 240, ctrl_freq: int = 240, fixed=False, gui=False):
        self.pyb_freq = pyb_freq
        self.ctrl_freq = ctrl_freq
        if self.pyb_freq % self.ctrl_freq != 0:
            raise ValueError('[ERROR] in BaseEnv.__init__(), pyb_freq is not divisible by env_freq.')
        
        self.pyb_steps_per_control = int(self.pyb_freq / self.ctrl_freq)
        self.pyb_timestep = 1. / self.pyb_freq
        self.history = int(self.ctrl_freq // 2)
        self.elapsed_time = 0.0

        self.fixed = fixed
        self.GUI = gui

        if self.GUI:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()

        self._init_state = None
        self._init()

        self._obs_buffer = deque(maxlen=self.history)
        self._act_buffer = deque(maxlen=self.history)

        for _ in range(self.history):
            self._obs_buffer.append(np.zeros(self._bot._base_obs_lower_bound.size))
        for _ in range(self.history):
            self._act_buffer.append(self._bot._base_act_lower_bound)

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
    
    def reset(self, seed : int = None, options : dict = None):
        if (self._init_state is not None):
            self._pybullet_client.restoreState(self._init_state)
        else:
            self._init()

        obs = self._compute_obs()
        info = self._compute_info()

        return obs, info

    def step(self, action):
        action_dict = {
            'frequency': action[0],
            'ratio': action[1],
            'step_lengths': [action[2], action[3], action[4], action[5]],
            'step_heights': [action[6], action[7], action[8], action[9]],
            'nominal_heights': [action[10], action[11], action[12], action[13]],
            'directions': [action[14], action[15], action[16], action[17]],
        }
        for _ in range(self.pyb_steps_per_control):
            self._bot.step(self.elapsed_time, action_dict)
            self.elapsed_time += self.pyb_timestep
            self._pybullet_client.stepSimulation()

        pos, quat = self._pybullet_client.getBasePositionAndOrientation(self._bot._bot_id)
        rpy = self._pybullet_client.getEulerFromQuaternion(quat)
        vel, ang_vel = self._pybullet_client.getBaseVelocity(self._bot._bot_id)

        obs_current = np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], vel[0], vel[1], vel[2], ang_vel[0], ang_vel[1], ang_vel[2]])
        act_current = np.concatenate([np.array([action_dict['frequency']]), np.array([action_dict['ratio']]), np.array(action_dict['step_lengths']), np.array(action_dict['step_heights']), np.array(action_dict['nominal_heights']), np.array(action_dict['directions'])])

        self._obs_buffer.append(obs_current)
        self._act_buffer.append(act_current)

        obs = self._compute_obs()
        reward = self._compute_reward()
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        info = self._compute_info()

        return obs, reward, terminated, truncated, info

    def close(self):
        self._pybullet_client.disconnect()

    def _init(self):    
        self._pybullet_client.resetSimulation()
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._pybullet_client.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._pybullet_client.setGravity(0, 0, -9.81)
        self._pybullet_client.setTimeStep(self.pyb_timestep)
        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._plane_id = self._pybullet_client.loadURDF("plane.urdf")
        self._bot = Bot(self._pybullet_client, ctrl_freq=self.ctrl_freq, fixed=self.fixed)  

        self.elapsed_time = 0.0   
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)

        self._init_state=self._pybullet_client.saveState()

    def _observation_space(self):
        bot_obs = self._bot._observation_space_bounds()
        return spaces.Box(low=bot_obs[0], high=bot_obs[1], dtype=np.float32)

    def _action_space(self):
        bot_act = self._bot._action_space_bounds()
        return spaces.Box(low=bot_act[0], high=bot_act[1], dtype=np.float32)
    
    def _compute_obs(self):
        obs_complete = np.array([])
        for obs in self._obs_buffer:
            obs_complete = np.hstack([obs_complete, obs])
        for act_obs in self._act_buffer:
            obs_complete = np.hstack([obs_complete, act_obs])

        return obs_complete.astype(np.float32)

    def _compute_reward(self):
        raise NotImplementedError
    
    def _compute_terminated(self):
        raise NotImplementedError
    
    def _compute_truncated(self):
        raise NotImplementedError
    
    def _compute_info(self):
        return {}
    
    @property
    def obs_buffer(self):
        return self._obs_buffer
    
    @property
    def act_buffer(self):
        return self._act_buffer


