import pkg_resources
import numpy as np
from collections import deque
from paws_gym.motion.model import Model

class Bot(object):
    def __init__(self, pybullet_client, ctrl_freq=240, fixed=False):
        self._pybullet_client = pybullet_client
        self._ctrl_freq = ctrl_freq
        self._history = int(self._ctrl_freq // 2)

        self._fixed = fixed

        self._base_act_lower_bound = np.array([0.1,   np.pi/2, 0.0,  0.0,  0.0,  0.0,  0.01, 0.01, 0.01, 0.01, 0.08, 0.08, 0.08, 0.08, -np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2])
        self._base_act_upper_bound = np.array([1.0, 3*np.pi/2, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.12, 0.12, 0.12, 0.12,  np.pi/2,  np.pi/2,  np.pi/2,  np.pi/2])

        self._base_obs_lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self._base_obs_upper_bound = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])

        self._bot_model = Model("trot")

        self._obs_buffer = deque(maxlen=self._history)
        self._act_buffer = deque(maxlen=self._history)

        for _ in range(self._history):
            self._obs_buffer.append(np.zeros(self._base_obs_lower_bound.size))
        for _ in range(self._history):
            self._act_buffer.append(self._base_act_lower_bound)

        self.reset()

    def reset(self):
        self._bot_id = self._pybullet_client.loadURDF(pkg_resources.resource_filename('paws_gym', 'assets/paws/urdf/paws.urdf'),
                            [0, 0, 0.2],
                            [0, 0, 0, 1],
                            # flags = self._pybullet_client.URDF_USE_SELF_COLLISION,
                            useFixedBase=self._fixed)
        
        self._build_name_id_map()
        for _ in range(100):
            for motor in self._joint_name_to_id:
                self._set_angle_by_name(motor, 0)
                self._pybullet_client.stepSimulation()

    def step(self, elapsed_time, action):
        self._bot_model.fl_tg.frequency = action['frequency']
        self._bot_model.fr_tg.frequency = action['frequency']
        self._bot_model.bl_tg.frequency = action['frequency']
        self._bot_model.br_tg.frequency = action['frequency']
        self._bot_model.fl_tg.swing_stance_ratio = action['ratio']
        self._bot_model.fr_tg.swing_stance_ratio = action['ratio']
        self._bot_model.bl_tg.swing_stance_ratio = action['ratio']
        self._bot_model.br_tg.swing_stance_ratio = action['ratio']
        self._bot_model.fl_tg.adjustable_params = [action['step_lengths'][0], action['step_heights'][0], action['nominal_heights'][0], action['directions'][0]]
        self._bot_model.fr_tg.adjustable_params = [action['step_lengths'][1], action['step_heights'][1], action['nominal_heights'][1], action['directions'][1]]
        self._bot_model.bl_tg.adjustable_params = [action['step_lengths'][2], action['step_heights'][2], action['nominal_heights'][2], action['directions'][2]]
        self._bot_model.br_tg.adjustable_params = [action['step_lengths'][3], action['step_heights'][3], action['nominal_heights'][3], action['directions'][3]]

        (fl, fr, bl, br) = self._bot_model.compute(elapsed_time)

        self._set_angle_by_name('fl_j1', fl[0])
        self._set_angle_by_name('fl_j2', fl[1])
        self._set_angle_by_name('fl_j3', fl[2])
        self._set_angle_by_name('fr_j1', fr[0])
        self._set_angle_by_name('fr_j2', fr[1])
        self._set_angle_by_name('fr_j3', fr[2])
        self._set_angle_by_name('bl_j1', bl[0])
        self._set_angle_by_name('bl_j2', bl[1])
        self._set_angle_by_name('bl_j3', bl[2])
        self._set_angle_by_name('br_j1', br[0])
        self._set_angle_by_name('br_j2', br[1])
        self._set_angle_by_name('br_j3', br[2])

        pos, quat = self._pybullet_client.getBasePositionAndOrientation(self._bot_id)
        rpy = self._pybullet_client.getEulerFromQuaternion(quat)
        vel, ang_vel = self._pybullet_client.getBaseVelocity(self._bot_id)

        obs = np.array([pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2], vel[0], vel[1], vel[2], ang_vel[0], ang_vel[1], ang_vel[2]])
        act = np.concatenate([np.array([action['frequency']]), np.array([action['ratio']]), np.array(action['step_lengths']), np.array(action['step_heights']), np.array(action['nominal_heights']), np.array(action['directions'])])

        self._obs_buffer.append(obs)
        self._act_buffer.append(act)

    def _build_name_id_map(self):
        num_joints = self._pybullet_client.getNumJoints(self._bot_id)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self._bot_id, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        
    def _set_angle_by_id(self, motor_id, angle):
        self._pybullet_client.setJointMotorControl2(bodyIndex=self._bot_id,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.POSITION_CONTROL,
                                                targetPosition=angle)

    def _set_angle_by_name(self, motor_name, angle):
        self._set_angle_by_id(self._joint_name_to_id[motor_name], angle)

    def _action_space_bounds(self):
        return (self._base_act_lower_bound, self._base_act_upper_bound)
    
    def _observation_space_bounds(self):
        obs_lower_bound = np.array([])
        obs_upper_bound = np.array([])

        for _ in range(self._history):
            obs_lower_bound = np.hstack([obs_lower_bound, self._base_obs_lower_bound])
            obs_upper_bound = np.hstack([obs_upper_bound, self._base_obs_upper_bound])
        for _ in range(self._history):
            obs_lower_bound = np.hstack([obs_lower_bound, self._base_act_lower_bound])
            obs_upper_bound = np.hstack([obs_upper_bound, self._base_act_upper_bound])

        return (obs_lower_bound, obs_upper_bound)
    
    @property
    def obs_buffer(self):
        return self._obs_buffer
    
    @property
    def act_buffer(self):
        return self._act_buffer