import pkg_resources
import numpy as np
from paws_gym.motion.model import Model

class Bot(object):
    def __init__(self, pybullet_client, ctrl_freq=240, fixed=False):
        self._pybullet_client = pybullet_client
        self._ctrl_freq = ctrl_freq
        self._history = int(self._ctrl_freq // 2)

        self._fixed = fixed

        self._base_act_lower_bound = np.array([0.0,  0.01, 0.10, -np.pi/2])
        self._base_act_upper_bound = np.array([0.03, 0.03, 0.14,  np.pi/2])

        self._base_obs_lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self._base_obs_upper_bound = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])

        self._bot_model = Model("trot")

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
        # self._bot_model.fl_tg.frequency = action[0]
        # self._bot_model.fr_tg.frequency = action[0]
        # self._bot_model.bl_tg.frequency = action[0]
        # self._bot_model.br_tg.frequency = action[0]
        # self._bot_model.fl_tg.swing_stance_ratio = action[1]
        # self._bot_model.fr_tg.swing_stance_ratio = action[1]
        # self._bot_model.bl_tg.swing_stance_ratio = action[1]
        # self._bot_model.br_tg.swing_stance_ratio = action[1]
        self._bot_model.fl_tg.adjustable_params = [action[0], action[1], action[2], action[3]]
        self._bot_model.fr_tg.adjustable_params = [action[0], action[1], action[2], action[3]]
        self._bot_model.bl_tg.adjustable_params = [action[0], action[1], action[2], action[3]]
        self._bot_model.br_tg.adjustable_params = [action[0], action[1], action[2], action[3]]

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
    
