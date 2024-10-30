import pkg_resources
import numpy as np
from paws_gym.motion.model import Model
from paws_gym.motion.motor import Motor

class Bot(object):
    def __init__(self, pybullet_client, fixed=False, velocity_control=True):
        self._pybullet_client = pybullet_client

        self._fixed = fixed
        self._velocity_control = velocity_control

        # Foot residuals
        self._base_act_lower_bound = np.array([-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01])
        self._base_act_upper_bound = np.array([ 0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01])

        # posx, posy, posz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz
        self._base_obs_lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self._base_obs_upper_bound = np.array([ np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf])

        self._bot_model = Model("trot")
        self._motor_model = Motor(5.0, 10)

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
        # action = foot residuals 4 x (delta_x, delta_y, delta_z)
        mapped_action = []
        for i, act in enumerate(action):
            mapped_action.append(float(self._map(act, self._base_act_lower_bound[i], self._base_act_upper_bound[i])))

        # self._bot_model.fl_tg.adjustable_params = [mapped_action[0], mapped_action[1], mapped_action[2], mapped_action[3], mapped_action[4]]
        # self._bot_model.fr_tg.adjustable_params = [mapped_action[0], mapped_action[1], mapped_action[2], mapped_action[3], mapped_action[4]]
        # self._bot_model.bl_tg.adjustable_params = [mapped_action[0], mapped_action[1], mapped_action[2], mapped_action[3], mapped_action[4]]
        # self._bot_model.br_tg.adjustable_params = [mapped_action[0], mapped_action[1], mapped_action[2], mapped_action[3], mapped_action[4]]

        (fl, fr, bl, br) = self._bot_model.compute(elapsed_time, mapped_action)

        self._apply_motor_command('fl_j1', fl[0], self._velocity_control)
        self._apply_motor_command('fl_j2', fl[1], self._velocity_control)
        self._apply_motor_command('fl_j3', fl[2], self._velocity_control)
        self._apply_motor_command('fr_j1', fr[0], self._velocity_control)
        self._apply_motor_command('fr_j2', fr[1], self._velocity_control)
        self._apply_motor_command('fr_j3', fr[2], self._velocity_control)
        self._apply_motor_command('bl_j1', bl[0], self._velocity_control)
        self._apply_motor_command('bl_j2', bl[1], self._velocity_control)
        self._apply_motor_command('bl_j3', bl[2], self._velocity_control)
        self._apply_motor_command('br_j1', br[0], self._velocity_control)
        self._apply_motor_command('br_j2', br[1], self._velocity_control)
        self._apply_motor_command('br_j3', br[2], self._velocity_control)
            

    def _map(self, val, min, max):
        return min + (val + 1) * (max - min) / 2

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

    def _set_velocity_by_id(self, motor_id, omega):
        self._pybullet_client.setJointMotorControl2(bodyIndex=self._bot_id,
                                                jointIndex=motor_id,
                                                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                                                targetVelocity=omega)
        
    def _set_velocity_by_name(self, motor_name, omega):
        self._set_velocity_by_id(self._joint_name_to_id[motor_name], omega)

    def _get_joint_data_by_id(self, motor_id):
        (position, velocity, _, _) = self._pybullet_client.getJointState(self._bot_id, motor_id)
        return (position, velocity)

    def _get_joint_data_by_name(self, motor_name):
        return self._get_joint_data_by_id(self._joint_name_to_id[motor_name])
    
    def _apply_motor_command(self, motor_name, target, velocity_control):
        if velocity_control:
            data = self._get_joint_data_by_name(motor_name)
            omega = self._motor_model.calculate_velocity_command(target, data[0])
            self._set_velocity_by_name(motor_name, omega)
        else:
            self._set_angle_by_name(motor_name, target)


    def _action_space_bounds(self):
        return (-1*np.ones(self._base_act_lower_bound.size), 1*np.ones(self._base_act_upper_bound.size))
    
    def _observation_space_bounds(self):
        obs_lower_bound = np.hstack([self._base_obs_lower_bound, -1*np.ones(self._base_act_lower_bound.size)])
        obs_upper_bound = np.hstack([self._base_obs_upper_bound,  1*np.ones(self._base_act_upper_bound.size)])
        return (obs_lower_bound, obs_upper_bound)
    
