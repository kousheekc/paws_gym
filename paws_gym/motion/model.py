import numpy as np
from paws_gym.motion.tg import ParametricTrajectoryGenerator
from paws_gym.motion.leg import Leg

class Model:
    def __init__(self, gait):
        self._gait_params = {
            'crawl': {'phases': [0, np.pi, np.pi/2, 3*np.pi/2], 'ratio': 3*np.pi/8},
            'trot': {'phases': [0, np.pi, np.pi, 0], 'ratio': np.pi},
            'bound': {'phases': [0, 0, np.pi, np.pi], 'ratio': np.pi}
        }

        assert gait in self._gait_params, f"'{gait}' is not a valid gait."

        self._fl_leg = Leg("fl", 0.038, 0.08, 0.08)
        self._fr_leg = Leg("fr", 0.038, 0.08, 0.08)
        self._bl_leg = Leg("bl", 0.038, 0.08, 0.08)
        self._br_leg = Leg("br", 0.038, 0.08, 0.08)
        
        self._fl_tg = ParametricTrajectoryGenerator(self._gait_params[gait]['phases'][0], 0.8, 0.01, 0.02, 0.12, self._gait_params[gait]['ratio'], 0)
        self._fr_tg = ParametricTrajectoryGenerator(self._gait_params[gait]['phases'][1], 0.8, 0.01, 0.02, 0.12, self._gait_params[gait]['ratio'], 0)
        self._bl_tg = ParametricTrajectoryGenerator(self._gait_params[gait]['phases'][2], 0.8, 0.01, 0.02, 0.12, self._gait_params[gait]['ratio'], 0)
        self._br_tg = ParametricTrajectoryGenerator(self._gait_params[gait]['phases'][3], 0.8, 0.01, 0.02, 0.12, self._gait_params[gait]['ratio'], 0)

    def compute(self, t, residuals):
        fl_pos = self._fl_tg.compute(t)
        fr_pos = self._fr_tg.compute(t)
        bl_pos = self._bl_tg.compute(t)
        br_pos = self._br_tg.compute(t)

        fl_pos[0] += residuals[0]
        fl_pos[1] += residuals[1]
        fl_pos[2] += residuals[2]
        fr_pos[0] += residuals[3]
        fr_pos[1] += residuals[4]
        fr_pos[2] += residuals[5]
        bl_pos[0] += residuals[6]
        bl_pos[1] += residuals[7]
        bl_pos[2] += residuals[8]
        br_pos[0] += residuals[9]
        br_pos[1] += residuals[10]
        br_pos[2] += residuals[11]

        fl_joint = self._fl_leg.compute(fl_pos)
        fr_joint = self._fr_leg.compute(fr_pos)
        bl_joint = self._bl_leg.compute(bl_pos)
        br_joint = self._br_leg.compute(br_pos)

        return (fl_joint, fr_joint, bl_joint, br_joint)
    
    @property
    def fl_tg(self): 
        return self._fl_tg 
    
    @property
    def fr_tg(self): 
        return self._fr_tg 
    
    @property
    def bl_tg(self): 
        return self._bl_tg 
    
    @property
    def br_tg(self): 
        return self._br_tg 