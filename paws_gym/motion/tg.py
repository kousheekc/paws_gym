import numpy as np

class ParametricTrajectoryGenerator:
    def __init__(self, phase_offset, frequency, step_length, step_height, nominal_height, swing_stance_ratio, direction):
        self._phase_offset = phase_offset
        self._frequency = frequency
        self._step_length = step_length
        self._step_height = step_height
        self._nominal_height = nominal_height
        self._swing_stance_ratio = swing_stance_ratio
        self._direction = direction

    def _mod_2_pi(self, x):
        return x % (2 * np.pi)

    def _forward_backward(self, phase):
        assert 0 <= phase <= 2 * np.pi, f"phase must be in the range [0, 2π), but got phase={phase}"

        if 0 <= phase < self._swing_stance_ratio:
            return -self._step_length * np.cos(np.pi * phase / self._swing_stance_ratio)
        elif self._swing_stance_ratio <= phase < 2*np.pi:
            return self._step_length * np.cos(np.pi * (phase - self._swing_stance_ratio) / (2 * np.pi - self._swing_stance_ratio))

    def _upward_downward(self, phase):
        assert 0 <= phase <= 2 * np.pi, f"phase must be in the range [0, 2π), but got phase={phase}"

        if 0 <= phase < self._swing_stance_ratio:
            return -self._nominal_height + self._step_height * np.sin(np.pi * phase / self._swing_stance_ratio)
        elif self._swing_stance_ratio <= phase < 2*np.pi:
            return -self._nominal_height

    def compute(self, t):
        phase = self._mod_2_pi(self._mod_2_pi(2*np.pi*self._frequency*t) + self.phase_offset)
        x_aligned = self._forward_backward(phase)

        assert -np.pi/2 <= self._direction <= np.pi/2, f"direction must be in the range [-π/2, π/2], but got self.direction={self._direction}"

        x = np.cos(self._direction) * x_aligned
        y = np.sin(self._direction) * x_aligned
        z = self._upward_downward(phase)
        return (x, y, z)

    @property
    def phase_offset(self): 
        return self._phase_offset 
    
    @property
    def frequency(self): 
        return self._frequency

    @property
    def step_length(self): 
        return self._step_length 
    
    @property
    def step_height(self): 
        return self._step_height 
    
    @property
    def nominal_height(self): 
        return self._nominal_height
    
    @property
    def swing_stance_ratio(self): 
        return self._swing_stance_ratio
    
    @property
    def direction(self): 
        return self._direction
    
    @property
    def adjustable_params(self): 
        return (self.frequency, self.step_length, self.step_height, self.nominal_height, self.swing_stance_ratio, self.direction)
    
    @phase_offset.setter 
    def phase_offset(self, phase_offset): 
        self._phase_offset = np.clip(phase_offset, 0, 2*np.pi)

    @frequency.setter 
    def frequency(self, frequency): 
        self._frequency = frequency

    @step_length.setter 
    def step_length(self, step_length): 
        self._step_length = step_length

    @step_height.setter 
    def step_height(self, step_height): 
        self._step_height = step_height

    @nominal_height.setter 
    def nominal_height(self, nominal_height): 
        self._nominal_height = nominal_height

    @swing_stance_ratio.setter 
    def swing_stance_ratio(self, swing_stance_ratio): 
        self._swing_stance_ratio = np.clip(swing_stance_ratio, 0, 2*np.pi)
    
    @direction.setter 
    def direction(self, direction): 
        self._direction = np.clip(direction, -np.pi/2, np.pi/2)

    @adjustable_params.setter 
    def adjustable_params(self, adjustable_params): 
        self._frequency = adjustable_params[0]
        self._step_length = adjustable_params[1]
        self._step_height = adjustable_params[2]
        self._nominal_height = adjustable_params[3]
        self._direction = adjustable_params[4]
