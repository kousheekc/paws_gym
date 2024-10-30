import numpy as np

class Motor:
    def __init__(self, kp, max_velocity):
        self.kp = kp
        self.max_velocity = max_velocity

    def calculate_velocity_command(self, theta_target, theta_current):
        velocity_command = self.kp * (theta_target - theta_current)
        velocity_command = np.clip(velocity_command, -self.max_velocity, self.max_velocity)
        return velocity_command
