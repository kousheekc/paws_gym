import numpy as np
import matplotlib.pyplot as plt

# 1. Foot trajectory generator function based on phi_i (phase of the gait cycle)
def foot_trajectory(phi, h):
    # Piecewise function for the foot trajectory
    if 0 <= phi < np.pi:
        return h * np.sin(phi)
    elif np.pi <= phi < 2*np.pi:
        return 0

# 2. Parameters for the trajectory
h = 0.1  # Maximum foot height
phi_values = np.linspace(0, 2 * np.pi, 500)  # Phase values from 0 to 2π

# 3. Compute the foot trajectory for all phase values
foot_positions_z = [foot_trajectory(phi, h) for phi in phi_values]

# 4. Plot the foot trajectory
plt.figure()
plt.plot(phi_values, foot_positions_z)
plt.xlabel("Phase (phi_i) [radians]")
plt.ylabel("Foot z-position [m]")
plt.title("Foot Trajectory (Vertical Position) Over Phase")
plt.grid(True)
plt.show()
