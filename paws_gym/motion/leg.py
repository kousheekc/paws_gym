import numpy as np

class Leg:
    def __init__(self, leg, l1, l2, l3):
        assert leg in ['fl', 'fr', 'bl', 'br'], f"{leg} must be one of the following: [fl, fr, bl, br]"

        self._leg = leg
        self._l1 = l1
        self._l2 = l2
        self._l3 = l3

    def compute(self, point):        
        (x, y, z) = point

        if (self._leg == "bl" or self._leg == "br"):
            x = -x
            y = -y

        if (self._leg == "bl" or self._leg == "fr"):
            theta1 = -np.arctan2(z, y) - np.arctan2(np.sqrt(y**2 + z**2 - self._l1**2), -self._l1)
            d = (y**2 + z**2 - self._l1**2 + x**2 - self._l2**2 - self._l3**2)/(2 * self._l2 * self._l3)
            theta3 = np.arctan2(-np.sqrt(1 - d**2), d)
            theta2 = np.arctan2(-x, np.sqrt(y**2 + z**2 - self._l1**2)) - np.arctan2(self._l3 * np.sin(theta3), self._l2 + self._l3 * np.cos(theta3))

        elif (self._leg == "fl" or self._leg == "br"):
            theta1 = -np.arctan2(z, y) - np.arctan2(np.sqrt(y**2 + z**2 - self._l1**2), self._l1)
            d = (y**2 + z**2 - self._l1**2 + x**2 - self._l2**2 - self._l3**2)/(2 * self._l2 * self._l3)
            theta3 = np.arctan2(np.sqrt(1 - d**2), d)
            theta2 = np.arctan2(x, np.sqrt(y**2 + z**2 - self._l1**2)) - np.arctan2(self._l3 * np.sin(theta3), self._l2 + self._l3 * np.cos(theta3))

        return [theta1, theta2, theta3]