
import FloatSatEnv as FloatSatEnv
from stable_baselines3 import SAC

from Basilisk.utilities import RigidBodyKinematics as rbk

import numpy as np

# Load the trained model
model = SAC.load("satellite_attitude_control")

# Test the trained agent
env = FloatSatEnv.SatelliteEnv()
obs = env.reset()
done = False

target = None

time = []
angle = []
while not done:
    #TODO get observation from UART
    action, _states = model.predict(obs)
    # TODO send action to UART
    obs, reward, done, info = env.step(action)

    C_BN = rbk.MRP2C(info["sigma_BN"])
    # Extract roll, pitch, yaw (Euler angles) from the DCM
    yaw = np.arctan2(C_BN[1, 0], C_BN[0, 0])  # Psi
    pitch = np.arcsin(-C_BN[2, 0])            # Theta
    roll = np.arctan2(C_BN[2, 1], C_BN[2, 2]) # Phi
    angle.append(np.degrees(yaw))
    time.append(info["time"])

    if target is None:
        target = info["target"]

    if done:
        break

#plot attitude
import matplotlib.pyplot as plt
plt.plot(time, angle)
plt.plot(time, [target]*len(time))
plt.xlabel("Time (s)")
plt.ylabel("Yaw (degrees)")
plt.title("Yaw vs Time")
plt.show()


