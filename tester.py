
import FloatSatEnv as FloatSatEnv
from stable_baselines3 import SAC

from Basilisk.utilities import RigidBodyKinematics as rbk

import numpy as np

# Load the trained model
model = SAC.load("satellite_attitude_control")

# Test the trained agent
env = FloatSatEnv.SatelliteEnv()
obs = env.reset()[0]
env.config(target_deg=0.5, target_angular_velocity=0.8, bonus_reward=200)
done = False

target = None

time = []
angle = []
actions = []
angular_velocity = []
rw_speed = []
last_error = 0
last_error2 = 0
last_action = 0
d_angular_velocity = []
while not done:
    last_error = obs[0] #* 0.2 + last_error * 0.8
    last_error2 = obs[1] #* 0.2 + last_error2 * 0.8
    action, _states = model.predict(obs)
    # action = action * 0.3 + last_action * 0.7
    actions.append(action)
    last_action = action
    obs, reward, done, truncated, info = env.step(action)
    done = done or truncated
    C_BN = rbk.MRP2C(info["sigma_BN[deg]"])
    # Extract roll, pitch, yaw (Euler angles) from the DCM
    yaw = np.arctan2(C_BN[1, 0], C_BN[0, 0])  # Psi
    pitch = np.arcsin(-C_BN[2, 0])            # Theta
    roll = np.arctan2(C_BN[2, 1], C_BN[2, 2]) # Phi
    angle.append(np.degrees(yaw))
    d_angular_velocity.append(obs[2])
    time.append(info["time[s]"])
    if info["time[s]"] < 60:
        done = False
    rw_speed.append(info["RW_speed[RPM]"])

    if target is None:
        target = info["target[deg]"]

    if done:
        break

#plot attitude
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

# First subplot: Yaw and Target
axes[0].plot(time, angle, label="Yaw [deg]")
axes[0].plot(time, [target] * len(time), label="Target [deg]", linestyle="dashed")
axes[0].set_ylabel("Yaw (degrees)")
axes[0].set_title("Yaw vs Time")
axes[0].legend()
axes[0].grid(True)

# Second subplot: Action
axes[1].plot(time, actions, label="Action [Nm]", color="r")
axes[1].set_ylabel("Torque (Nm)")
axes[1].set_title("Action vs Time")
axes[1].legend()
axes[1].grid(True)

# Third subplot: RW Speed
axes[2].plot(time, rw_speed, label="RW Speed [rad/s]", color="g")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("RW Speed (rad/s)")
axes[2].set_title("Reaction Wheel Speed vs Time")
axes[2].legend()
axes[2].grid(True)


# d_angular_velocity
axes[3].plot(time, d_angular_velocity, label="Angular Velocity [rad/s/s]", color="b")
axes[3].set_ylabel("Difference of Angular Velocity (rad/s)")
axes[3].set_title("Difference of Angular Velocity vs Time")
axes[3].legend()
axes[3].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure
plt.show()
