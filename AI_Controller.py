
from stable_baselines3 import SAC
import numpy as np

import os
dirname = os.path.dirname(__file__)
modelname = os.path.join(dirname, 'satellite_attitude_control')

# Load the trained model
model = SAC.load(modelname, costum_objects={"lr_scheduler": None})
action_buffer = [0., 0., 0., 0., 0., 0.]
def get_control(yaw, target, angular_velocity):
    attitude_error = yaw - target
    if attitude_error < -180:
        attitude_error = 360 + attitude_error 
    if attitude_error > 180:
        attitude_error = -(360 - attitude_error)

    action, _states = model.predict([attitude_error, angular_velocity,  np.sum(action_buffer)])
    action_buffer.pop(0)
    action_buffer.append(action[0])
    print(action_buffer)
    return action


