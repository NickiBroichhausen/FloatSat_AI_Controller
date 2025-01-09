
from stable_baselines3 import SAC
import numpy as np

import serial

# Load the trained model
model = SAC.load("satellite_attitude_control")

# Initialize UART
ser = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

# time = []
# angle = []
done = False
while not done:
    #TODO get observation from UART, maybe angular velocity, target and yaw?
    # test: sudo apt install minicom
    # test: minicom -b 9600 -o -D /dev/serial0
    data = ser.readline()
    print(data.decode('utf-8'))
    yaw = data[0]  # TODO maybe subtract 180 degrees, or convert from radians to degrees 
    target = data[1]
    angular_velocity = data[2]

    attitude_error = np.degrees(yaw) - target
    if attitude_error < -180:
        attitude_error = 360 + attitude_error 
    if attitude_error > 180:
        attitude_error = -(360 - attitude_error)

    action, _states = model.predict([attitude_error, angular_velocity])

    # TODO send action to UART
    ser.write('action\n')

    # angle.append(np.degrees(yaw))
    # time.append(info["time"])

# #plot attitude
# import matplotlib.pyplot as plt
# plt.plot(time, angle)
# plt.plot(time, [target]*len(time))
# plt.xlabel("Time (s)")
# plt.ylabel("Yaw (degrees)")
# plt.title("Yaw vs Time")
# plt.show()


ser.close()


