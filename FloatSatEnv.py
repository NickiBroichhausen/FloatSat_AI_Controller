
import gym
from gym import spaces

import numpy as np

from Basilisk import __path__
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import (mrpFeedback, attTrackingError,
                                    inertial3D, rwMotorTorque, rwMotorVoltage)
from Basilisk.simulation import reactionWheelStateEffector, motorVoltageInterface, simpleNav, spacecraft
from Basilisk.utilities import (SimulationBaseClass, fswSetupRW, macros,
                                orbitalMotion, simIncludeGravBody,
                                simIncludeRW, unitTestSupport, vizSupport)
from Basilisk.utilities import RigidBodyKinematics as rbk

import time

show_plots = False
useJitterSimple = True
useRWVoltageIO = False


# Custom OpenAI Gym-like Environment for Basilisk
class SatelliteEnv(gym.Env):
    def __init__(self):

        self.target = np.random.uniform(-180, 180)
        # self.target = -176

        # Observation space (pitch,roll,yaw,angular_velocities)
        self.observation_space = spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32)
        
        # Action space (yaw torque)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


        # Create simulation variable names
        simTaskName = "simTask"
        simProcessName = "simProcess"

        #  Create a sim module as an empty container
        self.scSim = SimulationBaseClass.SimBaseClass()

        # set the simulation time variable used later on
        simulationTime = macros.sec2nano(1.)

        #
        #  create the simulation process
        #
        dynProcess = self.scSim.CreateNewProcess(simProcessName)

        # create the dynamics task and specify the integration update time
        self.simulationTimeStep = macros.sec2nano(0.1)   # simulation becomes unstable for higher values. idk why seems to be problem with basilisks RW
        dynProcess.addTask(self.scSim.CreateNewTask(simTaskName, self.simulationTimeStep))

        #
        #   setup the simulation tasks/objects
        #

        # initialize spacecraft object and set properties
        self.scObject = spacecraft.Spacecraft()
        self.scObject.ModelTag = "bsk-Sat"
        # # define the simulation inertia
        # I = [1., 0.04, -0.02,
        #     -0.02, 0.02, -1.,
        #     -0.04, 1., 0.02]
        # I = [0.02, -1.  , -0.02,
        #     -1.  ,  0.02,  0.04,
        #     -0.02,  0.04,  1.  ]
        I = [1., 0., -0.,       # TODO wtf????
            -0., 1., -0.,
            -0., 0., 1.]
        self.scObject.hub.mHub = 1.703  # kg - spacecraft mass
        self.scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # m - position vector of body-fixed point B relative to CM
        self.scObject.hub.IHubPntBc_B = unitTestSupport.np2EigenMatrix3d(I)

        # add spacecraft object to the simulation process
        self.scSim.AddModelToTask(simTaskName, self.scObject, 1)

        #
        # add RW devices
        #
        # Make a fresh RW factory instance, this is critical to run multiple times
        rwFactory = simIncludeRW.rwFactory()

        # store the RW dynamical model type
        varRWModel = messaging.BalancedWheels
        if useJitterSimple:
            varRWModel = messaging.JitterSimple

        # create each RW by specifying the RW type, the spin axis gsHat, plus optional arguments
        RW1 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=20.  # RPM
                            , RWModel=varRWModel
                            )
        # RW2 = rwFactory.create('Honeywell_HR16', [0, 1, 0], maxMomentum=50., Omega=200.  # RPM
        #                     , RWModel=varRWModel
        #                     )
        # RW3 = rwFactory.create('Honeywell_HR16', [0, 0, 1], maxMomentum=50., Omega=300.  # RPM
        #                     , rWB_B=[0.5, 0.5, 0.5]  # meters
        #                     , RWModel=varRWModel
        #                     )
        # In this simulation the RW objects RW1, RW2 or RW3 are not modified further.  However, you can over-ride
        # any values generate in the `.create()` process using for example RW1.Omega_max = 100. to change the
        # maximum wheel speed.

        numRW = rwFactory.getNumOfDevices()

        # create RW object container and tie to spacecraft object
        self.rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        self.rwStateEffector.ModelTag = "RW_cluster"
        rwFactory.addToSpacecraft(self.scObject.ModelTag, self.rwStateEffector, self.scObject)

        # add RW object array to the simulation process.  This is required for the UpdateState() method
        # to be called which logs the RW states
        self.scSim.AddModelToTask(simTaskName, self.rwStateEffector, 2)

        # if useRWVoltageIO:
        #     rwVoltageIO = motorVoltageInterface.MotorVoltageInterface()
        #     rwVoltageIO.ModelTag = "rwVoltageInterface"

        #     # set module parameters(s)
        #     rwVoltageIO.setGains(np.array([0.2 / 10.] * 3)) # [Nm/V] conversion gain

        #     # Add test module to runtime call list
        #     scSim.AddModelToTask(simTaskName, rwVoltageIO)

        # add the simple Navigation sensor module.  This sets the SC attitude, rate, position
        # velocity navigation message
        sNavObject = simpleNav.SimpleNav()
        sNavObject.ModelTag = "SimpleNavigation"
        self.scSim.AddModelToTask(simTaskName, sNavObject)

        
        #
        #   setup the FSW algorithm tasks
        #

        # setup inertial3D guidance module
        # inertial3DObj = inertial3D.inertial3D()
        # inertial3DObj.ModelTag = "inertial3D"
        # self.scSim.AddModelToTask(simTaskName, inertial3DObj)
        # inertial3DObj.sigma_R0N = [0., 0., 0.]  # set the desired inertial orientation

        # setup the attitude tracking error evaluation module
        # self.attError = attTrackingError.attTrackingError()
        # self.attError.ModelTag = "attErrorInertial3D"
        # self.scSim.AddModelToTask(simTaskName, self.attError)

            
        # add module that maps the Lr control torque into the RW motor torques
        # rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
        # rwMotorTorqueObj.ModelTag = "rwMotorTorque"
        # scSim.AddModelToTask(simTaskName, rwMotorTorqueObj)

        # # Make the RW control all three body axes
        # controlAxes_B = [
        #     1, 0, 0, 0, 1, 0, 0, 0, 1
        # ]
        # rwMotorTorqueObj.controlAxes_B = controlAxes_B

        # if useRWVoltageIO:
        #     fswRWVoltage = rwMotorVoltage.rwMotorVoltage()
        #     fswRWVoltage.ModelTag = "rwMotorVoltage"

        #     # Add test module to runtime call list
        #     scSim.AddModelToTask(simTaskName, fswRWVoltage)

        #     # set module parameters
        #     fswRWVoltage.VMin = 0.0  # Volts
        #     fswRWVoltage.VMax = 10.0  # Volts

        #
        #   Setup data logging before the simulation is initialized
        #
        numDataPoints = 100
        samplingTime = unitTestSupport.samplingTime(simulationTime, self.simulationTimeStep, numDataPoints)
        # rwMotorLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
        # attErrorLog = self.attError.attGuidOutMsg.recorder(samplingTime)
        self.snTransLog = sNavObject.transOutMsg.recorder(samplingTime)
        self.snAttLog = sNavObject.attOutMsg.recorder(samplingTime)
        # scSim.AddModelToTask(simTaskName, rwMotorLog)
        # self.scSim.AddModelToTask(simTaskName, attErrorLog)
        self.scSim.AddModelToTask(simTaskName, self.snTransLog)
        self.scSim.AddModelToTask(simTaskName, self.snAttLog)

        # To log the RW information, the following code is used:
        self.mrpLog = self.rwStateEffector.rwSpeedOutMsg.recorder(samplingTime)
        self.scSim.AddModelToTask(simTaskName, self.mrpLog)

        # A message is created that stores an array of the \f$\Omega\f$ wheel speeds.  This is logged
        # here to be plotted later on.  However, RW specific messages are also being created which
        # contain a wealth of information.  The vector of messages is ordered as they were added.  This
        # allows us to log RW specific information such as the actual RW motor torque being applied.
        self.rwLogs = []
        for item in range(numRW):
            self.rwLogs.append(self.rwStateEffector.rwOutMsgs[item].recorder(samplingTime))
            self.scSim.AddModelToTask(simTaskName, self.rwLogs[item])
        if useRWVoltageIO:
            rwVoltLog = fswRWVoltage.voltageOutMsg.recorder(samplingTime)
            self.scSim.AddModelToTask(simTaskName, rwVoltLog)

        #
        # create simulation messages
        #

        # create the FSW vehicle configuration message
        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = I  # use the same inertia in the FSW algorithm as in the simulation
        vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)

        # Two options are shown to setup the FSW RW configuration message.
        # First case: The FSW RW configuration message
        # uses the same RW states in the FSW algorithm as in the simulation.  In the following code
        # the fswSetupRW helper functions are used to individually add the RW states.  The benefit of this
        # method of the second method below is that it is easy to vary the FSW parameters slightly from the
        # simulation parameters.  In this script the second method is used, while the fist method is included
        # to show both options.
        fswSetupRW.clearSetup()
        for key, rw in rwFactory.rwList.items():
            fswSetupRW.create(unitTestSupport.EigenVector3d2np(rw.gsHat_B), rw.Js, 0.2)
        fswRwParamMsg1 = fswSetupRW.writeConfigMessage()

        # Second case: If the exact same RW configuration states are to be used by the simulation and fsw, then the
        # following helper function is convenient to extract the fsw RW configuration message from the
        # rwFactory setup earlier.
        fswRwParamMsg2 = rwFactory.getConfigMessage()
        fswRwParamMsg = fswRwParamMsg2

        
        #
        #   set initial Spacecraft States
        #
        # setup the orbit using classical orbit elements
        # oe = orbitalMotion.ClassicElements()
        # oe.a = 10000000.0  # meters
        # oe.e = 0.01
        # oe.i = 33.3 * macros.D2R
        # oe.Omega = 48.2 * macros.D2R
        # oe.omega = 347.8 * macros.D2R
        # oe.f = 85.3 * macros.D2R
        # rN, vN = orbitalMotion.elem2rv(mu, oe)
        # scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
        # scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
        # self.scObject.hub.sigma_BNInit = [[0.0], [0.2], [-0.3]]  # sigma_CN_B     # TODO make same as in reset. call reset here
        # self.scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_CN_B

        # if this scenario is to interface with the BSK Viz, uncomment the following lines
        # viz = vizSupport.enableUnityVisualization(scSim, simTaskName, scObject
                                                # # , saveFile=fileName
                                                # , rwEffectorList=rwStateEffector
                                                # )
        # link messages
        sNavObject.scStateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        # self.attError.attNavInMsg.subscribeTo(sNavObject.attOutMsg)
        # self.attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
        # mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
        # mrpControl.vehConfigInMsg.subscribeTo(vcMsg)
        # mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)
        # mrpControl.rwSpeedsInMsg.subscribeTo(rwStateEffector.rwSpeedOutMsg)
        # rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
        # rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
        # if useRWVoltageIO:
        #     fswRWVoltage.torqueInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
        #     fswRWVoltage.rwParamsInMsg.subscribeTo(fswRwParamMsg)
        #     rwVoltageIO.motorVoltageInMsg.subscribeTo(fswRWVoltage.voltageOutMsg)
        #     rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwVoltageIO.motorTorqueOutMsg)
        # else:
        #     rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)


        self.msgData = messaging.ArrayMotorTorqueMsgPayload()
        self.msgData.motorTorque = [0]
        self.msg = messaging.ArrayMotorTorqueMsg()
        
        self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.msg)

        self.msg.write(self.msgData)

        # self.msg.write(self.msgData)
        # self.msgData.motorTorque = [-10]
        # self.msg.write(self.msgData)



        # Reset spacecraft state (position, velocity, attitude, etc.)
        # TODO do random initialization
        self.scObject.hub.r_CN_NInit = np.zeros(3)
        self.scObject.hub.v_CN_NInit = np.zeros(3)
        self.scObject.hub.sigma_BNInit =  [0,0, np.random.uniform(-180, 180, 1)] # np.zeros(3) #np.random.uniform(-0.1, 0.1, 3)  # Small random orientation
        self.scObject.hub.omega_BN_BInit = [0,0,np.random.uniform(-1, 1, 1)]  #np.zeros(3) #np.random.uniform(-0.01, 0.01, 3)  # Small random rotation rate

        self.last_action = 0
        self.i_angular_error = 0

        #
        #   initialize Simulation
        #
        self.scSim.InitializeSimulation()



    def reset(self):
        """Reset the simulation."""

        self.__init__()

        self.iteration = 0

        # Run one simulation step
        self.scSim.InitializeSimulation()
        # self.scSim.ExecuteSimulation(self.simulationTime)
        
        self.scSim.TotalSim.SingleStepProcesses()

        return self._get_observation()


    
    def step(self, action):
        """Apply action and advance simulation."""
        # Apply torque action
        # change input message and continue simulation
        # ArrayMotorTorqueMsgPayload

        # print("--------------")
        # print(f"action: {action}")
        # self.msgData = messaging.ArrayMotorTorqueMsgPayload()
        # self.msgData.motorTorque = [action[0] - self.last_action]
        self.msgData.motorTorque = [action[0]*10]   # TODO map this to the actual torque
        self.last_action = action[0]
        # self.msg = messaging.ArrayMotorTorqueMsg()
        self.msg.write(self.msgData)
        # self.rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.msg)

        # self.scSim.ConfigureStopTime(macros.sec2nano(20.0))
        # self.scSim.ExecuteSimulation()

        # set motor with radians/seconds from controller input  -> but there has to be a delay?

        # Step simulation
        self.scSim.TotalSim.SingleStepProcesses()

        # Get next observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs)

        # Define done condition
        # when yaw close to target 100 and angular velocity close to 0
        
        # print("\033[{0};{1}H{2}".format(30, 40, obs))
        # print(f"{obs}               ", end="\r")
        # print(f"reward: {reward}")
        # print(f"done: {abs(obs[0] - 100) < 1 and abs(obs[1]) < 0.01}")
        # # print wheel speed
        # print(f"wheel speed: {self.rwLogs[0].Omega[-1]}")
        good = abs(obs[0]) < 0.5 and abs(obs[1]) < 0.1  # TODO terminate with big plus here instead of positive reward??
        if good:
            reward = reward + 5
        done = False
        if self.iteration * self.simulationTimeStep >= macros.min2nano(1):
            done = True
            # done = abs(obs[0]) < 0.5 and abs(obs[1]) < 0.1
            # reward = -10000
            # print("#########################################################################################")
            # time.sleep(1)
            
        self.iteration += 1

        print(
            f"Step: {action[0]:+.2f}  | Obs:  {[f'{x:+3.2f}' for x in obs]} | Reward: {reward} | "
             f"Time (seconds): {self.iteration * self.simulationTimeStep / 1_000_000_000:.2f}"
        )


        # if done:
        #     sigma_BN = self.snAttLog.sigma_BN
        #     print(sigma_BN)
        #     # TODO change to smth like this: mod2.dataOutMsg.read().dataVector
        #     # sigma_BN = self.scObject.hub.sigma_BN
        #     omega_BN_B = self.snAttLog.omega_BN_B
        #     print(omega_BN_B)

        info = {}
        if True:
            info = {
                "sigma_BN": self.snAttLog.sigma_BN[-1],
                "time": self.iteration * self.simulationTimeStep / 1_000_000_000,
                "target": self.target
            }    
            
        return obs, reward, done, info
    

    def _get_observation(self):
        """Retrieve spacecraft state as observation."""
        # attitude = self.scObject.hub.sigma_BN
        # angular_velocity = self.scObject.hub.omega_BN_B
        # return np.concatenate([attitude, angular_velocity])
        # Get the current MRPs (sigma_BN) and angular velocities (omega_BN_B)
        # print(self.scObject.hub)
        # sigma_BN = self.attErrorLog.sigma_BR[-1]
        sigma_BN = self.snAttLog.sigma_BN[-1]
        # TODO change to smth like this: mod2.dataOutMsg.read().dataVector
        # sigma_BN = self.scObject.hub.sigma_BN
        omega_BN_B = self.snAttLog.omega_BN_B[-1]
        # print(omega_BN_B)

        # Convert MRPs to Direction Cosine Matrix (DCM)
        C_BN = rbk.MRP2C(sigma_BN)

        # Extract roll, pitch, yaw (Euler angles) from the DCM
        yaw = np.arctan2(C_BN[1, 0], C_BN[0, 0])  # Psi
        pitch = np.arcsin(-C_BN[2, 0])            # Theta
        roll = np.arctan2(C_BN[2, 1], C_BN[2, 2]) # Phi

        # print(f"roll: {np.degrees(roll)}")
        # print(f"pitch: {np.degrees(pitch)}")
        # print(f"yaw: {np.degrees(yaw)}")

        yaw_w = omega_BN_B[2]
        if abs(yaw_w) > 100:   # TODO check this 100
            yaw_w = np.copysign(100, yaw_w)

        attitude_error = np.degrees(yaw) - self.target
        if attitude_error < -180:
            attitude_error = 360 + attitude_error 
        if attitude_error > 180:
            attitude_error = -(360 - attitude_error)

        # Combine Roll, Pitch, Yaw and Angular Velocities
        state_vector = np.array([
            attitude_error,
            # target,
            # np.degrees(roll),  # Convert to degrees if needed
            # np.degrees(pitch),
            # np.degrees(yaw),
            # omega_BN_B[0],     # Angular velocity x-component
            # omega_BN_B[1],     # Angular velocity y-component, apperently pitch
            yaw_w      # Angular velocity z-component
        ])

        return state_vector


    def _compute_reward(self, obs):
        """Calculate the reward based on observation."""
        # attitude_error = np.linalg.norm(obs[:3] - self.target_quaternion[:3])
        # angular_velocity_error = np.linalg.norm(obs[3:])
        # return - (10 * attitude_error + angular_velocity_error)  # Penalize errors

        # print(obs)

        # TODO -170 is close to 170, how to account for flip at 180?
        # attitude_error = -abs(obs[1] - target)
        # if attitude_error < -180:
        #     attitude_error = -(360 + attitude_error)

        angular_velocity_error = abs(obs[1]) if abs(obs[1]) > 0.1 else 0  # clipping angular velocity error 
                                                                            # as this introduces steady state error

        # print(f"error: {attitude_error}")

        # TODO add integral parameter
        # TODO or abs(angular_error) if angular_velocity_error < 0.1 else 0)
        # steady_state_error = 0 if abs(angular_velocity_error) < 0.1 else -abs(obs[0])  # shouldnt this be when angel < 0.1???
        
        # self.i_angular_error = self.i_angular_error + abs(obs[0]) * 0.01
        
        return  -np.sqrt(abs(obs[0])) - angular_velocity_error #- self.i_angular_error  # Penalize errors
