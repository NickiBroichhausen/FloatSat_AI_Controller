

import FloatSatEnv
from stable_baselines3 import SAC
import time
# Instantiate the environment
env = FloatSatEnv.SatelliteEnv()
env.reset()
# Train RL agent using PPO
a = input("Create new?")
if a == "y":
    print("Creating new model")
    #easier to learn only rough goal, optimize in later trainings
    env.config(target_deg=10, target_torque=70)
    model = SAC(
        "MlpPolicy",
        env,
        # # learning_rate=3e-5,  # Lower LR for stability
        buffer_size=10000,  # Large buffer to avoid forgetting
        # learning_starts=1000,  # Ensure some exploration before updates
        batch_size=64,  # Small enough for stability
        # tau=0.005,  # Soft update coefficient for target Q
        # gamma=0.99,  # Discount factor
        # train_freq=1,  # Update every step
        # gradient_steps=1,  # One gradient step per env step
        # ent_coef="auto",  # Automatic entropy tuning
        # target_update_interval=1,  # Smooth updates
        # policy_kwargs=dict(
        #     net_arch=[64, 64],  # Small network since the state space is small
        #     # activation_fn="relu",
        # ),
        # verbose=1,  # Set to 0 for no logging
    )
    model.learn(total_timesteps=5000)
    time.sleep(1)
    env.reset()
    env.config(target_torque=70, target_deg=200)
    model.learn(total_timesteps=5000)
    time.sleep(1)
    env.reset()
    env.config(target_angular_velocity=0, target_deg=200)
    model.learn(total_timesteps=2000)
else:
    print("Loading existing model")
    model = SAC.load("satellite_attitude_control")
    model.set_env(env)
    time.sleep(1)
    env.reset()
    env.config(target_deg=400) # set parameters here for fine tuning
    model.learn(total_timesteps=10000)

# Save the trained model
model.save("satellite_attitude_control")

