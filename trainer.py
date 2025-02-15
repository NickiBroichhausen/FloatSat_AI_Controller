

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
    env.config(target_deg=5, target_angular_velocity=5, bonus_reward=2000)
    model = SAC(
        "MlpPolicy",  
        env,          
        verbose=1,    
        learning_rate=5e-4,  # Adjusted for stable training
        gamma=0.99,          # Discount factor
        tau=0.005,           # Soft update coefficient
        batch_size=256,      # Larger batch size for better updates
        buffer_size=100000, # Replay buffer size
        train_freq=1,        # Train every timestep
        gradient_steps=1,    # Gradient steps per training step
        policy_kwargs=dict(
            net_arch=[32, 32],  # Keep the architecture simple and effective
        ),
        ent_coef="0.2",        # Automatic entropy coefficient tuning
        # target_entropy=-2,  # Target entropy tuning
    )
    model.learn(total_timesteps=4000)
    time.sleep(1)
    env.reset()
    env.config(target_deg=1, target_angular_velocity=1, bonus_reward=200)
    model.learn(total_timesteps=2000)
    time.sleep(1)
    env.reset()
    env.config(target_deg=0.5, target_angular_velocity=0.8, bonus_reward=200)
    model.learn(total_timesteps=2000)
else:
    print("Loading existing model")
    model = SAC.load("satellite_attitude_control")
    model.set_env(env)
    time.sleep(1)
    env.reset()
    env.config(target_deg=0.5, target_angular_velocity=0.8, bonus_reward=200)
    model.learn(total_timesteps=2000)

# Save the trained model
model.save("satellite_attitude_control")

