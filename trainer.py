
# import sys
# sys.path.append("/home/nicki/GDrive/Uni-Wuerzburg/2024WS/FloatSat/basilisk/dist3")


import FloatSatEnv
from stable_baselines3 import SAC

# Instantiate the environment
env = FloatSatEnv.SatelliteEnv()
env.reset()
# Train RL agent using PPO
a = input("Create new?")
if a == "y":
    print("Creating new model")
    model = SAC(
        "MlpPolicy",  
        env,          
        verbose=1,    
        learning_rate=5e-4,  # Adjusted for stable training
        gamma=0.99,          # Discount factor
        tau=0.005,           # Soft update coefficient
        batch_size=256,      # Larger batch size for better updates
        buffer_size=1000000, # Replay buffer size
        train_freq=1,        # Train every timestep
        gradient_steps=1,    # Gradient steps per training step
        policy_kwargs=dict(
            net_arch=[256, 256],  # Keep the architecture simple and effective
        ),
        ent_coef="auto",        # Automatic entropy coefficient tuning
        target_entropy=-2,  # Target entropy tuning
    )
else:
    print("Loading existing model")
    model = SAC.load("satellite_attitude_control")
    model.set_env(env)

model.learn(total_timesteps=5000)

# Save the trained model
model.save("satellite_attitude_control")

# done = False
# while not done:
#     step = input("Enter the step: ")
#     # print([step])
#     obs, reward, done, _ = env.step([float(step)])
#     # print(reward)
#     # print(obs)


# # env.step([-1])
# # env.step([-1])
# # env.step([-1])
# # env.step([-1])
# # env.step([-1])
# # env.step([-1])
# # env.step([-1])
# # env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# env.step([-1])
# # env.reset()
# env.step([0])
# env.step([0])
# env.step([0])
# env.step([0])
# env.step([0])
# env.step([0])
# env.step([0])
# env.step([0])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])
# env.step([1])

# env.step([0])
# env.step([0])
# env.step([0])
