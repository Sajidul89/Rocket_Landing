import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

# Load our existing AI instead of creating a new one.
model = PPO.load("lander_ppo", env=env, device="cpu")

# Practice for roughly 900,000 additional steps.
model.learn(total_timesteps=900_000, reset_num_timesteps=False)

# Save separately to keep our first model for comparison.
model.save("lander_ppo_1m")

env.close()
print("Finished! Saved as lander_ppo_1m.zip")