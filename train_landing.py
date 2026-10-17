import gymnasium as gym
from stable_baselines3 import PPO

# Create the game without displaying a window.
env = gym.make("LunarLander-v3")

# Create a new AI using the PPO learning algorithm.
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu",
    seed=42
)

# Let the AI practice for approximately 100,000 game steps.
model.learn(total_timesteps=100_000)

# Save what it learned.
model.save("lander_ppo")

env.close()

print("Training finished! Saved as lander_ppo.zip")