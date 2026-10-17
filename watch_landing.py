import gymnasium as gym
from stable_baselines3 import PPO

# Load the AI I already trained.
model = PPO.load("lander_ppo_1m", device="cpu")

# Show the game window.
env = gym.make("LunarLander-v3", render_mode="human")

try:
    for episode in range(1, 21):
        observation, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # The trained AI chooses an action.
            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = env.step(
                int(action)
            )

            total_reward += reward
            done = terminated or truncated

        print(f"Attempt {episode}: score = {total_reward:.2f}")

finally:
    env.close()