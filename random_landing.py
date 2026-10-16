import gymnasium as gym

# Create the game and show its window.
env = gym.make("LunarLander-v3", render_mode="human")

# Start a new landing attempt.
observation, info = env.reset()

episode_reward = 0
episode_number = 1

# Run for up to 10 game steps.
while episode_number <= 10:
    # Randomly choose an action.
    action = env.action_space.sample()

    # Perform the action and get the result.
    observation, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward
    # Restart if the attempt ends.
    if terminated or truncated:
                
     print(f"Attempt {episode_number}: score = {episode_reward:.2f}")

     episode_reward = 0
     episode_number += 1
     observation, info = env.reset()
# Close the game window.
env.close()