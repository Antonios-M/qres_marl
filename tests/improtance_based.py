import quake_envs_pkg.quake_envs as quake_envs
import gymnasium as gym
import numpy as np

import matplotlib.pyplot as plt

env = gym.make("quake-res-30-v1")
env = env.unwrapped

obs, info = env.reset()
returns = []

for i in range(300):
    buildings = env.buildings_objs
    roads = env.road_objs
    n_crews = env.n_crews

    # Get building and road indices sorted by value (descending priority)
    building_indices = sorted(range(len(buildings)), key=lambda i: -buildings[i].value)
    road_indices = sorted(range(len(roads)), key=lambda i: -roads[i].value)

    action = [0] * 30  # Initialize all actions to "do nothing"

    crews_used = 0

    # Assign actions to buildings (first 15 slots)
    for idx in building_indices:
        if crews_used >= n_crews:
            break
        if idx >= 15:
            continue  # Skip if more buildings than action slots
        if buildings[idx].has_debris:
            action[idx] = 1  # clear debris
        else:
            action[idx] = 2  # repair
        crews_used += 1

    # Assign actions to roads (last 15 slots)
    for idx in road_indices:
        if crews_used >= n_crews:
            break
        if idx >= 15:
            continue  # Skip if more roads than action slots
        action[15 + idx] = 1  # repair
        crews_used += 1

    obs, reward, done, term, info = env.step(tuple(action))
    returns.append(reward)

    if done or term:
        obs, info = env.reset()

print("Total return:", sum(returns))

# Plot the rewards
plt.figure(figsize=(10, 5))
plt.plot(returns, label="Reward per step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
