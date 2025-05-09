import quake_envs
import gymnasium as gym

env4 = gym.make("quake-res-4-v1").unwrapped
env30 = gym.make("quake-res-30-v1").unwrapped

env30.resilience.simulation.viz_environment("Toy test bed - 4 components", show_traffic_ids=True, figsize=(12,12))