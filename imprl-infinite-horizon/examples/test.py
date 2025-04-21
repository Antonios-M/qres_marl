import quake_envs
import gymnasium as gym
import imprl.envs
from imprl.post_process.inference import AgentInference
env = gym.make("quake-res-30-v1").unwrapped
algorithm = "VDN_PS"
TrainedAgent = AgentInference(algorithm, env) # initialize agent

TrainedAgent.plot_rollout(figsize=(5,5))