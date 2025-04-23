import imprl.envs
from imprl.post_process.inference import AgentInference
import quake_envs
import gymnasium as gym

env = gym.make("quake-res-30-v1").unwrapped

algorithm = "VDN_PS"
TrainedAgent = AgentInference(algorithm, env) # initialize agent
# # Load model
checkpt_dir = r"C:\Users\Anton\OneDrive\Desktop\tudelft_thesis\qres_marl\wandb\run-20250422_080051-014gqury\files"
ep = 12000 # episode number
TrainedAgent.load_weights(checkpt_dir, ep)

# TrainedAgent.plot_rollout(figsize=(10,10)) # each call will plot a new rollout

TrainedAgent.run()