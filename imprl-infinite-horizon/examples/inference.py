import imprl.envs
from imprl.post_process.inference import AgentInference
import quake_envs
import gymnasium as gym

env = gym.make("quake-res-4-v1").unwrapped

algorithm = "DCMAC"
TrainedAgent = AgentInference(algorithm, env) # initialize agent
# # Load model
checkpt_dir = r"C:\Users\Anton\OneDrive\Desktop\tudelft_thesis\qres_marl\wandb\run-20250516_170603-4wdjnq32\files"
ep = 5000 # episode number
TrainedAgent.load_weights(checkpt_dir, ep)

TrainedAgent.plot_3d(figsize=(10,10)) # each call will plot a new rollout
# TrainedAgent.plot_rollout(figsize=(10,10), plot_econ_relocation=True, plot_econ_traffic=True, plot_delay=True) # each call will plot a new rollout
# TrainedAgent.plot_3d(figsize=(10,10), n=100)
TrainedAgent.get_rollout()
TrainedAgent.plot_components(figsize=(10,10))

# TrainedAgent.run()