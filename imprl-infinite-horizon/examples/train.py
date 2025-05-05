import time
import torch
import imprl.agents
import imprl.envs
from imprl.runners.serial import training_rollout
from imprl.agents.configs.get_config import load_config
import sys
import gymnasium as gym
import quake_envs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

run_qres = "y"
ALGORITHM = "VDN_PS"
SINGLE_AGENT = False

if run_qres == "y":
    # Environment
    env = gym.make("quake-res-4-v1").unwrapped
    # env = gym.make("ma-grid-world-v0").unwrapped

else:
    ENV_NAME = "k_out_of_n_infinite"
    ENV_SETTING = "hard-4-of-4_infinite"
    ENV_KWARGS = {"percept_type": "belief", "reward_shaping": True}
    env = imprl.envs.make(ENV_NAME, ENV_SETTING, single_agent=SINGLE_AGENT, **ENV_KWARGS)


alg_config = load_config(algorithm=ALGORITHM)  # load default config
agent_class = imprl.agents.get_agent_class(ALGORITHM)
LearningAgent = agent_class(env, alg_config, device)  # initialize agent
print(f"Loaded default configuration for {ALGORITHM}.")

time0 = time.time()

# training loop
for ep in range(100):
    print(f"--------------------------------------------------")
    episode_return = training_rollout(env, LearningAgent)
    LearningAgent.report()
    print(f"--------------------------------------------------")

print(f"Total time: {time.time()-time0:.2f}")
