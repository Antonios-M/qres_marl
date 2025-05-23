import time
import math
import concurrent.futures

import torch
import wandb
import numpy as np
import quake_envs
import imprl.agents
import imprl.envs
from imprl.runners.serial import training_rollout, evaluate_agent
from imprl.agents.configs.get_config import load_config
import gymnasium as gym
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = "k_out_of_n_infinite"
ENV_SETTING = "hard-1-of-4_infinite"
ENV_KWARGS = {"percept_type": "belief"}

run_qres = "y"
ALGORITHM = "DDMAC"
SINGLE_AGENT = False

if run_qres == "y":
    # Environment
    env = gym.make("quake-res-4-v1").unwrapped
    inference_env = gym.make("quake-res-4-v1").unwrapped

else:
    ENV_NAME = "k_out_of_n_infinite"
    ENV_SETTING = "hard-4-of-4_infinite"
    ENV_KWARGS = {"percept_type": "belief", "reward_shaping": True}
    env = imprl.envs.make(ENV_NAME, ENV_SETTING, single_agent=SINGLE_AGENT, **ENV_KWARGS)
    inference_env = imprl.envs.make(
        ENV_NAME,
        ENV_SETTING,
        single_agent=SINGLE_AGENT,
        **ENV_KWARGS,
    )

# Agent
alg_config = load_config(algorithm=ALGORITHM)  # load default config
agent_class = imprl.agents.get_agent_class(ALGORITHM)
LearningAgent = agent_class(env, alg_config, device)  # initialize agent
print(f"Loaded default configuration for {ALGORITHM}.")

PROJECT = "main-DDMAC-4-comp-v1"
ENTITY = "antoniosmavrotas-tu-delft"
WANDB_DIR = "./experiments/data"
# WANDB_DIR = "/scratch/pbhustali"

LOGGING_FREQUENCY = 100
CHECKPT_FREQUENCY = 5_000
INFERENCING_FREQUENCY = 5_000
NUM_INFERENCE_EPISODES = 500


def parallel_rollout(args):
    checkpt_dir, ep = args
    agent = agent_class(inference_env, alg_config, device)
    agent.load_weights(checkpt_dir, ep)
    return evaluate_agent(inference_env, agent)


if __name__ == "__main__":

    run = wandb.init(project=PROJECT, entity=ENTITY)

    # logging and checkpointing
    training_log = {}  # log for training metrics

    best_reward = -math.inf  # Initialize to negative infinity
    best_checkpt = 0
    is_time_to_checkpoint = (
        lambda ep: ep % CHECKPT_FREQUENCY == 0 or ep == wandb.config.NUM_EPISODES - 1
    )
    is_time_to_log = (
        lambda ep: ep % LOGGING_FREQUENCY == 0 or ep == wandb.config.NUM_EPISODES - 1
    )
    is_time_to_infer = (
        lambda ep: ep % INFERENCING_FREQUENCY == 0
        or ep == wandb.config.NUM_EPISODES - 1
    )

    checkpt_dir = wandb.run.dir
    print("Checkpoint directory: ", checkpt_dir)

    wandb.config.update(alg_config)  # log the config to wandb
    wandb.config.update(
        {"env_name": ENV_NAME, "setting": ENV_SETTING, "single_agent": SINGLE_AGENT}
    )

    # baselines
    if run_qres == "n":
        _baseline = env.core.baselines
    else:
        _baseline = env.baselines

    time0 = time.time()

    # training loop
    for ep in range(alg_config["NUM_EPISODES"]):

        episode_return = training_rollout(env, LearningAgent)

        LearningAgent.report()

        # CHECKPOINT
        if is_time_to_checkpoint(ep):
            print(f"Checkpointing at episode {ep}")
            LearningAgent.save_weights(checkpt_dir, ep)

        # INFERENCE
        if is_time_to_infer(ep):
            print(f"Inference at episode {ep}")
            # serial evaluation instead of parallel
            args_list = [(checkpt_dir, ep) for _ in range(NUM_INFERENCE_EPISODES)]
            list_func_evaluations = [parallel_rollout(args) for args in args_list]

            # Combine the results
            eval_costs = np.hstack(list_func_evaluations)

            _mean = np.mean(eval_costs)
            _stderr = np.std(eval_costs) / np.sqrt(len(eval_costs))

            # Directly track maximum reward (no negation needed)
            if _mean > best_reward:
                best_reward = _mean
                best_checkpt = ep

            training_log.update(
                {
                    "inference_ep": ep,
                    "inference_mean": _mean,
                    "inference_stderr": _stderr,
                    "best_reward": best_reward,
                    "best_checkpt": best_checkpt,
                }
            )

        # LOGGING
        if is_time_to_log(ep):
            print(f"Logging at episode {ep}")
            training_log.update(LearningAgent.logger)  # agent logger
            training_log.update(_baseline)  # baseline logger
            wandb.log(training_log, step=ep)  # log to wandb

    print(f"Total time: {time.time()-time0:.2f}")







