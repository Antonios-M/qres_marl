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
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
ENV_NAME = "quake-res-30-v1"
# ENV_NAME = "ma-grid-world-v0"
ENV_SETTING = "qres-marl-4"
ENV_KWARGS = {"percept_type": "state"}

run_qres = "y"
ALGORITHM = "DCMAC"
SINGLE_AGENT = False


# Environment
env = gym.make(ENV_NAME).unwrapped
inference_env = gym.make(ENV_NAME).unwrapped

# else:
#     ENV_NAME = "k_out_of_n_infinite"
#     ENV_SETTING = "hard-4-of-4_infinite"
#     ENV_KWARGS = {"percept_type": "belief", "reward_shaping": True}
#     env = imprl.envs.make(ENV_NAME, ENV_SETTING, single_agent=SINGLE_AGENT, **ENV_KWARGS)
#     inference_env = imprl.envs.make(
#         ENV_NAME,
#         ENV_SETTING,
#         single_agent=SINGLE_AGENT,
#         **ENV_KWARGS,
#     )

# Agent
alg_config = load_config(algorithm=ALGORITHM)  # load default config
agent_class = imprl.agents.get_agent_class(ALGORITHM)
LearningAgent = agent_class(env, alg_config, device)  # initialize agent
print(f"Loaded default configuration for {ALGORITHM}.")

PROJECT = "final-DCMAC-toy-city-30"
ENTITY = "antoniosmavrotas-tu-delft"
# WANDB_DIR = "./experiments/data"
# WANDB_DIR = "/scratch/pbhustali"

LOGGING_FREQUENCY = 100
CHECKPT_FREQUENCY = 5_000
INFERENCING_FREQUENCY = 5_000
NUM_INFERENCE_EPISODES = 500


def parallel_rollout(args):
    checkpt_dir, ep = args
    agent = agent_class(inference_env, alg_config, device)
    agent.load_weights(checkpt_dir, ep)
    returns, cal = evaluate_agent(inference_env, agent)
    return returns, cal


if __name__ == "__main__":

    run = wandb.init(project=PROJECT, entity=ENTITY)

    # logging and checkpointing
    training_log = {}  # log for training metrics

    best_reward = -math.inf
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
            LearningAgent.save_weights(checkpt_dir, ep)

        # INFERENCE
        if is_time_to_infer(ep):

            # parallel evaluation
            args_list = [(checkpt_dir, ep) for _ in range(NUM_INFERENCE_EPISODES)]

            # MAX_WORKERS = 2

            list_func_evaluations = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
                # Submit all the tasks and gather the futures
                futures = [
                    executor.submit(parallel_rollout, args) for args in args_list
                ]
                # Wait for all futures to complete and extract the results
                list_func_evaluations = []
                list_cal = []

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    list_func_evaluations.append(result[0])
                    list_cal.append(result[1])

                # Combine the results
                eval_costs = np.hstack(list_func_evaluations)
                # print(f"Evaluated {len(eval_costs)} episodes in parallel. Total Rewards: {eval_costs}")

            _mean_cal = np.mean(list_cal)
            _mean = np.mean(eval_costs)
            _stderr = np.std(eval_costs) / np.sqrt(len(eval_costs))

            if _mean > best_reward:
                best_reward = _mean
                best_checkpt = ep

            training_log.update(
                {
                    "inference_ep": ep,
                    "inference_mean_return": _mean,
                    "inference_stderr": _stderr,
                    "best_reward": best_reward,
                    "best_checkpt": best_checkpt,
                    "inference_mean_cal": _mean_cal
                }
            )

        # LOGGING
        if is_time_to_log(ep):
            training_log.update(LearningAgent.logger)  # agent logger
            training_log.update(_baseline)  # baseline logger
            wandb.log(training_log, step=ep)  # log to wandb

    print(f"Total time: {time.time()-time0:.2f}")
