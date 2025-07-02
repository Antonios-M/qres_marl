'''

This module contains the abstract base class: Agent, inherited by all agents.

'''

import random

from imprl.agents.modules.exploration_schedulers import LinearExplorationScheduler
from imprl.agents.modules.replay_memory import AbstractReplayMemory


class Agent:

    def __init__(self, env, config, device):

        # Environment
        self.env = env
        self.device = device  # to send neural network ops to device

        # Initialization
        self.episode = 0
        self.total_time = 0  # total time steps in lifetime
        self.time = 0  # time steps in current episode
        self.episode_return = 0  # return in current episode

        ## Training parameters
        self.discount_factor = config["DISCOUNT_FACTOR"]
        self.batch_size = config["BATCH_SIZE"]
        self.exploration_strategy = config["EXPLORATION_STRATEGY"]
        self.exploration_param = self.exploration_strategy["max_value"]

        # Evaluation parameters
        # try to use the discount factor from the environment
        try:
            self.eval_discount_factor = env.core.discount_factor
        # if env doesn't specify, compute undiscounted return
        except AttributeError:
            self.eval_discount_factor = 1.0

        # exploration scheduler
        self.exp_scheduler = LinearExplorationScheduler(
            self.exploration_strategy["min_value"],
            num_episodes=self.exploration_strategy["num_episodes"],
        )

        # replay memory stores (a subset of) experience across episodes
        self.replay_memory = AbstractReplayMemory(config["MAX_MEMORY_SIZE"])

        # logger
        self.logger = {"exploration_param": self.exploration_param}

    def reset_episode(self, training=True):

        self.episode_return = 0
        self.episode += 1
        self.time = 0

        ## qres-marl specific
        self._res = 0.0
        self._res_loss = 0.0


        if training:

            # update exploration param
            self.exploration_param = self.exp_scheduler.step()

            # logging
            self.logger["exploration_param"] = self.exploration_param

    def epsilon_greedy_strategy(self, observation, training):

        # select random action
        if self.exploration_param > random.random():
            return self.get_random_action()
        else:
            # select greedy action
            return self.get_greedy_action(observation, training)

    def select_action(self, observation, training):

        if training:
            if self.exploration_strategy["name"] == "epsilon_greedy":
                return self.epsilon_greedy_strategy(observation, training)
        else:
            return self.get_greedy_action(observation, training=False)

    def process_rewards(self, reward):

        # discounting
        self.episode_return += reward * self.eval_discount_factor**self.time

        # updating time here so that only this method needs to be called
        # during inference
        self.time += 1
        self.total_time += 1


    def process_avoided_losses(self, info):
        self._res += info["reward"]["resilience"]
        self._res_loss += info["reward"]["loss"]

    def compute_td_target(
        self, t_next_beliefs, t_rewards, t_terminations, t_truncations
    ):

        # bootstrapping
        future_values = self.get_future_values(t_next_beliefs)

        # set future values of terminal states to zero
        not_terminals = 1 - t_terminations
        future_values *= not_terminals
        td_target = t_rewards + self.discount_factor * future_values

        return td_target.detach()

    def report(self, stats=None):
        """Print stats to console."""

        print(f"Ep:{self.episode:05}| Reward: {self.episode_return:.2f}", flush=True)

        if stats is not None:
            print(stats)

    def process_experience(self):
        NotImplementedError

    def train(self):
        NotImplementedError

    def _preprocess_inputs(self):
        NotImplementedError

    def get_random_action(self):
        NotImplementedError

    def future_values(self):
        NotImplementedError

    def get_greedy_action(self):
        NotImplementedError

    def compute_loss(self):
        NotImplementedError

    def save_weights(self):
        NotImplementedError

    def load_weights(self):
        NotImplementedError