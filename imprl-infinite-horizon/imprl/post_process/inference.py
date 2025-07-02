import numpy as np
import torch
import matplotlib.pyplot as plt

import imprl.agents
from imprl.agents.configs.get_config import load_config
from imprl.runners.parallel import (
    parallel_agent_rollout,
    parallel_heursitic_rollout,
    parallel_generic_rollout,
)
from imprl.post_process.plotter.rollout_plotter import AgentPlotter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Inference:
    def __init__(self) -> None:
        pass

    def run(self, num_episodes=100):
        pass

    @staticmethod
    def compute_stats(episode_costs, verbose=True):
        mean = np.mean(episode_costs, axis=0)
        stderr = np.std(episode_costs, axis=0) / np.sqrt(len(episode_costs))

        if verbose:
            print(f"Total costs | mean: {mean:.2f}, std err: {stderr:.2f}")

        return mean, stderr

    @staticmethod
    def plot_costs(episode_costs, **kwargs):
        plt.hist(episode_costs, bins=100, **kwargs)


class AgentInference(Inference):
    def __init__(self, name, env):
        self.env = env
        self.name = name
        config = load_config(algorithm=name)  # load default config
        agent_class = imprl.agents.get_agent_class(name)
        self.agent = agent_class(env, config, DEVICE)  # initialize agent
        self.plotter = AgentPlotter(env, self.agent, name=name)

    def load_weights(self, location, episode):
        self.agent.load_weights(location, episode)

    def run(self, num_episodes=100, verbose=True):
        self.episode_costs = parallel_agent_rollout(self.env, self.agent, num_episodes)

        _ = self.compute_stats(self.episode_costs, verbose=verbose)

        return self.episode_costs

    def evaluate_critic(self, obs=None):
        if obs is None:
            obs = self.env.reset()

        return self.agent.evaluate_critic(obs)

    def plot_rollout(self, **kwargs):
        self.plotter.plot(**kwargs)

    def plot_3d(self, **kwargs):
        self.plotter.plot_returns_3d(**kwargs)

    def plot_components(self, **kwargs):
        self.plotter.plot_components(**kwargs)

    def run_n(self, num_episodes=100):
        self.plotter.run_n_steps(num_episodes)

    def get_rollout(self, **kwargs):
        self.plotter.get_sample_rollout(**kwargs)

    def plot_avoided_losses(self, **kwargs):
        self.plotter.plot_avoided_losses_stats(**kwargs)

    def get_n_rollouts(self, **kwargs):
        self.plotter.get_n_rollouts(**kwargs)


class HeuristicInference(Inference):
    def __init__(self, name, env):
        self.env = env
        self.name = name
        self.baseline=name
        self.plotter = AgentPlotter(env=self.env, agent=self.baseline, name=self.name)


    # def run(self, num_episodes=100):
    #     if self.name == "random":
    #         self.episode_costs = parallel_heursitic_rollout(
    #             self.env, self.baseline, num_episodes
    #         )

    #     _ = self.compute_stats(self.episode_costs)

    #     return self.episode_costs

    def plot_rollout(self, **kwargs):
        self.plotter.plot(**kwargs, policy=self.name)

    def plot_3d(self, **kwargs):
        self.plotter.plot_returns_3d(**kwargs)

    def plot_components(self, **kwargs):
        self.plotter.plot_components(**kwargs)

    def run_n(self, num_episodes=100):
        self.plotter.run_n_steps(num_episodes)

    def get_rollout(self, **kwargs):
        self.plotter.get_sample_rollout(**kwargs)

    def plot_avoided_losses(self, **kwargs):
        self.plotter.plot_avoided_losses_stats(**kwargs)

    def get_n_rollouts(self, **kwargs):
        self.plotter.get_n_rollouts(**kwargs)
