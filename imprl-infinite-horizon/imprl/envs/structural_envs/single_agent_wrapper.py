"""
A gymnasium 

"""

import itertools
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class SingleAgentWrapper(gym.Env):

    def __init__(self, env) -> None:

        self.core = env
        self.dtype = np.float32  # or np.float64

        # Gym Spaces (joint spaces)

        # Percept space (belief/state/obs space)
        self.state_space = gym.spaces.MultiDiscrete(
            np.array([self.core.n_damage_states] * self.core.n_components)
        )
        self.belief_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.core.n_damage_states, self.core.n_components),
            dtype=self.dtype,
        )
        if env.time_perception:
            self.state_space = gym.spaces.Tuple(
                (
                    # normalized time
                    gym.spaces.Box(0, 1, shape=(1,), dtype=self.dtype),
                    # state space
                    self.state_space,
                )
            )
            self.belief_space = gym.spaces.Tuple(
                (
                    # normalized time
                    gym.spaces.Box(0, 1, shape=(1,), dtype=self.dtype),
                    # belief space
                    self.belief_space,
                )
            )

        self.observation_space = self.state_space

        if self.core.percept_type in ["belief", "Belief"]:
            self.percept_space = self.belief_space
        elif self.core.percept_type in ["state", "State"]:
            self.percept_space = self.state_space
        elif self.core.percept_type in ["obs", "Obs"]:
            self.percept_space = self.observation_space

        # Action space
        self.action_space = spaces.Discrete(
            self.core.n_comp_actions**self.core.n_components
        )

        _action_space = list(
            itertools.product(
                np.arange(self.core.n_comp_actions), repeat=self.core.n_components
            )
        )
        self.joint_action_map = [list(action) for action in _action_space]

        self.perception_dim = gym.spaces.utils.flatdim(self.percept_space)
        self.action_dim = gym.spaces.utils.flatdim(self.action_space)

    def step(self, action: int):
        action = self.joint_action_map[action]

        next_percept, reward, terminated, truncated, info = self.core.step(action)

        return next_percept, self.dtype(reward), terminated, truncated, info

    def reset(self, **kwargs):
        return self.core.reset(**kwargs)

    def system_percept(self, percept):
        return spaces.utils.flatten(self.percept_space, percept)
