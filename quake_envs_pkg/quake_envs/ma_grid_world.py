import numpy as np
import random
import gymnasium as gym

class SimpleGridWorld(gym.Env):
    def __init__(self, grid_size=5):
        self.grid_size = grid_size
        self.n_agents = 2
        self.local_observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(2,), dtype=np.int32)
        self.observation_space = self.local_observation_space
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(5)))
        self.dtype = np.float32
        self.baselines = {}
        self.time_horizon = (self.grid_size * 2) + 1

    def reset(self,seed=None, options=None):
        self.time = 0
        # Initialize agents' positions randomly
        self.agent_positions = [self._random_position() for _ in range(self.n_agents)]
        # Assign a goal for each agent
        self.goals = [self._random_position() for _ in range(self.n_agents)]
        return self._get_obs(), {}

    def _random_position(self):
        return (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

    def _get_obs(self):
        state = np.array([
            self.agent_positions[0],
            self.goals[0],
            self.agent_positions[1],
            self.goals[1]
        ])
        return state

    def truncated(self):
        if self.time >= self.time_horizon:
            return True
        else:
          return False

    def terminated(self):
        term = [pos == goal for pos, goal in zip(self.agent_positions, self.goals)]
        return all(term)

    def system_percept(self, percept):
        local_percept = gym.spaces.utils.flatten(
            self.local_observation_space, percept
        )
        return local_percept

    def multiagent_percept(self, percept):
        local_percept = gym.spaces.utils.flatten(self.local_observation_space, percept.T)
        local_percept = local_percept.reshape(-1, self.n_agents, order="F")

        return local_percept.T

    def multiagent_idx_percept(self, percept):

        # (id, shared_percept, local_percept)
        eye = np.eye(self.n_agents, dtype=self.dtype)
        _ma_percept = self.multiagent_percept(percept)

        return np.concatenate((eye, _ma_percept), axis=1)

    def step(self, actions):
        rewards = [0, 0]
        new_positions = []

        for i, action in enumerate(actions):
            x, y = self.agent_positions[i]
            if action == 0:    # up
                x = max(x - 1, 0)
            elif action == 1:  # down
                x = min(x + 1, self.grid_size - 1)
            elif action == 2:  # left
                y = max(y - 1, 0)
            elif action == 3:  # right
                y = min(y + 1, self.grid_size - 1)
            elif action == 4:  # stay
                pass
            new_positions.append((x, y))

            # Reward if agent reaches its goal
            if (x, y) == self.goals[i]:
                rewards[i] = 1

        self.time += 1
        self.agent_positions = new_positions
        obs = self._get_obs()

        return obs, sum(rewards), self.terminated(), self.truncated(), {}