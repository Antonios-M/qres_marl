import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import gymnasium as gym
import seaborn as sns
from .simulations.building_funcs import BuildingAction
from .simulations.road_funcs import RoadAction
from .simulations.resilience import Resilience
from .simulations.earthquake_funcs import DamageStates

class Qres_env_wrapper(gym.Env):

    def seed(self, seed: int = None) -> int:
        """
        Sets the seed for the environment and uses it to select a random earthquake magnitude.
        Returns the seed used.
        """

        if seed is None:
            self.eq_magnitude = np.random.choice(self.earthquake_choices)
            self.return_period = np.random.choice(self._return_periods)
        else:
            self.eq_magnitude = seed
            self.return_period = seed


    def __get_action_space(self):
        num_components = self.num_components
        does_match = len(BuildingAction) == len(RoadAction)
        if not does_match:
            raise ValueError("Building and Road Actions must have equal cardinality")
        self.num_actions = len(BuildingAction)
        agent_action_space = gym.spaces.Discrete(self.num_actions)
        self.action_space = gym.spaces.Tuple(
            (agent_action_space for _ in range(num_components))
        )

    def __get_observation_space(self):
        # self.shared_observation_space = gym.spaces.Box(0, len(DamageStates), shape=(1,), dtype=self.dtype) ## damage state
        max_obs = self.resilience._max_obs
        self.shared_observation_space = gym.spaces.Box(0, max_obs, shape=(1,), dtype=self.dtype)
        self.observation_space = gym.spaces.Tuple(
            tuple([self.shared_observation_space] * self.n_agents)
        )
        self.local_observation_space = self.observation_space
        self.perception_space = self.local_observation_space

    def __init__(self,
        verbose: bool=False,
        single_agent: bool=False,
        time_step_duration: int = 30,
        trucks_per_building_per_day: float = 0.1,
        n_agents: int = 30,
        n_crews: int = 25,
        time_horizon: int = 20,
        w_econ: float = 0.2,
        w_crit: float = 0.4,
        w_health: float = 0.4,
        quake_choices: list = [7.5, 8.0, 8.5, 9.0],
        reward_type: str = "loss"
    ):
        self.reward_type = reward_type
        self.single_agent = single_agent
        self.dtype = np.float32
        self.verbose = verbose
        self.time_step_duration = time_step_duration
        self.time_horizon = time_horizon
        self.trucks_per_building_per_day = trucks_per_building_per_day
        self.n_agents = n_agents
        self.num_components = self.n_agents
        self.n_crews = n_crews
        self.w_econ = w_econ
        self.w_crit = w_crit
        self.w_health = w_health
        self.n_damage_states = len(DamageStates)
        self.time_horizon = time_horizon
        self.baselines ={}
        self.earthquake_choices = quake_choices

        self.resilience = Resilience(
            n_agents=self.n_agents,
            n_crews=self.n_crews,
            time_horizon=self.time_horizon,
            time_step_duration=self.time_step_duration,
            truck_debris_per_day=self.trucks_per_building_per_day,
            w_econ=self.w_econ,
            w_crit=self.w_crit,
            w_health=self.w_health,
            w_health_bed=0.7,
            w_health_doc=0.3
        )
        if (self.resilience.num_buildings + self.resilience.num_roads) != self.n_agents:
            raise ValueError("Number of agents must match number of buildings and roads")

        self._return_periods = range(475, 3000)
        self.__get_observation_space()
        self.__get_action_space()

    def reset(self, seed: int = None, options: dict = None):

        self.seed(seed=seed)
        # print(f"Resetting with return period: {self.return_period}")
        # print(f"Resetting with earthquake magnitude: {self.eq_magnitude}")
        info = self.resilience.reset(
            rp=self.return_period,
            eq_magnitude=self.eq_magnitude
        )
        self.functionality = info["q"]["community_robustness"]
        self.functionality_econ = info["q"]["community_robustness_econ"]
        self.functionality_crit = info["q"]["community_robustness_crit"]
        self.functionality_health = info["q"]["community_robustness_health"]
        self.pq_functionality = self.functionality
        self.pq_functionality_econ = self.functionality_econ
        self.pq_functionality_crit = self.functionality_crit
        self.pq_functionality_health = self.functionality_health

        self.post_quake_funcs = np.array([
            self.pq_functionality,
            self.pq_functionality_econ,
            self.pq_functionality_crit,
            self.pq_functionality_health
        ])
        # print(f"Post-quake functionality: {self.post_quake_funcs}")
        obs = self.resilience.state(dtype=self.dtype)
        info["state"] = self._get_state(obs)
        q_deltas = self.__get_runtime_reward_metrics(info)

        info["reward"] = {
            "total": self.get_loss_reward(0, q_deltas),
            "resilience": self.R,
            "loss": - self.RL,
            "econ": self.get_loss_reward(1, q_deltas),
            "crit": self.get_loss_reward(2, q_deltas),
            "health": self.get_loss_reward(3, q_deltas),
        }

        # self.com_resilience = 0.0
        # self.econ_resilience = 0.0
        # self.crit_resilience = 0.0
        # self.health_resilience = 0.0
        return obs, info

    def get_loss_reward(self,
        q_idx: int,
        q_deltas: np.array
    ) -> float:
        q1, q2 = q_deltas[q_idx]

        R = 0.5 * self.time_step_duration * (q2 + q1)
        RL = self.time_step_duration - R

        self.RL = RL
        self.R = R

        return -RL


    def get_termination_reward(self,
        q_td: float,
        t_term: int
    ) -> float:
        """
        R(sbar) = [1 - Q(td)] * [th-TR]
        """
        return np.float32((1 - q_td) * (self.time_horizon - t_term))

    def get_state_reward(self,
        q_t: float,
        q_t_prev: float,
        q_min: float,
        q_max:float=1.0
    ) -> float:
        res_t = 0.5 * self.time_step_duration * (q_t + q_t_prev)
        max_res_t = self.time_step_duration * (q_max - q_min)
        loss_t = max_res_t - res_t

        return np.float32(-loss_t)

    def __get_runtime_reward_metrics(self, info):
        """
        - q_1 = q(t-1)
        - q_2 = q(t)
        where q(t) is the functionality at time t and measured with 0 being the post-earthquake functionality.
        """
        pq_funcs = self.post_quake_funcs

        ## community
        q_1_com = self.functionality
        q_2_com = info["q"]["community"]

        ## economic
        q_1_econ = self.functionality_econ
        q_2_econ = info["q"]["econ"]


        ## critical
        q_1_crit = self.functionality_crit
        q_2_crit = info["q"]["crit"]


        ## healthcare
        q_1_health = self.functionality_health
        q_2_health = info["q"]["health"]


        return np.array([ ## q(t-1), q(t)
            [q_1_com, q_2_com],
            [q_1_econ, q_2_econ],
            [q_1_crit, q_2_crit],
            [q_1_health, q_2_health]
        ])

    def step(self, actions: Tuple[int]):
        # Store previous functionality
        prev_functionality = self.functionality

        obs = self.resilience.state(dtype=self.dtype)
        terminated = self.resilience.terminated
        info = self.resilience.step(actions=actions)
        trunc_horizon, trunc_actions = self.resilience.truncated
        time = self.resilience.time
        q_deltas = self.__get_runtime_reward_metrics(info)

        # reward = np.float32(- (1 - info["q"]["community"]))
        rewards_parts = np.zeros((4, 2))  # change this if you add more subs-system functionalities
        rewards = np.zeros(4, dtype=np.float32)
        for i in range(4):
            reward = self.get_loss_reward(i, q_deltas)
            rewards[i] = reward
            rewards_parts[i] = [self.RL, self.R]


        reward = rewards[0]
        reward_econ = rewards[1]
        reward_crit = rewards[2]
        reward_health = rewards[3]

        self.functionality = info["q"]["community"]
        self.functionality_econ = info["q"]["econ"]
        self.functionality_crit = info["q"]["crit"]
        self.functionality_health = info["q"]["health"]

        # if reward == 0.0:
        #     reward = 50.0

        # print(f"func: {self.functionality}")
        # print(f"func_econ: {self.functionality_econ}")
        # print(f"func_crit: {self.functionality_crit}")
        # print(f"func_health: {self.functionality_health}")
        # print("-------")
        info["state"] = self._get_state(obs)

        info["reward"] = {
            "total": reward,
            "econ": reward_econ,
            "crit": reward_crit,
            "health": reward_health,
            "resilience": rewards_parts[0][1],
            "loss": reward
        }
        # print(f"reward: {reward}")
        return obs, reward, terminated, trunc_horizon, info

    def _get_state(self, obs) -> np.ndarray:
        n_states = int(self.shared_observation_space.high[0]) + 1
        obs = np.asarray(obs, dtype=int)
        one_hot = np.zeros((n_states, self.n_agents), dtype=int)
        one_hot[obs, np.arange(self.n_agents)] = 1
        return one_hot

    def system_percept(self, percept):
        # print(f"percept: {percept}")
        # print(f"percept shape: {type(percept)}")
        # print(f"local_observation_space: {self.local_observation_space}")
        local_percept = gym.spaces.utils.flatten(
            self.local_observation_space, percept
        )
        local_percept = local_percept.flatten()
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

    def __log(self, message):
        if self.verbose:
            print(message)
