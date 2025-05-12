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

    def seed(self, seed: int = None, quake_seed: int = None) -> int:
        """
        Sets the seed for the environment and uses it to select a random earthquake magnitude.
        Returns the seed used.
        """
        if quake_seed:
            self.eq_magnitude = quake_seed
            return
        if seed is None:
            # If no seed is provided, choose one at random from the valid indices
            seed = np.random.randint(0, len(self.earthquake_choices))


        # Create a reproducible RNG with the given seed
        rng = np.random.RandomState(seed)
        # Select a random earthquake magnitude from the choices
        self.eq_magnitude = rng.choice(self.earthquake_choices)

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
        quake_seed = None,
        quake_choices: list = [7.5, 8.0, 8.5, 9.0],
    ):
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
        self.quake_seed = quake_seed

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

        self.__get_observation_space()
        self.__get_action_space()

    def reset(self, seed: int = None, options: dict = None):

        self.seed(seed=seed, quake_seed=self.quake_seed)
        q_community, q_econ, q_crit, q_health = self.resilience.reset(eq_magnitude=self.eq_magnitude)
        self.functionality = q_community
        self.functionality_econ = q_econ
        self.functionality_crit = q_crit
        self.functionality_health = q_health
        obs = self.resilience.state(dtype=self.dtype)
        info = {
            "state" : self._get_state(obs)
        }
        self.com_resilience = 0.0
        self.econ_resilience = 0.0
        self.crit_resilience = 0.0
        self.health_resilience = 0.0
        return obs, info

    def get_cal_reward(self,
        q_1: float,
        q_2: float,
        q_min: float,
        q_max: float,
    )-> float:
        """
        Calculate the reward as the ***cumulative avoided losses***
        CAL = R / ( L + R )
        where R is the resilience, L is the loss, q_max is the maximum functionality,
        q_min is the minimum, post-quake functionality.

        **Derivation**:

        - R = 0.5Δt * (q1 + q2 - 2qmin)
        - L + R = Δt * (qmax - qmin)
        - CAL = R / (L + R) = 0.5Δt * (q1 + q2 - 2qmin) / (Δt * (qmax - qmin))
        - CAL = 0.5 * (q1 + q2 - 2qmin) / (qmax - qmin)

        - where:
            - q1 is the pre-quake functionality, q2 is the post-quake functionality,
            - q_min is the minimum post-quake functionality, and q_max is the maximum functionality.
        """
        # Calculate the reward as the cumulative avoided losses
        cal_reward = 0.5 * (q_1 + q_2 - 2 * q_min) / (q_max - q_min)
        if cal_reward < 0:
            print(f"cal_reward: {cal_reward}")
            print(f"q_1: {q_1}")
            print(f"q_2: {q_2}")
            print(f"q_min: {q_min}")
            print(f"q_max: {q_max}")

        return np.float32(cal_reward)


    def step(self, actions: Tuple[int]):
        info = self.resilience.step(actions=actions)
        obs = self.resilience.state(dtype=self.dtype)
        post_quake_func = info["q"]["community_robustness"]
        post_quake_func_econ = info["q"]["community_robustness_econ"]
        post_quake_func_crit = info["q"]["community_robustness_crit"]
        post_quake_func_health = info["q"]["community_robustness_health"]
        current_func = info["q"]["community"]
        current_func_econ = info["q"]["econ"]
        current_func_crit = info["q"]["crit"]
        current_func_health = info["q"]["health"]

        q_1_com = max((self.functionality - post_quake_func), post_quake_func)
        q_2_com = max((current_func - post_quake_func), post_quake_func)
        q_1_econ = max((self.functionality_econ - post_quake_func_econ), post_quake_func_econ)
        q_2_econ = max((current_func_econ - post_quake_func_econ), post_quake_func_econ)
        q_1_crit = max((self.functionality_crit - post_quake_func_crit), post_quake_func_crit)
        q_2_crit = max((current_func_crit - post_quake_func_crit), post_quake_func_crit)
        q_1_health = max((self.functionality_health - post_quake_func_health), post_quake_func_health)
        q_2_health = max((current_func_health -
        post_quake_func_health), post_quake_func_health)


        reward_econ = np.float32(0.5 * self.time_step_duration * (q_1_econ + q_2_econ))
        reward_crit = np.float32(0.5* self.time_step_duration * (q_1_crit + q_2_crit))
        reward_health = np.float32(0.5 * self.time_step_duration * (q_1_health + q_2_health))
        reward = np.float32(0.5 * self.time_step_duration * (q_1_com + q_2_com))

        # print(f"Reward: {reward}")
        # print(f"res_a_community: {res_a_community}")
        # print(f"res_b_community: {res_b_community}")

        # Update state trackers
        self.functionality = current_func
        self.functionality_econ = current_func_econ
        self.functionality_crit = current_func_crit
        self.functionality_health = current_func_health

        terminated = self.resilience.terminated
        trunc_horizon, trunc_actions = self.resilience.truncated

        info["state"] = self._get_state(obs)
        info["reward"] = {
            "total": reward,
            "econ": reward_econ,
            "crit": reward_crit,
            "health": reward_health
        }
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