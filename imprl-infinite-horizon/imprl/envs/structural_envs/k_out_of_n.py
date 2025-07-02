import numpy as np


class KOutOfN:

    def __init__(
        self,
        env_config: dict,
        baselines: dict = None,
        percept_type: str = "belief",
        reward_shaping: bool = False,
    ) -> None:

        self.time_perception = True  # agent perceives time
        self.reward_to_cost = True
        self.reward_shaping = reward_shaping

        self.k = env_config["k"]
        self.time_horizon = env_config["time_horizon"]
        self.discount_factor = env_config["discount_factor"]
        self.FAILURE_PENALTY_FACTOR = env_config["failure_penalty_factor"]
        self.n_components = env_config["n_components"]
        self.n_damage_states = env_config["n_damage_states"]
        self.n_comp_actions = env_config["n_comp_actions"]

        # Transition model

        # shape: (n_components, n_damage_states, n_damage_states)
        self.deterioration_table = np.array(env_config["transition_model"])

        self.replacement_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):
            r = env_config["replacement_accuracies"][c]
            self.replacement_table[c] = np.array(
                [[1, 0, 0, 0], [r, 1 - r, 0, 0], [r, 0, 1 - r, 0], [r, 0, 0, 1 - r]]
            )

        self.transition_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: deterioration
            self.transition_model[c, 0, :, :] = self.deterioration_table[c, :, :]

            # replacement: replace instantly + deterioration
            # D^T @ R^T @ belief ==> (R @ D)^T @ belief
            self.transition_model[c, 1, :, :] = (
                self.replacement_table[c] @ self.deterioration_table[c, :, :]
            )

            # inspect: deterioration
            self.transition_model[c, 2, :, :] = self.deterioration_table[c, :, :]

        # Reward model
        self.rewards_table = np.zeros(
            (self.n_components, self.n_damage_states, self.n_comp_actions)
        )

        self.rewards_table[:, :, 1] = np.array(
            [env_config["replacement_rewards"]] * self.n_damage_states
        ).T
        self.rewards_table[:, :, 2] = np.array(
            [env_config["inspection_rewards"]] * self.n_damage_states
        ).T

        self.system_replacement_reward = sum(env_config["replacement_rewards"])

        # Observation model
        self.inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )

        for c in range(self.n_components):
            p = env_config["obs_accuracies"][c]
            self.inspection_model[c] = np.array(
                [
                    [p, 1 - p, 0.0, 0.0],
                    [(1 - p) / 2, p, (1 - p) / 2, 0.0],
                    [0.0, 1 - p, p, 0.0],
                    [0.0, 0.0, 0.0, 1],
                ]
            )

        self.failure_obs_model = np.array(
            [
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [1 / 3, 1 / 3, 1 / 3, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.observation_model = np.zeros(
            (
                self.n_components,
                self.n_comp_actions,
                self.n_damage_states,
                self.n_damage_states,
            )
        )

        for c in range(self.n_components):

            # do nothing: only failure observation
            self.observation_model[c, 0, :, :] = self.failure_obs_model

            # replacement: only failure observation
            self.observation_model[c, 1, :, :] = self.failure_obs_model

            # inspection: inspection
            self.observation_model[c, 2, :, :] = self.inspection_model[c]

        self.percept_type = percept_type
        self.baselines = baselines

        self.state = self.reset()

    @staticmethod
    def pf_sys(pf, k):
        """Computes the system failure probability pf_sys for k-out-of-n components

        Args:
            pf: Numpy array with components' failure probability.
            k: Integer indicating k (out of n) components.

        Returns:
            PF_sys: Numpy array with the system failure probability.
        """
        n = pf.size
        nk = n - k
        m = k + 1
        A = np.zeros(m + 1)
        A[1] = 1
        L = 1
        for j in range(1, n + 1):
            h = j + 1
            Rel = 1 - pf[j - 1]
            if nk < j:
                L = h - nk
            if k < j:
                A[m] = A[m] + A[k] * Rel
                h = k
            for i in range(h, L - 1, -1):
                A[i] = A[i] + (A[i - 1] - A[i]) * Rel
        PF_sys = 1 - A[m]
        return PF_sys

    def _is_system_functional(self, state):

        # check number of failed components
        _is_failed = state // (self.n_damage_states - 1)  # 0 if working, 1 if failed
        n_working = self.n_components - np.sum(_is_failed)
        functional = n_working >= self.k

        return functional

    def get_reward(
        self, state: list, belief: np.array, action: list, next_belief: np.array
    ) -> tuple[float, float, float, float]:

        reward_replacement = 0
        reward_inspection = 0
        reward_system = 0

        failure_cost = self.system_replacement_reward * self.FAILURE_PENALTY_FACTOR

        for c in range(self.n_components):

            if action[c] == 1:  # replacement
                reward_replacement += self.rewards_table[c, state[c], action[c]]

            elif action[c] == 2:  # inspection
                reward_inspection += self.rewards_table[c, state[c], action[c]]

        if self.reward_shaping:
            pf = belief[-1, :]
            pf_sys = self.pf_sys(pf, self.k)
            reward_system = failure_cost * pf_sys
        else:  # state-based reward
            if not self._is_system_functional(state):
                reward_system = failure_cost

        # discounted reward
        _discount_factor = self.discount_factor**self.time
        reward_replacement *= _discount_factor
        reward_inspection *= _discount_factor

        if self.reward_shaping:
            reward_system *= _discount_factor

        reward = reward_replacement + reward_inspection + reward_system

        return reward, reward_replacement, reward_inspection, reward_system

    def get_next_state(self, state: np.array, action: list) -> np.array:

        _next_states = np.zeros(self.n_components, dtype=int)

        for c in range(self.n_components):

            next_damage_state = np.random.choice(
                np.arange(self.n_damage_states),
                p=self.transition_model[c, action[c], state[c], :],
            )

            _next_states[c] = next_damage_state

        return _next_states

    def get_observation(self, nextstate: list, action: list) -> np.array:

        _observations = np.zeros(self.n_components, dtype=int)

        for c in range(self.n_components):

            obs = np.random.choice(
                np.arange(self.n_damage_states),
                p=self.observation_model[c, action[c], nextstate[c], :],
            )

            _observations[c] = obs

        return _observations

    def belief_update(
        self, belief: np.array, action: list, observation: list
    ) -> np.array:

        next_belief = np.empty((self.n_damage_states, self.n_components))

        for c in range(self.n_components):

            belief_c = belief[:, c]

            # transition model
            belief_c = self.transition_model[c, action[c]].T @ belief_c

            # observation model
            state_probs = self.observation_model[c, action[c], :, observation[c]]
            belief_c = state_probs * belief_c

            # normalise
            belief_c = belief_c / np.sum(belief_c)

            next_belief[:, c] = belief_c

        return next_belief

    def step(self, action: list) -> tuple[np.array, float, bool, dict]:

        # compute next damage state
        next_state = self.get_next_state(self.damage_state, action)

        # compute observation
        self.observation = self.get_observation(next_state, action)

        # update belief only if percept_type is belief to avoid unnecessary computation
        if self.percept_type in ["belief", "Belief"]:
            next_belief = self.belief_update(self.belief, action, self.observation)

        # collect reward: R(s,a)
        reward, reward_replacement, reward_inspection, reward_system = self.get_reward(
            self.damage_state, self.belief, action, next_belief
        )

        self.damage_state = next_state
        if self.percept_type in ["belief", "Belief"]:
            self.belief = next_belief

        # update time
        self.time += 1
        self.norm_time = self.time / self.time_horizon

        # check if terminal state
        terminated = True if self.time == self.time_horizon else False

        # update info dict
        info = {
            "system_failure": self._is_system_functional(self.damage_state),
            "reward_replacement": reward_replacement,
            "reward_inspection": reward_inspection,
            "reward_system": reward_system,
            "state": self._get_state(),
            "observation": self.observation,
        }

        return self._get_percept(), reward, terminated, False, info

    def reset(self, **kwargs) -> tuple[np.array, np.array]:

        # In the initial state, all components are undamaged
        initial_damage = 0
        self.damage_state = np.array([initial_damage] * self.n_components, dtype=int)
        self.observation = np.array([initial_damage] * self.n_components, dtype=int)
        if self.percept_type in ["belief", "Belief"]:
            belief = np.zeros((self.n_damage_states, self.n_components))
            belief[initial_damage, :] = 1
            self.belief = belief

        # reset the time
        self.time = 0
        self.norm_time = self.time / self.time_horizon

        info = {
            "system_failure": False,
            "reward_replacement": 0,
            "reward_inspection": 0,
            "reward_system": 0,
            "state": self._get_state(),
            "observation": self.observation,
        }

        return self._get_percept(), info

    def _get_percept(self) -> tuple[np.array, np.array]:

        if self.percept_type in ["belief", "Belief"]:
            return self._get_belief()
        elif self.percept_type in ["state", "State"]:
            return self._get_state()
        elif self.percept_type in ["obs", "Obs"]:
            return self._get_observation()

    def _get_state(self) -> tuple[np.array, np.array]:
        one_hot = np.zeros((self.n_damage_states, self.n_components))
        one_hot[self.damage_state, np.arange(self.n_components)] = 1
        return (np.array([self.norm_time]), one_hot)

    def _get_observation(self) -> tuple[np.array, np.array]:
        one_hot = np.zeros((self.n_damage_states, self.n_components))
        one_hot[self.observation, np.arange(self.n_components)] = 1
        return (np.array([self.norm_time]), one_hot)

    def _get_belief(self) -> tuple[np.array, np.array]:
        return (np.array([self.norm_time]), self.belief)
