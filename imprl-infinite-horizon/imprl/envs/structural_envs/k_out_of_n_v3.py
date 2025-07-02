import numpy as np


class KOutOfN:

    def __init__(
        self,
        env_config: dict,
        baselines: dict = None,
        percept_type: str = "belief",
    ) -> None:

        self.reward_to_cost = True

        self.k = env_config["k"]
        self.time_horizon = env_config["time_horizon"]
        self.discount_factor = env_config["discount_factor"]
        self.FAILURE_PENALTY_FACTOR = env_config["failure_penalty_factor"]
        self.n_components = env_config["n_components"]
        self.n_damage_states = env_config["n_damage_states"]
        self.n_comp_actions = env_config["n_comp_actions"]
        try:
            self.initial_belief = env_config["initial_belief"]
        except KeyError:
            print("Initial belief not specified")

        ####################### Transition Model #######################

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

        ######################### Reward Model #########################

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
        try:
            self.mobilisation_reward = env_config["mobilisation_reward"]
        except KeyError:
            print("Mobilisation reward not specified.")

        ####################### Observation Model ######################

        inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )
        no_inspection_model = np.zeros(
            (self.n_components, self.n_damage_states, self.n_damage_states)
        )
        for c in range(self.n_components):

            p = env_config["obs_accuracies"][c]
            try:
                f_p = env_config["failure_obs_accuracies"][c]
            except KeyError:
                print("Failure observation accuracy not specified.")

            inspection_model[c] = np.array(
                [
                    [p, 1 - p, 0.0, 0.0],
                    [(1 - p) / 2, p, (1 - p) / 2, 0.0],
                    [0.0, (1 - p) / 2, p, (1 - p) / 2],
                    [0.0, 0.0, 1 - f_p, f_p],
                ]
            )
            no_inspection_model[c] = np.array(
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
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

            # do nothing
            self.observation_model[c, 0, :, :] = no_inspection_model[c]

            # replacement
            self.observation_model[c, 1, :, :] = no_inspection_model[c]

            # inspection
            self.observation_model[c, 2, :, :] = inspection_model[c]

        self.percept_type = percept_type
        self.baselines = baselines

        self.state = self.reset()

    def get_reward(
        self, state: list, action: list
    ) -> tuple[float, float, float, float]:

        reward_penalty = 0
        reward_replacement = 0
        reward_inspection = 0

        for c in range(self.n_components):

            if action[c] == 1:
                reward_replacement += self.rewards_table[c, state[c], action[c]]

            elif action[c] == 2:
                reward_inspection += self.rewards_table[c, state[c], action[c]]

        # mobilisation cost
        mobilised = sum(action) > 0
        reward_mobilisation = mobilised * self.mobilisation_reward

        # check number of failed components
        _temp = state // (self.n_damage_states - 1)  # 0 if working, 1 if failed
        n_working = self.n_components - np.sum(_temp)  # number of working components

        functional = True if n_working >= self.k else False

        if not functional:
            reward_penalty = (
                self.FAILURE_PENALTY_FACTOR * self.system_replacement_reward
            )

        # discounted reward
        _discount_factor = self.discount_factor**self.time
        reward_replacement *= _discount_factor
        reward_inspection *= _discount_factor
        reward_penalty *= _discount_factor
        reward_mobilisation *= _discount_factor

        reward = (
            reward_replacement
            + reward_inspection
            + reward_penalty
            + reward_mobilisation
        )

        return (
            reward,
            reward_replacement,
            reward_inspection,
            reward_penalty,
            reward_mobilisation,
        )

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

        # collect reward: R(s,a)
        (
            reward,
            reward_replacement,
            reward_inspection,
            reward_penalty,
            reward_mobilisation,
        ) = self.get_reward(self.damage_state, action)

        # compute next damage state
        next_state = self.get_next_state(self.damage_state, action)
        self.damage_state = next_state

        # compute observation
        self.observation = self.get_observation(next_state, action)

        # update belief only if percept_type is belief to avoid unnecessary computation
        if self.percept_type in ["belief", "Belief"]:
            self.belief = self.belief_update(self.belief, action, self.observation)

        # update time
        self.time += 1
        self.norm_time = self.time / self.time_horizon

        # check if terminal state
        done = True if self.time == self.time_horizon else False

        # update info dict
        info = {
            "system_failure": reward_penalty < 0,
            "reward_replacement": reward_replacement,
            "reward_inspection": reward_inspection,
            "reward_penalty": reward_penalty,
            "reward_mobilisation": reward_mobilisation,
            "state": self._get_state(),
            "observation": self.observation,
        }

        return self._get_percept(), reward, done, info

    def reset(self, **kwargs) -> tuple[np.array, np.array]:

        # duplicate the initial belief for each component
        self.belief = np.tile(self.initial_belief, (self.n_components, 1)).T

        self.damage_state = np.random.choice(
            self.n_damage_states, p=self.initial_belief, size=self.n_components
        )
        self.observation = np.random.choice(
            self.n_damage_states, p=self.initial_belief, size=self.n_components
        )

        # reset the time
        self.time = 0
        self.norm_time = self.time / self.time_horizon

        info = {
            "system_failure": False,
            "reward_replacement": 0,
            "reward_inspection": 0,
            "reward_penalty": 0,
            "reward_mobilisation": 0,
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
