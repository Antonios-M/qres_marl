import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import gymnasium as gym
import seaborn as sns
from .simulations.building_funcs import BuildingAction
from .simulations.road_funcs import RoadAction
from .simulations.resilience import Resilience

class Quake_Res_30(gym.Env):

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
        agent_observation_space = gym.spaces.Box(low=0.0, high=600, shape=(1,), dtype=self.dtype)
        self.observation_space = agent_observation_space
        return self.observation_space

    def __init__(self,
        verbose: bool=False,
    ):
        self.dtype = np.float32
        self.verbose = verbose
        self.time_step_duration = 150
        self.trucks_per_building_per_day = 0.5
        self.n_agents = 30
        self.n_crews = 25
        self.time_horizon = 20
        self.baselines ={}

        self.resilience = Resilience(
            n_crews=self.n_crews,
            time_horizon=self.time_horizon,
            time_step_duration=self.time_step_duration,
            truck_debris_per_day=self.trucks_per_building_per_day,
            w_econ=0.2,
            w_crit=0.4,
            w_health=0.4,
            w_health_bed=0.7,
            w_health_doc=0.3
        )
        self.num_components = self.resilience.num_buildings + self.resilience.num_roads
        self.__get_observation_space()
        self.__get_action_space()

    def reset(self, seed: int = None, options: dict = None):
        self.earthquake_choices = [7.5, 8.0, 8.5, 9.0]
        self.seed(seed=seed)
        q_community, q_econ, q_crit, q_health = self.resilience.reset(eq_magnitude=self.eq_magnitude)
        self.functionality = q_community
        self.functionality_econ = q_econ
        self.functionality_crit = q_crit
        self.functionality_health = q_health
        info = {}
        obs = self.resilience.state(dtype=self.dtype)

        return obs, info


    def step(self, actions: Tuple[int]):
        info = self.resilience.step(actions=actions)
        obs = self.resilience.state(dtype=self.dtype)
        current_func = info["q"]["community"]
        current_func_econ = info["q"]["econ"]
        current_func_crit = info["q"]["crit"]
        current_func_health = info["q"]["health"]



        func_difference = current_func - self.functionality
        func_econ_difference = current_func_econ - self.functionality_econ
        func_crit_difference = current_func_crit - self.functionality_crit
        func_health_difference = current_func_health - self.functionality_health

        reward = np.float32(func_difference)

        self.functionality = current_func
        self.functionality_econ = current_func_econ
        self.functionality_crit = current_func_crit
        self.functionality_health = current_func_health

        terminated = self.resilience.terminated
        truncated = self.resilience.truncated

        info["reward"] = {
            "total": reward,
            "econ": func_econ_difference,
            "crit": func_crit_difference,
            "health": func_health_difference
        }

        return obs, reward, terminated, truncated, info

    def system_percept(self, percept):
        ## return flatten joint obseration space
        ## used in QMIX_PS
        pass

    def multiagent_percept(self, percept):
        local_percept = gym.spaces.utils.flatten(
            self.observation_space, percept.T
        )
        local_percept = local_percept.reshape(-1, self.num_components, order="F")
        return local_percept.T

    def multiagent_idx_percept(self, percept):
        eye = np.eye(self.n_agents, dtype=self.dtype)
        _map_percept = self.multiagent_percept(percept)
        return np.concatenate((eye, _map_percept), axis=1)

    def __log(self, message):
        if self.verbose:
            print(message)

    def plot_rollout(
        self,
        plot_econ_income=False,
        plot_econ_bldg_repair=False,
        plot_econ_road_repair=False,
        plot_econ_traffic=False,
        plot_econ_relocation=False,
        figsize=(10, 6),
        agent=None ## this is where trained agents are passed to be used during inference, see imprl-infinite-horizon:examples:inference.py
    ):
        padding_length = int(0.2 * self.time_horizon)
        # Reset environment and set up padding
        obs, info = self.reset()

        returns = [0.0] * padding_length
        returns_econ = [0.0] * padding_length
        returns_crit = [0.0] * padding_length
        returns_health = [0.0] * padding_length
        q_community_values = [1.0] * padding_length
        q_econ_values = [1.0] * padding_length
        q_income_values = [1.0] * padding_length
        q_econ_bldg_repair_costs = [1.0] * padding_length
        q_econ_road_repair_costs = [1.0] * padding_length
        q_econ_traffic_delay_costs = [1.0] * padding_length
        q_econ_relocation_costs = [1.0] * padding_length
        q_crit_values = [1.0] * padding_length
        q_health_values = [1.0] * padding_length
        bldg_repairs_counts = []
        debris_clear_counts = []
        func_rest_counts = []
        road_repair_counts = []
        time = list(range(-padding_length, 0))  # Time steps from -100 to -1

        current_time = 0
        has_terminated, has_truncated = False, False

        def has_rollout_ended(term, trunc):
            return trunc or term

        rollout_start = 0
        rollout_end = 0
        while not has_rollout_ended(has_terminated, has_truncated):
            rollout_end += 1
            if agent:
                action = agent.select_action(obs, training=False)
                action = tuple(action)
            else:
                action = self.action_space.sample()
            obs, cost, has_terminated, has_truncated, info = self.step(action)

            reward = info["reward"]["total"]
            reward_econ = info["reward"]["econ"]
            reward_crit = info["reward"]["crit"]
            reward_health = info["reward"]["health"]
            q_community = info["q"]["community"]
            q_econ = info["q"]["econ"]
            q_crit = info["q"]["crit"]
            q_health = info["q"]["health"]
            q_econ_bldg_income = info["q_econ_components"]["income"]
            q_econ_bldg_repair = info["q_econ_components"]["buildings_repair_cost"]
            q_econ_road_repair = info["q_econ_components"]["roads_repair_cost"]
            q_econ_traffic = info["q_econ_components"]["traffic_delay_cost"]
            q_econ_bldg_relocation = info["q_econ_components"]["relocation_cost"]
            repairs_count = sum(info["completions"]["bldg_repairs"])
            road_repair_count = sum(info["completions"]["road_repairs"])
            debris_clear_count = sum(info["completions"]["bldg_debris"])
            func_rest_count = sum(info["completions"]["bldg_funcs"])


            returns.append(reward)
            returns_econ.append(reward_econ)
            returns_crit.append(reward_crit)
            returns_health.append(reward_health)
            q_community_values.append(q_community)
            q_econ_values.append(q_econ)
            q_econ_bldg_repair_costs.append(q_econ_bldg_repair)
            q_income_values.append(q_econ_bldg_income)
            q_econ_road_repair_costs.append(q_econ_road_repair)
            q_econ_traffic_delay_costs.append(q_econ_traffic)
            q_econ_relocation_costs.append(q_econ_bldg_relocation)
            q_crit_values.append(q_crit)
            q_health_values.append(q_health)
            bldg_repairs_counts.append(repairs_count)
            road_repair_counts.append(road_repair_count)
            debris_clear_counts.append(debris_clear_count)
            func_rest_counts.append(func_rest_count)
            time.append(current_time)
            current_time += 1

        all_data = [returns, returns_econ, returns_crit, returns_health, q_community_values, q_econ_values, q_income_values,  q_econ_bldg_repair_costs, q_econ_road_repair_costs, q_econ_traffic_delay_costs, q_econ_relocation_costs, q_crit_values, q_health_values, bldg_repairs_counts, road_repair_counts, debris_clear_counts, func_rest_counts]
        time.extend(range(current_time, current_time + padding_length))

        # Add padding at the end
        def extend_end():
            for data in all_data:
                data.extend([data[-1]] * padding_length)

        extend_end()


        # Set up Seaborn style
        sns.set_theme(style="whitegrid", palette="pastel")

        # Create figure with two vertically stacked subplots sharing the x-axis
        fig, (ax_main, ax_rewards) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}
        )

        fig.patch.set_facecolor('white')
        ax_main.set_facecolor('white')
        ax_rewards.set_facecolor('beige')

        # Background colors
        normal_op_color = 'forestgreen'
        normal_op_alpha = 0.1
        recovery_color = 'orange'
        recovery_alpha = 0.1

        # Green background before timestep 0
        ax_main.axvspan(xmin=-padding_length, xmax=0, color=normal_op_color, alpha=normal_op_alpha, zorder=-10)

        # Determine recovery and normal op ranges
        finished_repair = 0
        recovery_ranges = []
        normal_op_ranges = []
        start_time = None
        for t, count in zip(time[padding_length:], func_rest_counts):
            if count > 0:
                finished_repair = t
        for t, total_reward in zip(time[padding_length:], returns[padding_length:]):
            if total_reward != returns[-1]:
                if start_time is None:
                    start_time = t
            else:
                if start_time is not None:
                    recovery_ranges.append((start_time, t))
                    start_time = None
                normal_op_ranges.append((t, t + 1))
        if start_time is not None:
            recovery_ranges.append((start_time, time[-1]))

        # Background spans for phases
        try:
            ax_main.axvspan(rollout_start, rollout_end, color=recovery_color, alpha=recovery_alpha, zorder=-10)
        except:
            pass
        try:
            ax_main.axvspan(rollout_end, normal_op_ranges[-1][1], color=normal_op_color, alpha=normal_op_alpha, zorder=-10)
        except:
            pass

        color_community = "black"
        color_econ = "teal"
        color_crit = "darkorange"
        color_health = "crimson"
        # Functionality line plots
        def plot_sub_functionalities(ax):
            sns.lineplot(x=time, y=q_econ_values, ax=ax, label='Economic Functionality', color=color_econ, linewidth=1.5, zorder=10)
            sns.lineplot(x=time, y=q_crit_values, ax=ax, label='Critical Functionality', color=color_crit, linewidth=1.5, zorder=10)
            sns.lineplot(x=time, y=q_health_values, ax=ax, label='Healthcare Functionality', color=color_health, linewidth=1.5, zorder=10)

        def plot_sub_econ_functionalities(ax):
            if plot_econ_income:
                sns.lineplot(x=time, y=q_income_values, ax=ax, label="Total Income", color='green', linewidth=0.75, zorder=10)
            if plot_econ_bldg_repair:
                sns.lineplot(x=time, y=q_econ_bldg_repair_costs, ax=ax, label='Building Repair Costs', color='darkgreen', linestyle=(0, (1, 3)), linewidth=0.75, zorder=10)
            if plot_econ_road_repair:
                sns.lineplot(x=time, y=q_econ_road_repair_costs, ax=ax, label='Road Repair Costs', color='darkorange', linestyle=(0, (1, 3)), linewidth=0.75, zorder=10)
            if plot_econ_traffic:
                sns.lineplot(x=time, y=q_econ_traffic_delay_costs, ax=ax, label='Traffic Delay Costs', color='dimgray', linestyle=(0, (3, 1, 1, 1)), linewidth=0.75, zorder=10)
            if plot_econ_relocation:
                sns.lineplot(x=time, y=q_econ_relocation_costs, ax=ax, label='Relocation Costs', color='rosybrown', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=0.75, zorder=10)

        plot_sub_functionalities(ax_main)
        plot_sub_econ_functionalities(ax_main)
        sns.lineplot(x=time, y=q_community_values, ax=ax_main, label='Community Functionality', linewidth=2.5, color=color_community, zorder=10)

        def plot_markers():
            # Marker positioning
            repairs_y = 1.1
            debris_y = repairs_y + 0.05
            func_y = repairs_y + 0.1
            road_repair_y = repairs_y + 0.2
            # Track which labels have been used to avoid duplicates
            plotted_labels = {
                'Repairs Completed': False,
                'Debris Cleared': False,
                'Functional Restoration': False,
                'Road Repairs': False
            }

            # Repairs Completed
            for t, count in zip(time[padding_length:], bldg_repairs_counts):
                if rollout_start <= t <= rollout_end and count > 0:
                    size = 50 + 10 * count
                    ax_main.scatter(
                        t, repairs_y,
                        s=size,
                        marker='>',
                        color='slateblue',
                        edgecolor='black',
                        alpha=0.7,
                        label='Repairs Completed' if not plotted_labels['Repairs Completed'] else "",
                        zorder=15
                    )
                    plotted_labels['Repairs Completed'] = True

            # Debris Cleared
            for t, count in zip(time[padding_length:], debris_clear_counts):
                if rollout_start <= t <= rollout_end and count > 0:
                    size = 50 + 10 * count
                    ax_main.scatter(
                        t, debris_y,
                        s=size,
                        marker=10,
                        color='darkmagenta',
                        edgecolor='black',
                        alpha=0.7,
                        label='Debris Cleared' if not plotted_labels['Debris Cleared'] else "",
                        zorder=14
                    )
                    plotted_labels['Debris Cleared'] = True

            # Functional Restoration
            for t, count in zip(time[padding_length:], func_rest_counts):
                if rollout_start <= t <= rollout_end and count > 0:
                    size = 50 + 10 * count
                    ax_main.scatter(
                        t, func_y,
                        s=size,
                        marker='d',
                        color='green',
                        edgecolor='black',
                        alpha=0.7,
                        label='Functional Restoration' if not plotted_labels['Functional Restoration'] else "",
                        zorder=13
                    )
                    plotted_labels['Functional Restoration'] = True

            # Road Repairs
            for t, count in zip(time[padding_length:], road_repair_counts):
                if rollout_start <= t <= rollout_end and count > 0:
                    size = 50 + 10 * count
                    ax_main.scatter(
                        t, road_repair_y,
                        s=size,
                        marker='4',
                        color='green',
                        edgecolor='black',
                        alpha=0.7,
                        label='Road Repairs' if not plotted_labels['Road Repairs'] else "",
                        zorder=13
                    )
                    plotted_labels['Road Repairs'] = True

        plot_markers()

        def plot_reward():
            # Split time and returns
            time_np = np.array(time)
            returns_np = np.array(returns)
            returns_econ_np = np.array(returns_econ)
            returns_crit_np = np.array(returns_crit)
            returns_health_np = np.array(returns_health)
            post_zero_mask = (time_np >= 0) & (time_np <= rollout_end)

            # Plot post-zero (solid, darkorchid)
            sns.lineplot(
                x=time_np[post_zero_mask],
                y=returns_np[post_zero_mask],
                ax=ax_rewards,
                label="Instantaneous Agent Rewards",
                color=color_community,
                linewidth=2.0,
                zorder=20
            )
            sns.lineplot(
                x=time_np[post_zero_mask],
                y=returns_econ_np[post_zero_mask],
                ax=ax_rewards,
                label="Economic Reward Component",
                color=color_econ,
                linewidth=0.5,
                zorder=10
            )
            sns.lineplot(
                x=time_np[post_zero_mask],
                y=returns_health_np[post_zero_mask],
                ax=ax_rewards,
                label="Critical Reward Component",
                color=color_health,
                linewidth=0.5,
                zorder=10
            )
            sns.lineplot(
                x=time_np[post_zero_mask],
                y=returns_crit_np[post_zero_mask],
                ax=ax_rewards,
                label="Health Reward Component",
                color=color_crit,
                linewidth=0.5,
                zorder=10
            )
            # Final touches
            ax_rewards.set_ylabel("Instantaneous Agent Rewards")
            ax_rewards.set_xlabel(f'Time Step / {self.time_step_duration} days')
            ax_rewards.legend(loc='upper right')
            ax_rewards.set_ylim(returns_np.min() - 0.1 * returns_np.max(), returns_np.max() + 0.1 * returns_np.max())

        plot_reward()
        # Legend setup
        repairs_marker = plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='slateblue', markersize=10, label='Repairs Completed', markeredgewidth=2)
        debris_marker = plt.Line2D([0], [0], marker=10, color='w', markerfacecolor='darkorange', markersize=10, label='Debris Cleared', markeredgewidth=2)
        func_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Functional Restoration', markeredgewidth=2)
        handles, labels = ax_main.get_legend_handles_labels()
        handles += [repairs_marker, debris_marker, func_marker]
        labels += ['Repairs Completed', 'Debris Cleared', 'Functional Restoration']
        by_label = dict(zip(labels, handles))
        ax_main.legend(by_label.values(), by_label.keys(), loc='lower right', ncol=2)

        # Mark rollout start and end with vertical dashed lines
        ax_rewards.axvline(x=rollout_start, color='lightseagreen', linestyle='--', linewidth=1.0, label='Rollout Start')
        ax_rewards.axvline(x=rollout_end, color='maroon', linestyle='--', linewidth=1.0, label='Rollout End')

        # Optional: Add legend entry if you want to label these
        # Only do this if you have space for the legend and want the lines explained
        # If the main plot already has a legend, you might want to skip this to avoid clutter
        ax_rewards.legend(loc='upper right')  # or 'best', or wherever looks best in your case
        ax_main.set_ylabel("Aggregate Functionality Metrics")
        fig.suptitle(
            r"$\bf{Earthquake\ Repair\ Scheduling\ Rollout}$" + "\n"
            + r"$\it{toy\text{-}city\text{-}" + str(self.num_components) + r"}$" + "\n"
            + f"Earthquake Magnitude: {self.eq_magnitude}",
            fontsize=14,
            y=0.98,
            fontfamily='serif'  # or 'sans-serif', 'monospace', or a specific one like 'DejaVu Sans'
        )
        plt.tight_layout()
        plt.show()
