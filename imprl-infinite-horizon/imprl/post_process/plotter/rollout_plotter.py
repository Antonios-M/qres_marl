import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from quake_envs.simulations.earthquake_funcs import DamageStates
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.patches as patches
from scipy.interpolate import griddata
from quake_envs.qres_env_wrapper import Qres_env_wrapper
import json
import os
def convert_numpy(obj):
    """
    Recursively convert NumPy data types to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

def save_dict_to_json(data_dict: dict, file_path: str) -> bool:
    """
    Saves a dictionary to a new JSON file at the specified path.
    """
    try:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        print(f"Saving data to {file_path}...")
        # Convert all NumPy types to native types
        cleaned_data = convert_numpy(data_dict)

        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(cleaned_data, json_file, indent=4, ensure_ascii=False)

        print(f"Successfully saved data to {file_path}")
        return True

    except TypeError as e:
        print(f"Error: Could not serialize the dictionary. Check for non-standard data types.")
        print(f"Details: {e}")
        return False
    except IOError as e:
        print(f"Error: Could not write to the file at {file_path}.")
        print(f"Details: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False


class BaselineImportanceBased:
    def __init__(self, env):
        self.env = env
        self.buildings = env.resilience.buildings_objs
        self.roads = env.resilience.road_objs
        self.n_crews = env.n_crews

    def get_road_value(
        self,
        capacity: float,
        damage_state: int,
        max_capacity: float,
        max_damage_state: int,
    ) -> float:
        return (capacity * damage_state) / (max_capacity * max_damage_state)

    def get_building_value(
        self,
        undisturbed_income: float,
        nominal_income: float,
        sqft: float,
        nominal_sqft: float,
        is_essential: bool,
        damage_state: int,
        w_econ: float = 0.2,
        w_crit: float = 0.5
    ) -> float:
        income_value = max(0.2, undisturbed_income / nominal_income)
        ds_value = damage_state / len(DamageStates)
        sqft_value = sqft / nominal_sqft
        dmg_value = (ds_value + sqft_value) / 2
        essential_value = 1 if is_essential else 0.5

        value = dmg_value * (w_econ * income_value + w_crit * essential_value)
        return value

    def update_road_values(self):
        max_capacity = max([road.capacity for road in self.roads])
        for road in self.roads:
            value = self.get_road_value(
                capacity=road.capacity,
                damage_state=road.current_damage_state,
                max_capacity=max_capacity,
                max_damage_state=max([ds.value for ds in DamageStates])
            )
            road.value = value

    def update_building_value(self):
        nominal_income = max([building.max_income for building in self.buildings])
        nominal_sqft = max([building.sqft for building in self.buildings])
        for building in self.buildings:
            value = self.get_building_value(
                undisturbed_income=building.max_income,
                nominal_income=nominal_income,
                sqft=building.sqft,
                nominal_sqft=nominal_sqft,
                is_essential=building.is_essential,
                damage_state=building.current_damage_state,
            )
            building.value = value

    def step(self, env):
        self.buildings = env.resilience.buildings_objs
        self.roads = env.resilience.road_objs
        self.update_road_values()
        self.update_building_value()

    def select_action(self, env):
        self.step(env)

        action = [0] * env.n_agents  # Initialize all actions to "do nothing"
        crews_used = 0

        # Combine buildings and roads into one list with type tag and index
        combined_assets = []

        for i, building in enumerate(self.buildings):
            combined_assets.append(('building', i, building.value))

        for i, road in enumerate(self.roads):
            combined_assets.append(('road', i, road.value))

        # Sort combined assets by value descending
        combined_assets.sort(key=lambda x: -x[2])

        # Assign actions based on sorted priorities
        for asset_type, idx, _ in combined_assets:
            if crews_used >= self.n_crews:
                break

            if asset_type == 'building':
                if self.buildings[idx].has_debris:
                    action[idx] = 1  # clear debris
                    crews_used += 1
                elif not self.buildings[idx].is_fully_repaired:
                    action[idx] = 1  # repair
                    crews_used += 1
            elif asset_type == 'road':
                if not self.roads[idx].is_fully_repaired:
                    action[len(self.buildings) + idx] = 1  # repair
                    crews_used += 1

        return action


class AgentPlotter:
    def __init__(self, env: Qres_env_wrapper, agent, name):
        self.name = name
        self.env = env
        self.agent = agent
        self.time_horizon = env.time_horizon
        self.rollout_dict = {}

        self.importance_based = BaselineImportanceBased(env)

    def select_action(self, obs):
        if self.agent not in ['random', 'importance_based', "do_nothing"]:
            # print(f"Agent {self.agent}")
            action = self.agent.select_action(obs, training=False)
            # action = tuple(action)
        elif self.agent == "random":
            action = self.env.action_space.sample()
            # action = np.array(action)
            # action[-15:] = 1
        elif self.agent == "do_nothing":
            action = [0] * self.env.n_agents
        else:
            action = self.importance_based.select_action(self.env)

        return action

    def reset_plot_data(self):
        self.returns = []
        self.returns_econ = []
        self.returns_crit = []
        self.returns_health = []
        self.q_community_values = []
        self.q_econ_values = []
        self.q_income_values = []
        self.q_econ_bldg_repair_costs = []
        self.q_econ_road_repair_costs = []
        self.q_econ_traffic_delay_costs = []
        self.q_econ_relocation_costs = []
        self.q_crit_values = []
        self.q_health_values = []
        self.bldg_repairs_counts = []
        self.debris_clear_counts = []
        self.func_rest_counts = []
        self.road_repair_counts = []
        self.loss_values = []
        self.resilience_values = []

        self.time = []

    def add_mobilisation(self, delay_time):
        self.returns = [self.returns[0]] * delay_time + self.returns
        self.returns_econ = [self.returns_econ[0]] * delay_time + self.returns_econ
        self.returns_crit = [self.returns_crit[0]] * delay_time + self.returns_crit
        self.returns_health = [self.returns_health[0]] * delay_time + self.returns_health
        self.q_community_values = [self.q_community_values[0]] * delay_time + self.q_community_values
        self.q_econ_values = [self.q_econ_values[0]] * delay_time + self.q_econ_values
        self.q_income_values = [self.q_income_values[0]] * delay_time + self.q_income_values
        self.q_econ_bldg_repair_costs = [self.q_econ_bldg_repair_costs[0]] * delay_time + self.q_econ_bldg_repair_costs
        self.q_econ_road_repair_costs = [self.q_econ_road_repair_costs[0]] * delay_time + self.q_econ_road_repair_costs
        self.q_econ_traffic_delay_costs = [self.q_econ_traffic_delay_costs[0]] * delay_time + self.q_econ_traffic_delay_costs
        self.q_econ_relocation_costs = [self.q_econ_relocation_costs[0]] * delay_time + self.q_econ_relocation_costs
        self.q_crit_values = [self.q_crit_values[0]] * delay_time + self.q_crit_values
        self.q_health_values = [self.q_health_values[0]] * delay_time + self.q_health_values
        self.bldg_repairs_counts = [0] * delay_time + self.bldg_repairs_counts
        self.debris_clear_counts = [0] * delay_time + self.debris_clear_counts
        self.func_rest_counts = [0] * delay_time + self.func_rest_counts
        self.road_repair_counts = [0] * delay_time + self.road_repair_counts
        self.time = list(range(-delay_time, self.time[0])) + self.time

    def add_pre_disaster_data(self, padding_length):
        w_econ = self.env.w_econ
        w_crit = self.env.w_crit
        w_health = self.env.w_health

        self.returns = [self.returns[0]] * padding_length + self.returns
        self.returns_econ = [0.0] * padding_length + self.returns_econ
        self.returns_crit = [0.0] * padding_length + self.returns_crit
        self.returns_health = [0.0] * padding_length + self.returns_health
        self.q_community_values = [1.0] * padding_length + self.q_community_values
        self.q_econ_values = [w_econ] * padding_length + self.q_econ_values
        self.q_crit_values = [w_crit] * padding_length + self.q_crit_values
        self.q_health_values = [w_health] * padding_length + self.q_health_values
        self.q_income_values = [1.0] * padding_length + self.q_income_values
        self.q_econ_bldg_repair_costs = [1.0] * padding_length + self.q_econ_bldg_repair_costs
        self.q_econ_road_repair_costs = [1.0] * padding_length + self.q_econ_road_repair_costs
        self.q_econ_traffic_delay_costs = [0.0] * padding_length + self.q_econ_traffic_delay_costs
        self.q_econ_relocation_costs = [0.0] * padding_length + self.q_econ_relocation_costs
        self.bldg_repairs_counts = [0] * padding_length + self.bldg_repairs_counts
        self.debris_clear_counts = [0] * padding_length + self.debris_clear_counts
        self.func_rest_counts = [0] * padding_length + self.func_rest_counts
        self.road_repair_counts = [0] * padding_length + self.road_repair_counts

        self.t_start = -padding_length + self.time[0]
        self.t_quake = self.time[0]
        self.time = list(range(-padding_length + self.time[0], self.time[0])) + self.time

    def add_post_rollout_data(self, padding_length):
        self.returns = self.returns + [self.returns[-1]] * padding_length
        self.returns_econ = self.returns_econ + [self.returns_econ[-1]] * padding_length
        self.returns_crit = self.returns_crit + [self.returns_crit[-1]] * padding_length
        self.returns_health = self.returns_health + [self.returns_health[-1]] * padding_length
        self.q_community_values = self.q_community_values + [self.q_community_values[-1]] * padding_length
        self.q_econ_values = self.q_econ_values + [self.q_econ_values[-1]] * padding_length
        self.q_income_values = self.q_income_values + [self.q_income_values[-1]] * padding_length
        self.q_econ_bldg_repair_costs = self.q_econ_bldg_repair_costs + [self.q_econ_bldg_repair_costs[-1]] * padding_length
        self.q_econ_road_repair_costs = self.q_econ_road_repair_costs + [self.q_econ_road_repair_costs[-1]] * padding_length
        self.q_econ_traffic_delay_costs = self.q_econ_traffic_delay_costs + [self.q_econ_traffic_delay_costs[-1]] * padding_length
        self.q_econ_relocation_costs = self.q_econ_relocation_costs + [self.q_econ_relocation_costs[-1]] * padding_length
        self.q_crit_values = self.q_crit_values + [self.q_crit_values[-1]] * padding_length
        self.q_health_values = self.q_health_values + [self.q_health_values[-1]] * padding_length
        self.bldg_repairs_counts = self.bldg_repairs_counts + [0] * padding_length
        self.debris_clear_counts = self.debris_clear_counts + [0] * padding_length
        self.func_rest_counts = self.func_rest_counts + [0] * padding_length
        self.road_repair_counts = self.road_repair_counts + [0] * padding_length
        self.time += list(range(self.time[-1] + 1, self.time[-1] + padding_length + 1))

    def get_component_data(self, info, reset=False):
        if reset:
            self.info_buildings = {}
            self.info_roads = {}
        brts, bdss, brcs, bincs, brelocs, actions = self.env.resilience.get_building_info()
        self.info_buildings[self.env.resilience.time] = {
            "repair_times": brts,
            "damage_states": bdss,
            "repair_costs": brcs,
            "incomes": bincs,
            "relocation_costs": brelocs,
            "debris_clears": info["completions"]["bldg_debris"],
            "functionality_restores": info["completions"]["bldg_funcs"],
            "repairs": info["completions"]["bldg_repairs"],
            "actions": actions
        }
        ## Road Info
        rrts, rdss, rrcs, rcrds, actions = self.env.resilience.get_road_info()
        self.info_roads[self.env.resilience.time] = {
            "repair_times": rrts,
            "damage_states": rdss,
            "repair_costs": rrcs,
            "capacity_reductions": rcrds,
            "repairs": info["completions"]["road_repairs"],
            "actions": actions
        }

    def add_data(self, info):
        reward = info["reward"]["total"]
        reward_econ = info["reward"]["econ"]
        reward_crit = info["reward"]["crit"]
        reward_health = info["reward"]["health"]
        q_community = info["q"]["community"]
        q_econ = info["q"]["econ"]
        q_crit = info["q"]["crit"]
        q_health = info["q"]["health"]
        q_econ_bldg_income = info["q_econ_components"]["income"]
        q_econ_bldg_repair = info["q_econ_components"]["bldg"]
        q_econ_road_repair = info["q_econ_components"]["road"]
        q_econ_traffic = info["q_econ_components"]["traffic"]
        q_econ_bldg_relocation = info["q_econ_components"]["reloc"]
        repairs_count = sum(info["completions"]["bldg_repairs"])
        road_repair_count = sum(info["completions"]["road_repairs"])
        debris_clear_count = sum(info["completions"]["bldg_debris"])
        func_rest_count = sum(info["completions"]["bldg_funcs"])
        resilience_loss = info["reward"]["loss"]
        resilience = info["reward"]["resilience"]


        self.returns.append(reward)
        self.returns_econ.append(reward_econ)
        self.returns_crit.append(reward_crit)
        self.returns_health.append(reward_health)
        self.q_community_values.append(q_community)
        self.q_econ_values.append(q_econ)
        self.q_econ_bldg_repair_costs.append(q_econ_bldg_repair)
        self.q_income_values.append(q_econ_bldg_income)
        self.q_econ_road_repair_costs.append(q_econ_road_repair)
        self.q_econ_traffic_delay_costs.append(q_econ_traffic)
        self.q_econ_relocation_costs.append(q_econ_bldg_relocation)
        self.q_crit_values.append(q_crit)
        self.q_health_values.append(q_health)
        self.bldg_repairs_counts.append(repairs_count)
        self.road_repair_counts.append(road_repair_count)
        self.debris_clear_counts.append(debris_clear_count)
        self.func_rest_counts.append(func_rest_count)
        self.loss_values.append(resilience_loss)
        # print(f"Resilience Loss: {resilience_loss}, Resilience: {resilience}")
        self.resilience_values.append(resilience)

        self.time.append(self.env.resilience.time)

    def get_sample_rollout(self, save=False):
        self.rollout_dict = {}
        self.reset_plot_data()
        padding_length = int(0.05 * self.env.time_horizon)
        # Reset environment and set up padding
        obs, info = self.env.reset()
        # print(r.current_repair_cost for r in self.env.resilience.road_objs)
        self.rollout_dict["reset_observations"] = obs
        self.rollout_dict["time_step_duration"] = self.env.time_step_duration
        # self.quake_mag = self.env.eq_magnitude
        # print(info)
        # print(f"Community Robustness: {info['q']['community_robustness']}")
        self.add_data(info)
        self.get_component_data(info, reset=True)
        delay_time = self.env.resilience.delay_time
        has_terminated, has_truncated = False, False

        def has_rollout_ended(term, trunc):
            return trunc or term

        rollout_start = 0
        rollout_end = 0
        while not has_rollout_ended(has_terminated, has_truncated):
            rollout_end += 1

            action = self.select_action(obs)
            next_obs, cost, has_terminated, has_truncated, info = self.env.step(action)

            if save:
                self.rollout_dict[self.env.resilience.time] = {
                    "observations": list(obs),
                    "actions": list(info["actions"]),
                    "rewards": float(cost)
                }

            self.get_component_data(info)
            self.add_data(info)

            obs = next_obs

        if save:
            directory = "tests/results"
            filename = f"{str(self.env.n_agents)}_{str(self.name)}.json"
            path = directory + "/" + filename
            save_dict_to_json(self.rollout_dict, path)
        self.padding_length = max(1, int(0.1 * self.env.resilience.time))
        self.t_end = self.env.resilience.time + self.padding_length
        if has_terminated:
            self.t_repair_end = self.env.resilience.time
        else:  # has_truncated
            self.t_repair_end = None
        # print("returns health")
        # print(self.returns_health)
        self.t_repair_end = self.env.resilience.time
        self.add_mobilisation(delay_time)
        self.add_pre_disaster_data(self.padding_length)
        self.add_post_rollout_data(self.padding_length)

        def trim_trailing_duplicates(values):
            if not values:
                return values
            trimmed = values[:]
            last = trimmed[-1]
            i = len(trimmed) - 2
            while i >= 0 and trimmed[i] == last:
                i -= 1
            return trimmed[:i + 2]

        # Apply filtering
        filtered_resilience = trim_trailing_duplicates(self.resilience_values)
        filtered_loss = trim_trailing_duplicates(self.loss_values)

        self.resilience_values = filtered_resilience
        self.loss_values = filtered_loss

    def get_n_rollouts(self, n=100):
        all_returns = []
        all_losses = []
        all_resilience = []
        # all_rps = []
        all_mags = []
        for _ in range(n):
            self.get_sample_rollout()
            all_returns.append(self.returns)
            all_losses.append(self.loss_values)
            all_resilience.append(self.resilience_values)
            # all_rps.append(self.rp)
            all_mags.append(self.env.eq_magnitude)
        self.batch_returns = all_returns
        self.batch_losses = all_losses
        self.batch_resilience = all_resilience
        # self.batch_return_periods = all_rps
        self.batch_mags = all_mags


    def plot_avoided_losses_matrix(self, n=100, figsize=(10, 10)):
        all_mags = self.batch_mags


    def plot_returns_3d(self, figsize=(10, 10)):
        all_returns = self.batch_returns
        n = len(all_returns)

        # Pad sequences to equal length
        max_len = max(len(r) for r in all_returns)
        all_returns = [
            np.pad(r, (0, max_len - len(r)), mode='edge')
            for r in all_returns
        ]
        all_returns = np.array(all_returns)

        # Sort by first reward value
        sorted_indices = np.argsort(all_returns[:, 0])
        all_returns = all_returns[sorted_indices]

        # Meshgrid
        X = np.arange(all_returns.shape[1])
        Y = np.arange(all_returns.shape[0])
        X, Y = np.meshgrid(X, Y)
        Z = all_returns

        # Summary stats
        total_returns = np.sum(all_returns, axis=1)
        mean_total = np.mean(total_returns)
        std_total = np.std(total_returns)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Wireframe surface
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='lightblue', alpha=0.5, linewidth=0.4)

        # Plot grey return lines and add markers
        for i, ret in enumerate(all_returns):
            timesteps = np.arange(len(ret))

            # Plot faint line
            ax.plot(timesteps, [i]*len(ret), ret, color='gray', linewidth=0.6, alpha=0.7)

            # --- Detect flatline end ---
            flat_value = ret[0]
            flat_end_idx = np.argmax(ret != flat_value)
            if flat_end_idx == 0 and ret[0] != ret[1]:
                flat_end_idx = 0  # Edge case: first value already different

            # --- Detect max value index ---
            max_idx = np.argmax(ret)

            # Plot flatline-end marker
            ax.scatter(flat_end_idx, i, ret[flat_end_idx], color='red', s=20, label='Flatline End' if i == 0 else "")

            # Plot max marker
            ax.scatter(max_idx, i, ret[max_idx], color='green', s=20, label='Max Reward' if i == 0 else "")

        # Final reward line
        last_timestep = all_returns.shape[1] - 1
        final_rewards = all_returns[:, last_timestep]
        rollout_indices = np.arange(all_returns.shape[0])
        ax.plot3D([last_timestep] * n, rollout_indices, final_rewards, color='black', linewidth=2, label='Final Rewards')

        # Axes and view
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Rollout (sorted by first reward)')
        ax.set_zlabel('Reward')
        ax.set_title(f'Reward Over {n} Rollouts')
        ax.view_init(elev=30, azim=-45)

        # Stats annotation
        legend_text = f"Mean Total Return: {mean_total:.2f}\nStd Dev: {std_total:.2f}"
        ax.text2D(0.05, 0.95, legend_text, transform=ax.transAxes)

        # Single legend entry for each type
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

        plt.tight_layout()
        plt.show()

    def plot_avoided_losses_stats(
        random=None,
        do_nothing=None,
        importance_based=None,
        trained_agent=None,
        bins=50,
        hist_color='skyblue',
        hist_edgecolor='black',
        figsize=(12, 7),
        plot_title="Distribution of Avoided Loss Ratio",
        xlabel="Avoided Loss Ratio (Resilience / (Losses + Resilience))",
        ylabel="Frequency",
        save_path=None
    ):
        """
        Plots avoided loss ratio distributions for multiple policies.

        Each policy input should be a tuple of (losses_list, resilience_list),
        where each is a list of per-rollout values.

        Args:
            random, do_nothing, importance_based, trained_agent: Optional[Tuple[List[List[float]], List[List[float]]]]
                Each policy's data as (losses, resilience).
            Other arguments control plot style.
        """

        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        sns.set_style("whitegrid")
        plt.figure(figsize=figsize)

        policy_data = {
            "Random": random,
            "Do Nothing": do_nothing,
            "Importance-Based": importance_based,
            "Trained Agent": trained_agent
        }

        for policy_name, data in policy_data.items():
            if data is None:
                continue

            losses, resilience = data

            if len(losses) != len(resilience):
                print(f"Warning: '{policy_name}' data has mismatched lengths. Skipping.")
                continue

            avoided_ratios = [
                res_sum / (loss_sum + res_sum)
                for loss_sum, res_sum in zip(map(sum, losses), map(sum, resilience))
                if (loss_sum + res_sum) > 0
            ]

            if not avoided_ratios:
                print(f"Warning: No valid avoided loss ratios for '{policy_name}'. Skipping.")
                continue

            sns.histplot(
                avoided_ratios,
                bins=bins,
                label=f"{policy_name} (N={len(avoided_ratios)})",
                stat="count",
                kde=True,
                edgecolor=hist_edgecolor,
                alpha=0.6
            )

        plt.title(plot_title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=300)
                print(f"Plot saved to {save_path}")
            except Exception as e:
                print(f"Error saving plot: {e}")

        plt.show()

    def run_n_steps(self, n=100):
        obs, info = self.env.reset()
        print(f"Reset Observations: {obs}")
        for i in range(n):
            print("----------------------------------------")
            print(f"Step {i}")
            actions = np.random.randint(0, 2, size=self.env.n_agents)
            print(f"Actions: {actions}")
            obs, reward, term, trunc, info = self.env.step(actions)
            print(f"Observation: {obs}, Reward: {reward}, Term: {term}, Trunc: {trunc}")
            print(f"Building Damage States: {[b.current_damage_state for b in self.env.resilience.buildings_objs]}")
            print(f"Road Damage States: {[r.current_damage_state for r in self.env.resilience.road_objs]}")
            print(f"Road Capacity Reductions: {[r.capacity_reduction for r in self.env.resilience.road_objs]}")
            print(f"Building Debris: {[b.has_debris for b in self.env.resilience.buildings_objs]}")
            if term or trunc:
                break

    def plot_components(self, figsize=(20, 20)):
        def normalize_to_range_0_n(values, n=4, min_val=None):
            # min_val = min(values) if values else 0
            max_val = max(values) if values else 1
            if max_val == min_val:
                return [0 for _ in values]
            return [n * (v - min_val) / (max_val - min_val) for v in values]
        plot_marker_size = 3
        road_data = {
            i: {
                'timesteps': [],
                'repair_times': [],
                'damage_states': [],
                'repair_costs': [],
                'capacity_reductions': [],
                "repair_completed": [],
                "actions": []
            }
            for i, r in enumerate(self.env.resilience.road_objs)
        }

        building_data = {
            i: {
                'timesteps': [],
                'repair_times': [],
                'damage_states': [],
                'repair_costs': [],
                'incomes': [],
                'relocation_costs': [],
                "debris_clears": [],
                "functionality_restored": [],
                "repair_completed": [],
                "actions": []
            }
            for i, b in enumerate(self.env.resilience.buildings_objs)
        }

        for timestep, attrs in self.info_buildings.items():
            rts = attrs['repair_times']
            dss = attrs['damage_states']
            max_ds = max(dss)
            rcs = attrs['repair_costs']
            incs = attrs['incomes']
            relocs = attrs['relocation_costs']
            func_restored = attrs['functionality_restores']
            repair_completed = attrs['repairs']
            debris_cleared = attrs['debris_clears']
            actions = attrs['actions']

            for i, b in enumerate(self.env.resilience.buildings_objs):
                building_data[i]['timesteps'].append(timestep)
                building_data[i]['debris_clears'].append(debris_cleared[i])
                building_data[i]['repair_times'].append(rts[i])
                building_data[i]['damage_states'].append(dss[i])
                building_data[i]['repair_costs'].append(rcs[i])
                building_data[i]['incomes'].append(incs[i])
                building_data[i]['relocation_costs'].append(relocs[i])
                building_data[i]['functionality_restored'].append(func_restored[i])
                building_data[i]['repair_completed'].append(repair_completed[i])
                building_data[i]['actions'].append(actions[i])


        for timestep, attrs in self.info_roads.items():
            rts = attrs['repair_times']
            dss = attrs['damage_states']
            max_ds = max(dss)
            rcs = attrs['repair_costs']
            crds = attrs['capacity_reductions']
            actions = attrs['actions']
            for i, r in enumerate(self.env.resilience.road_objs):
                road_data[i]['timesteps'].append(timestep)
                road_data[i]['repair_times'].append(rts[i])
                road_data[i]['damage_states'].append(dss[i])
                road_data[i]['repair_costs'].append(rcs[i])
                road_data[i]['capacity_reductions'].append(crds[i])
                road_data[i]['repair_completed'].append(attrs['repairs'][i])
                road_data[i]['actions'].append(actions[i])

        n_buildings = len(building_data)
        n_roads = len(road_data)
        cols = int((n_buildings + n_roads) / 2)
        rows = math.ceil((n_buildings + n_roads) / cols)
        if cols > 5:
            cols = 5
            rows = math.ceil((n_buildings + n_roads) / cols)

        sns.set_theme(style="whitegrid", context="notebook")
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        axes = axes.flatten()

        for i, b in enumerate(self.env.resilience.buildings_objs):
            ax = axes[i]

            timesteps = building_data[i]['timesteps']
            ds = building_data[i]['damage_states']
            ds_max = max(ds) if ds else 1

            rt = normalize_to_range_0_n(building_data[i]['repair_times'], ds_max, min_val=0)
            rc = normalize_to_range_0_n(building_data[i]['repair_costs'], ds_max, min_val=0)
            inc = normalize_to_range_0_n(building_data[i]['incomes'], ds_max, min_val=min(building_data[i]['incomes']))
            reloc = normalize_to_range_0_n(building_data[i]['relocation_costs'], ds_max, min_val=0)
            debris = building_data[i]['debris_clears']

            if sum(debris) > 1:
                raise ValueError("Debris should be cleared once")
            else:
                t_debris = -1
                for t, d in enumerate(debris):
                    if d > 0:
                        t_debris = t

            repair_completed = building_data[i]['repair_completed']
            if sum(repair_completed) > 1:
                raise ValueError("Repair should be completed once")
            else:
                t_repair = -1
                for t, r in enumerate(repair_completed):
                    if r > 0:
                        t_repair = t

            func_restored = building_data[i]['functionality_restored']
            if sum(func_restored) > 1:
                raise ValueError("Functionality should be restored once")
            else:
                t_func = -1
                for t, f in enumerate(func_restored):
                    if f > 0:
                        t_func = t

            if t_debris != -1:
                ax.axvspan(xmin=timesteps[0], xmax=t_debris, color='orange', alpha=0.1, label='_nolegend_')  # Pre-debris clearance
                ax.axvline(x=t_debris, color='orange', linestyle='--', linewidth=1, label='Debris Cleared')

            if t_debris != t_repair:
                ax.axvspan(xmin=max(t_debris, 0), xmax=t_repair, color='red', alpha=0.1, label='_nolegend_')  # Between debris clearance and repair

            if t_func != -1 and t_repair != -1:
                ax.axvspan(xmin=t_repair, xmax=timesteps[-1], color='green', alpha=0.1, label='_nolegend_')  # Between repair and functionality restoration
            else:
                ax.axvspan(xmin=timesteps[0], xmax=timesteps[-1], color='red', alpha=0.1, label='_nolegend_')  # Between repair and functionality restoration

            ax.plot(timesteps, rt, label='Repair Time', linestyle='-', marker='o', markersize=3)
            ax.plot(timesteps, rc, label='Repair Cost', linestyle='--', marker='^', markersize=plot_marker_size)
            ax.plot(timesteps, ds, label='Damage State', linestyle='-.', marker='s', markersize=plot_marker_size)
            ax.plot(timesteps, inc, label='Income', linestyle=':', marker='d', markersize=plot_marker_size)
            ax.plot(timesteps, reloc, label='Relocation Cost', linestyle='-', marker='x', markersize=plot_marker_size)

            ## markers
            actions = building_data[i]['actions']
            for t, a in enumerate(actions):
                if a == 1:
                    ax.scatter(t, 5, s=30, marker='>', color='slateblue', label='Repair Action' if t == 0 else "", zorder=50)
                else:
                    ax.scatter(t, 5, s=30, marker='o', color='gray', label='Do Nothing' if t == 0 else "", zorder=50)

            rect = patches.Rectangle((0, 4.75), timesteps[-1], 0.5, linewidth=1, edgecolor='black', facecolor='white', zorder=25)
            ax.add_patch(rect)

            ax.set_title(f"Building {i}")
            ax.set_ylim(0, 5.5)
            # ax.set_xlabel("Timestep")
            # ax.set_ylabel("Normalized Metric")

            # if i == 0:
            #     ax.legend(fontsize='small')

        for j, r in enumerate(self.env.resilience.road_objs):
            ax = axes[j + n_buildings]

            timesteps = road_data[j]['timesteps']
            ds = road_data[j]['damage_states']
            ds_max = max(ds) if ds else 1

            rt = normalize_to_range_0_n(road_data[j]['repair_times'], ds_max, min_val=0)
            rc = normalize_to_range_0_n(road_data[j]['repair_costs'], ds_max, min_val=0)
            crd = normalize_to_range_0_n(road_data[j]['capacity_reductions'], ds_max, min_val=0)

            repair_completed = road_data[j]['repair_completed']
            if sum(repair_completed) > 1:
                raise ValueError("Repair should be completed once")
            else:
                t_repair = -1
                for t, _r in enumerate(repair_completed):
                    if _r > 0:
                        t_repair = t
            if t_repair != -1:
                ax.axvspan(xmin=timesteps[0], xmax=t_repair, color='red', alpha=0.1, label='_nolegend_')  # Pre-repair
            ## repaired
            ax.axvspan(xmin=t_repair, xmax=timesteps[-1], color='green', alpha=0.1, label='_nolegend_')  # Post-repair

            ax.plot(timesteps, rt, label='Repair Time', linestyle='-', marker='o', markersize=plot_marker_size)
            ax.plot(timesteps, rc, label='Repair Cost', linestyle='--', marker='^', markersize=plot_marker_size)
            ax.plot(timesteps, ds, label='Damage State', linestyle='-.', marker='s', markersize=plot_marker_size)
            ax.plot(timesteps, crd, label='Capacity Reduction', linestyle=':', marker='d', markersize=plot_marker_size)

            ## markers
            actions = road_data[j]['actions']
            for t, a in enumerate(actions):
                if a == 1:
                    ax.scatter(t, 5, s=30, marker='>', color='slateblue', label='Repair Action' if t == 0 else "", zorder=50)
                else:
                    ax.scatter(t, 5, s=30, marker='o', color='gray', label='Do Nothing' if t == 0 else "", zorder=50)

            rect = patches.Rectangle((0, 4.75), timesteps[-1], 0.5, linewidth=1, edgecolor='black', facecolor='white', zorder=25)
            ax.add_patch(rect)

            ax.set_title(f"Road {j}")
            ax.set_ylim(0, 5.5)
            # ax.set_xlabel("Timestep")
            # ax.set_ylabel("Normalized Metric")

            # if j == 0:
            #     ax.legend(fontsize='small')

        for j in range(n_buildings + n_roads, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=1.0)
        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        plt.show()

    def plot(
        self,
        policy: str = "",
        env_setting: str = "",
        plot_econ_income=False,
        plot_econ_bldg_repair=False,
        plot_econ_road_repair=False,
        plot_econ_traffic=False,
        plot_econ_relocation=False,
        plot_delay=False,
        figsize=(10, 6),
        title_intensity=None,
    ):

        # Set up Seaborn style
        sns.set_theme(style="ticks", palette="pastel")



        # Create figure with two vertically stacked subplots sharing the x-axis
        fig, (ax_main, ax_rewards) = plt.subplots(
            2, 1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1}
        )

        fig.patch.set_facecolor('white')
        ax_main.set_facecolor('white')
        ax_rewards.set_facecolor('mintcream')
        # Assuming self.time is your list of x values
        xticks = self.time  # ticks at every value
        xlabels = [str(x) if x % 5 == 0 else "" for x in self.time]  # label every 5th value, else blank
        ax_rewards.set_xticks(xticks)
        ax_rewards.set_xticklabels(xlabels)

        ax_main.set_xticks(xticks)
        ax_main.set_xticklabels(xlabels)

        # Background colors
        normal_op_color = 'forestgreen'
        normal_op_alpha = 0.1
        recovery_color = 'orange'
        mobilisation_color = 'red'
        recovery_alpha = 0.1

        # Green background before timestep 0
        ax_main.axvspan(xmin=self.t_start, xmax=self.t_quake-1, color=normal_op_color, alpha=normal_op_alpha, zorder=-10)

        ax_main.axvspan(self.t_quake-1, 0, color=mobilisation_color, alpha=recovery_alpha, zorder=-10)

        try:
            ax_main.axvspan(0, self.t_repair_end, color=recovery_color, alpha=normal_op_alpha, zorder=-10)
            ax_main.axvspan(self.t_repair_end, self.t_end, color=normal_op_color, alpha=normal_op_alpha, zorder=-10)
        except:
            ax_main.axvspan(0, self.t_end, color=recovery_color, alpha=normal_op_alpha, zorder=-10)

        color_community = "black"
        color_econ = "teal"
        color_crit = "darkorange"
        color_health = "crimson"
        # Functionality line plots
        def plot_sub_functionalities(ax):
            sns.lineplot(x=self.time, y=self.q_econ_values, ax=ax, label='Economic Functionality', color=color_econ, linewidth=1.5, zorder=10)
            sns.lineplot(x=self.time, y=self.q_crit_values, ax=ax, label='Critical Functionality', color=color_crit, linewidth=1.5, zorder=10)
            sns.lineplot(x=self.time, y=self.q_health_values, ax=ax, label='Healthcare Functionality', color=color_health, linewidth=1.5, zorder=10)

        def plot_sub_econ_functionalities(ax):
            if plot_econ_income:
                sns.lineplot(x=self.time, y=self.q_income_values, ax=ax, label="Total Income", color='green', linewidth=0.75, zorder=10)
            if plot_econ_bldg_repair:
                sns.lineplot(x=self.time, y=self.q_econ_bldg_repair_costs, ax=ax, label='Building Repair Costs', color='darkgreen', linestyle=(0, (1, 3)), linewidth=0.75, zorder=10)
            if plot_econ_road_repair:
                sns.lineplot(x=self.time, y=self.q_econ_road_repair_costs, ax=ax, label='Road Repair Costs', color='darkorange', linestyle=(0, (1, 3)), linewidth=0.75, zorder=10)
            if plot_econ_traffic:
                sns.lineplot(x=self.time, y=self.q_econ_traffic_delay_costs, ax=ax, label='Traffic Delay Costs', color='dimgray', linestyle=(0, (3, 1, 1, 1)), linewidth=0.75, zorder=10)
            if plot_econ_relocation:
                sns.lineplot(x=self.time, y=self.q_econ_relocation_costs, ax=ax, label='Relocation Costs', color='rosybrown', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=0.75, zorder=10)

        plot_sub_functionalities(ax_main)
        plot_sub_econ_functionalities(ax_main)
        sns.lineplot(x=self.time, y=self.q_community_values, ax=ax_main, label='Community Functionality', linewidth=2.5, color=color_community, zorder=10)

        def plot_markers():
            # Marker positioning
            debris_y = 1.3
            repairs_y = debris_y + 0.05
            func_y = repairs_y + 0.05
            road_repair_y = func_y + 0.05
            def plot_marker_box():
                try:
                    x_end = self.t_repair_end
                except AttributeError:
                    x_end = self.t_end

                y_bottom = debris_y - 0.05
                y_top = road_repair_y + 0.05
                height = y_top - y_bottom + 0.05

                # Draw the rectangle
                rect = patches.Rectangle(
                    (0, y_bottom),     # (x, y)
                    x_end,             # width
                    height,            # height
                    facecolor='none',
                    edgecolor='dimgrey',
                    linewidth=1.0,
                    zorder=10
                )

                ax_main.add_patch(rect)

            plot_marker_box()
            # Track which labels have been used to avoid duplicates
            plotted_labels = {
                'Repairs Completed': False,
                'Debris Cleared': False,
                'Functional Restoration': False,
                'Road Repairs': False
            }

            # Repairs Completed
            for t, count in zip(self.time, self.bldg_repairs_counts):
                if count > 0:
                    ax_main.plot([t, t], [-1.0, repairs_y], color='slateblue', linestyle='--', linewidth=0.5, alpha=0.3, zorder=12)
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
            for t, count in zip(self.time, self.debris_clear_counts):
                if count > 0:
                    ax_main.plot([t, t], [-1.0, debris_y], color='peru', linestyle='--', linewidth=0.5, alpha=0.3, zorder=12)
                    size = 50 + 10 * count
                    ax_main.scatter(
                        t, debris_y,
                        s=size,
                        marker=10,
                        color='peru',
                        edgecolor='black',
                        alpha=0.7,
                        label='Debris Cleared' if not plotted_labels['Debris Cleared'] else "",
                        zorder=14
                    )
                    plotted_labels['Debris Cleared'] = True

            # Functional Restoration
            for t, count in zip(self.time, self.func_rest_counts):
                if count > 0:
                    ax_main.plot([t, t], [-1.0, func_y], color='green', linestyle='--', linewidth=0.5, alpha=0.3, zorder=12)
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
            for t, count in zip(self.time, self.road_repair_counts):
                if count > 0:
                    ax_main.plot([t, t], [-1.0, road_repair_y], color='green', linestyle='--', linewidth=0.5, alpha=0.3, zorder=12)
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
            # print(self.returns)
            # Split time and returns
            time_np = np.array(self.time)
            returns_np = np.array(self.returns)
            returns_econ_np = np.array(self.returns_econ)
            returns_crit_np = np.array(self.returns_crit)
            returns_health_np = np.array(self.returns_health)
            self.reward_t_max = self.t_repair_end if self.t_repair_end is not None else self.t_end
            post_zero_mask = (time_np >= 0) & (time_np <= self.reward_t_max)
            # print(returns_np)

            # print(returns_crit_np[post_zero_mask])
            # print(f"length: {len(returns_crit_np[post_zero_mask])}")

            # Plot post-zero (solid, darkorchid)
            sns.lineplot(
                x=time_np[post_zero_mask],
                y=returns_np[post_zero_mask],
                ax=ax_rewards,
                label="Instantaneous Agent Rewards",
                color=color_community,
                linewidth=2.0,
                zorder=20,
                marker='o',
                markersize=5
            )
            # sns.lineplot(
            #     x=time_np[post_zero_mask],
            #     y=returns_econ_np[post_zero_mask],
            #     ax=ax_rewards,
            #     label="Economic Reward Component",
            #     color=color_econ,
            #     linewidth=0.5,
            #     zorder=10
            # )

            # sns.lineplot(
            #     x=time_np[post_zero_mask],
            #     y=returns_health_np[post_zero_mask],
            #     ax=ax_rewards,
            #     label="Critical Reward Component",
            #     color=color_health,
            #     linewidth=0.5,
            #     zorder=10
            # )

            # sns.lineplot(
            #     x=time_np[post_zero_mask],
            #     y=returns_crit_np[post_zero_mask],
            #     ax=ax_rewards,
            #     label="Health Reward Component",
            #     color=color_crit,
            #     linewidth=0.5,
            #     zorder=10
            # )
            # Set axis limits and labels
            reward_range = 0
            for r in [returns_np, returns_econ_np, returns_crit_np, returns_health_np]:
                if r.max() - r.min() > reward_range:
                    reward_range = r.max() - r.min()

            ax_rewards.set_xlabel(f'Time Step / {self.env.time_step_duration} days')
            ax_rewards.set_ylim(returns_np.min() - (0.1 * reward_range), returns_np.max() + (0.1 * reward_range))

            # Draw all spines to form bounding box
            for spine in ax_rewards.spines.values():
                spine.set_visible(True)

            # Move the y-axis spine to x=0 (inside the plot)
            ax_rewards.spines['left'].set_position(('data', 0))
            ax_rewards.yaxis.set_ticks_position('left')
            ax_rewards.yaxis.set_label_position('left')

            # Use a workaround to place the y-axis label OUTSIDE the plot
            # Use axes coordinates for consistent placement (0 = left, 1 = top)
            ax_rewards.set_ylabel("")  # Clear default label
            ax_rewards.annotate(
                "Instantaneous Agent Rewards",
                xy=(0, 0.5), xycoords='axes fraction',
                xytext=(-45, 0), textcoords='offset points',
                ha='center', va='center',
                rotation=90,
                fontsize=12
            )

        plot_reward()
        # Legend setup
        def legend_setup():
            repairs_marker = plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='slateblue', markersize=10, label='Repairs Completed', markeredgewidth=2)
            debris_marker = plt.Line2D([0], [0], marker=10, color='w', markerfacecolor='peru', markersize=10, label='Debris Cleared', markeredgewidth=2)
            func_marker = plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='green', markersize=10, label='Functional Restoration', markeredgewidth=2)
            handles, labels = ax_main.get_legend_handles_labels()
            handles += [repairs_marker, debris_marker, func_marker]
            labels += ['Repairs Completed', 'Debris Cleared', 'Functional Restoration']
            by_label = dict(zip(labels, handles))
            ax_main.legend(by_label.values(), by_label.keys(),
               loc='upper left', bbox_to_anchor=(1.05, 1.0),
               borderaxespad=0., ncol=1)
            fig.subplots_adjust(right=0.8)

            # Mark rollout start and end with vertical dashed lines
            ax_rewards.axvline(x=0, color='lightseagreen', linestyle='--', linewidth=1.0, label='Rollout Start')
            ax_rewards.axvline(x=self.reward_t_max, color='maroon', linestyle='--', linewidth=1.0, label='Rollout End')

            # Optional: Add legend entry if you want to label these
            # Only do this if you have space for the legend and want the lines explained
            # If the main plot already has a legend, you might want to skip this to avoid clutter
            ax_rewards.legend(loc='lower left')  # or 'best', or wherever looks best in your case

        legend_setup()

        if title_intensity is None:
            # rp = self.env.return_period
            eq_magnitude = self.env.eq_magnitude
        else:
            # rp = title_intensity
            eq_magnitude = title_intensity

        # # Calculate cal
        # denominator = sum(self.loss_values) + sum(self.resilience_values)
        # cal = sum(self.resilience_values) / denominator

        # print(f"loss values: {self.loss_values}")
        # print(f"resilience values: {self.resilience_values}")

        CL = - sum(self.loss_values)
        # print(self.loss_values)

        ax_main.set_ylabel("Aggregate Functionality Metrics")
        fig.suptitle(
            (
                r"$\bf{Earthquake\ Repair\ Scheduling\ Rollout}$" + "\n"
                + r"$\it{toy\text{-}city\text{-}" + str(self.env.num_components) + str(env_setting) + r"}$" + "\n"
                + f"Policy: {self.name}" + "\n"
                + f"Quake Magnitude: {eq_magnitude} | "
                + f"CL: {CL:.2f}"
            ),
            fontsize=14,
            y=0.98,
            fontfamily='serif'
        )
        ax_main.grid(axis='y', linewidth=0.75, alpha=0.6)  # horizontal faint grid lines only
        ax_rewards.grid(linewidth=0.75, alpha=0.6)  # horizontal faint grid
        ax_main.set_ylim(-0.1, 1.5)
        ax_rewards.set_ylim(-self.env.time_step_duration, self.env.time_step_duration/3)
        plt.tight_layout()
        plt.show()

