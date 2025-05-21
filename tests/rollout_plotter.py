import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from quake_envs.simulations.earthquake_funcs import DamageStates
from mpl_toolkits.mplot3d import Axes3D
import math

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
        damage_state: int
    ) -> float:
        income_value = undisturbed_income / nominal_income
        ds_value = damage_state / len(DamageStates)
        sqft_value = sqft / nominal_sqft
        essential_value = 1 if is_essential else 0.5

        value = income_value * ds_value * sqft_value * essential_value
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

        # Determine the number of crews for buildings and roads
        crews_for_buildings = self.n_crews // 2
        crews_for_roads = self.n_crews - crews_for_buildings  # Handles odd numbers

        # Sort buildings and roads by value descending
        sorted_buildings = sorted(
            [(i, b.value) for i, b in enumerate(self.buildings)],
            key=lambda x: -x[1]
        )
        sorted_roads = sorted(
            [(i, r.value) for i, r in enumerate(self.roads)],
            key=lambda x: -x[1]
        )

        # Assign actions to top-priority buildings
        building_crews_used = 0
        for idx, _ in sorted_buildings:
            if building_crews_used >= crews_for_buildings:
                break
            building = self.buildings[idx]
            if building.has_debris:
                action[idx] = 1  # clear debris
                building_crews_used += 1
            elif not building.is_fully_repaired:
                action[idx] = 1  # repair
                building_crews_used += 1

        # Assign actions to top-priority roads
        road_crews_used = 0
        for idx, _ in sorted_roads:
            if road_crews_used >= crews_for_roads:
                break
            road = self.roads[idx]
            if not road.is_fully_repaired:
                action[len(self.buildings) + idx] = 1  # repair
                road_crews_used += 1
        return action


class AgentPlotter:
    def __init__(self, env, agent):

        self.env = env
        self.agent = agent
        self.time_horizon = env.time_horizon

        self.importance_based = BaselineImportanceBased(env)

    def get_sample_rollout(self):
        padding_length = int(0.05 * self.env.time_horizon)
        # Reset environment and set up padding
        obs, info = self.env.reset()
        delay_time = self.env.resilience.delay_time
        # print(f"Reset obs: {obs}")
        # print(f"Reset damage states: {[[b.current_damage_state for b in self.env.resilience.buildings_objs],[r.current_damage_state for r in self.env.resilience.road_objs]]}")
        self.info_buildings = {}
        self.info_roads = {}
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
            # print(f"--------- Step: {current_time}")
            rollout_end += 1
            # print(f"Step: {current_time}")
            if self.agent not in ['random', 'importance_based', "do_nothing"]:
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
            next_obs, cost, has_terminated, has_truncated, info = self.env.step(action)
            # print(f"Current Time: {current_time}")
            # print(f"Action: {action}")
            # print(f"Road Capacity Reductions: {[r.capacity_reduction for r in self.env.resilience.road_objs]}")
            # print(f"Road DS Capacity Reductions: {[r.capacity_red_damage_state for r in self.env.resilience.road_objs]}")
            # print(f"Road Debris Capacity Reductions: {[r.capacity_red_debris for r in self.env.resilience.road_objs]}")
            # print(f"Building has debris: {[b.has_debris for b in self.env.resilience.buildings_objs]}")
            # print(f"Building Damage States: {[b.current_damage_state for b in self.env.resilience.buildings_objs]}")

            ## Building Info
            brts, bdss, brcs, bincs, brelocs = self.env.resilience.get_building_info()
            self.info_buildings[self.env.resilience.time] = {
                "repair_times": brts,
                "damage_states": bdss,
                "repair_costs": brcs,
                "incomes": bincs,
                "relocation_costs": brelocs,
                "debris_clears": info["completions"]["bldg_debris"]
            }
            ## Road Info
            rrts, rdss, rrcs, rcrds = self.env.resilience.get_road_info()
            self.info_roads[self.env.resilience.time] = {
                "repair_times": rrts,
                "damage_states": rdss,
                "repair_costs": rrcs,
                "capacity_reductions": rcrds,
                "repairs": info["completions"]["road_repairs"]
            }
            # print(f"Resilience.info capacity reductions: {rcrds}")

            # print(f"Next obs: {next_obs}")
            # print(next_obs)
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
            obs = next_obs

        all_data = [returns, returns_econ, returns_crit, returns_health, q_community_values, q_econ_values, q_income_values,  q_econ_bldg_repair_costs, q_econ_road_repair_costs, q_econ_traffic_delay_costs, q_econ_relocation_costs, q_crit_values, q_health_values, bldg_repairs_counts, road_repair_counts, debris_clear_counts, func_rest_counts]
        time.extend(range(current_time, current_time + padding_length))
        # print(f"Episodic Returns: {sum(returns)}")
        # Add padding at the end
        def extend_end():
            for data in all_data:
                data.extend([data[-1]] * padding_length)

        extend_end()

        return all_data, padding_length, rollout_start, rollout_end, time, delay_time

    def plot_returns_3d(self, n=100, figsize=(10, 10)):
        all_returns = []

        for _ in range(n):
            all_data, _, _, _, _ = self.get_sample_rollout()
            returns = all_data[0]  # list of rewards per timestep
            all_returns.append(returns)

        all_returns = np.array(all_returns)  # shape: [n, T]

        X = np.arange(all_returns.shape[1])  # timesteps
        Y = np.arange(all_returns.shape[0])  # rollouts
        X, Y = np.meshgrid(X, Y)
        Z = all_returns

        # Compute summary statistics
        total_returns = np.sum(all_returns, axis=1)
        mean_total = np.mean(total_returns)
        std_total = np.std(total_returns)

        # Plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

        # Final reward line
        last_timestep = all_returns.shape[1] - 1
        final_rewards = all_returns[:, last_timestep]
        rollout_indices = np.arange(all_returns.shape[0])
        timesteps = np.full_like(rollout_indices, last_timestep)

        ax.plot3D(timesteps, rollout_indices, final_rewards, color='black', linewidth=2, label='Final Rewards', zorder=10)

        # Set viewing angle (elevation, azimuth)
        ax.view_init(elev=30, azim=-45)  # Adjust as needed for centering

        # Labels and annotation
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Rollout n')
        ax.set_zlabel('Reward')
        ax.set_title(f'Reward Over {n} Rollouts')

        legend_text = f"Mean Total Return: {mean_total:.2f}\nStd Dev: {std_total:.2f}"
        ax.text2D(0.05, 0.95, legend_text, transform=ax.transAxes)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show()

    def run_n_steps(self, n=100):
        obs, info = self.env.reset()
        print(f"Reset Observations: {obs}")
        for i in range(n):
            for j in range(10):
                print("----------------------------------------")
                print(f"Step {j}")
                actions = (0,0,0,0)
                print(f"Actions: {actions}")
                obs, reward, done, term, info = self.env.step(actions)
                print(f"Observation: {obs}, Reward: {reward}, Done: {done}, Term: {term}")

    def plot_components(self, figsize=(20, 20)):
        all_data, padding_length, rollout_start, rollout_end, time, delay_time = self.get_sample_rollout()
        def normalize_to_range_0_n(values, n=4, min_val=None):
            # min_val = min(values) if values else 0
            max_val = max(values) if values else 1
            if max_val == min_val:
                return [0 for _ in values]
            return [n * (v - min_val) / (max_val - min_val) for v in values]

        road_data = {
            i: {
                'timesteps': [],
                'repair_times': [],
                'damage_states': [],
                'repair_costs': [],
                'capacity_reductions': []
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
                "debris_clears": []
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
            for i, b in enumerate(self.env.resilience.buildings_objs):
                building_data[i]['debris_clears'].append(attrs['debris_clears'][i])
                building_data[i]['timesteps'].append(timestep)
                building_data[i]['repair_times'].append(rts[i])
                building_data[i]['damage_states'].append(dss[i])
                building_data[i]['repair_costs'].append(rcs[i])
                building_data[i]['incomes'].append(incs[i])
                building_data[i]['relocation_costs'].append(relocs[i])

        for timestep, attrs in self.info_roads.items():
            rts = attrs['repair_times']
            dss = attrs['damage_states']
            max_ds = max(dss)
            rcs = attrs['repair_costs']
            crds = attrs['capacity_reductions']
            for i, r in enumerate(self.env.resilience.road_objs):
                road_data[i]['timesteps'].append(timestep)
                road_data[i]['repair_times'].append(rts[i])
                road_data[i]['damage_states'].append(dss[i])
                road_data[i]['repair_costs'].append(rcs[i])
                road_data[i]['capacity_reductions'].append(crds[i])

        n_buildings = len(building_data)
        n_roads = len(road_data)
        cols = int((n_buildings + n_roads) / 2)
        rows = math.ceil((n_buildings + n_roads) / cols)

        sns.set_theme(style="whitegrid", context="notebook")
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False)
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

            ax.axvspan(xmin=timesteps[0], xmax=t_debris, color='orange', alpha=0.1, label='_nolegend_')  # Pre-debris

            ax.plot(timesteps, rt, label='Repair Time', linestyle='-', marker='o')
            ax.plot(timesteps, rc, label='Repair Cost', linestyle='--', marker='^')
            ax.plot(timesteps, ds, label='Damage State', linestyle='-.', marker='s')
            ax.plot(timesteps, inc, label='Income', linestyle=':', marker='d')
            ax.plot(timesteps, reloc, label='Relocation Cost', linestyle='-', marker='x')

            ax.set_title(f"Building {i}, ID: {i}, Occ: {b.occtype}, init_ds: {b.initial_damage_state}, init_rt: {b.initial_repair_time}")
            ax.set_ylim(0, 4)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Metric")

            if i == 0:
                ax.legend(fontsize='small')

        for j, r in enumerate(self.env.resilience.road_objs):
            ax = axes[j + n_buildings]

            timesteps = road_data[j]['timesteps']
            ds = road_data[j]['damage_states']
            ds_max = max(ds) if ds else 1

            rt = normalize_to_range_0_n(road_data[j]['repair_times'], ds_max, min_val=0)
            rc = normalize_to_range_0_n(road_data[j]['repair_costs'], ds_max, min_val=0)
            crd = normalize_to_range_0_n(road_data[j]['capacity_reductions'], ds_max, min_val=0)

            ax.plot(timesteps, rt, label='Repair Time', linestyle='-', marker='o')
            ax.plot(timesteps, rc, label='Repair Cost', linestyle='--', marker='^')
            ax.plot(timesteps, ds, label='Damage State', linestyle='-.', marker='s')
            ax.plot(timesteps, crd, label='Capacity Reduction', linestyle=':', marker='d')

            ax.set_title(f"Road {j}, ID: {j}, init_ds: {r.initial_damage_state}, init_rt: {r.initial_repair_time}")
            ax.set_ylim(0, 4)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Normalized Metric")

            if j == 0:
                ax.legend(fontsize='small')

        for j in range(n_buildings + n_roads, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=2.5)
        plt.subplots_adjust(hspace=0.6, wspace=0.5)
        plt.show()

    def plot(
        self,
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
        all_data, padding_length, rollout_start, rollout_end, time, delay_time = self.get_sample_rollout()
        # # [DEBUG] negatie rewards
        # returns = all_data[0]
        # def has_ups_and_downs(returns):
        #     diffs = np.diff(returns)
        #     return np.any(diffs > 0) and np.any(diffs < 0)

        # # Keep sampling until returns contain both ups and downs
        # while not has_ups_and_downs(returns):
        #     all_data, padding_length, rollout_start, rollout_end, time, delay_time = self.get_sample_rollout()
        #     returns = all_data[0]

        # # Now returns has both ups and downs
        # print("Accepted returns:", returns)
        # print(sum(returns))
        returns = all_data[0]
        returns_econ = all_data[1]
        returns_crit = all_data[2]
        returns_health = all_data[3]
        q_community_values = all_data[4]
        q_econ_values = all_data[5]
        q_income_values = all_data[6]
        q_econ_bldg_repair_costs = all_data[7]
        q_econ_road_repair_costs = all_data[8]
        q_econ_traffic_delay_costs = all_data[9]
        q_econ_relocation_costs = all_data[10]
        q_crit_values = all_data[11]
        q_health_values = all_data[12]
        bldg_repairs_counts = all_data[13]
        road_repair_counts = all_data[14]
        debris_clear_counts = all_data[15]
        func_rest_counts = all_data[16]

        adjust_padding_length = padding_length
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
        for t, count in zip(time[adjust_padding_length:], func_rest_counts):
            if count > 0:
                finished_repair = t
        for t, total_reward in zip(time[adjust_padding_length:], returns[adjust_padding_length:]):
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
            for t, count in zip(time[adjust_padding_length:], bldg_repairs_counts):
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
            for t, count in zip(time[adjust_padding_length:], debris_clear_counts):
                if rollout_start <= t <= rollout_end and count > 0:
                    size = 50 + 10 * count
                    ax_main.scatter(
                        t, debris_y,
                        s=size,
                        marker=10,
                        color='darkorange',
                        edgecolor='black',
                        alpha=0.7,
                        label='Debris Cleared' if not plotted_labels['Debris Cleared'] else "",
                        zorder=14
                    )
                    plotted_labels['Debris Cleared'] = True

            # Functional Restoration
            for t, count in zip(time[adjust_padding_length:], func_rest_counts):
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
            for t, count in zip(time[adjust_padding_length:], road_repair_counts):
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
            ax_rewards.set_xlabel(f'Time Step / {self.env.time_step_duration} days')
            ax_rewards.legend(loc='upper right')
            ax_rewards.set_ylim(returns_np.min() - 0.1 * returns_np.max(), returns_np.max() + 0.1 * returns_np.max())

        plot_reward()
        # Legend setup
        def legend_setup():
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

        legend_setup()

        if title_intensity is None:
            eq_magnitude = self.env.eq_magnitude
        else:
            eq_magnitude = title_intensity

        ax_main.set_ylabel("Aggregate Functionality Metrics")
        fig.suptitle(
            r"$\bf{Earthquake\ Repair\ Scheduling\ Rollout}$" + "\n"
            + r"$\it{toy\text{-}city\text{-}" + str(self.env.num_components) + str(env_setting) + r"}$" + "\n"
            + f"Earthquake Magnitude: {eq_magnitude}"
            + f" - Total Returns: {sum(returns):.2f}",
            fontsize=14,
            y=0.98,
            fontfamily='serif'  # or 'sans-serif', 'monospace', or a specific one like 'DejaVu Sans'
        )
        plt.tight_layout()
        plt.show()

