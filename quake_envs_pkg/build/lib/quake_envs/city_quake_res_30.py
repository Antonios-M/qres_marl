import sys
from enum import Enum
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import flatdim
import seaborn as sns
import copy
import itertools
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import geopandas as gpd
import math
from .simulations.utils import *
from .simulations.building_config import *
from .simulations.building_funcs import *
from .simulations.road_config import *
from .simulations.road_funcs import *
from .simulations.interdep_network import *
from .simulations.interdependencies import *
from .simulations.traffic_assignment import *
from .simulations.resilience import Resilience

class Quake_Res_30(gym.Env):

    def __init_quake_sim(self):
        in_buildings = gpd.read_file(PathUtils.buildings_toy_shp_2)
        in_roads = gpd.read_file(PathUtils.roads_toy_shp_2)
        in_traffic_gdf = gpd.read_file(PathUtils.traffic_toy_city_geojson_2)
        in_traffic_dem = pd.read_csv(PathUtils.traffic_toy_city_demand_2)
        in_traffic_net = pd.read_csv(PathUtils.traffic_toy_city_network_2)
        self.quake_IM_bldg_save_prefix = "toy_city_bldg_IM"
        self.quake_IM_road_save_prefix = "toy_city_road_IMs"
        sim = InterdependentNetworkSimulation(
            use_premade=True,
            buildings_study_gdf=in_buildings,
            roads_study_gdf=in_roads,
            traffic_net_df=in_traffic_net,
            traffic_dem_df=in_traffic_dem,
            traffic_links_gdf=in_traffic_gdf,
            verbose=False
        )
        sim.buildings_study.get_debris()
        return sim

    def __init_resilience(self):
        """
        Initialize the resilience object with the initial state of the buildings and roads.
        """
        self.resilience = Resilience(
            sum_initial_income=sum([b.max_income for b in self.buildings_objs]),
            sum_current_income=sum([b.current_income for b in self.buildings_objs]),
            sum_current_critical_funcs=sum([b.current_critical_func for b in self.buildings_objs]),
            sum_initial_critical_funcs=sum([b.initial_critical_func for b in self.buildings_objs]),
            sum_current_beds=sum([b.current_beds for b in self.buildings_objs]),
            sum_initial_beds=sum([b.initial_beds for b in self.buildings_objs]),
            sum_current_doctors=sum([b.current_doctors for b in self.buildings_objs]),
            sum_initial_doctors=sum([b.initial_doctors for b in self.buildings_objs]),
            costs=np.array([0.0,0.0,0.0,0.0])
        )

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
            (agent_action_space for _ in range(num_components)),
        )

    def __get_observation_space(self):
        agent_observation_space = gym.spaces.Box(low=0.0, high=600, shape=(1,), dtype=self.dtype)
        self.observation_space = agent_observation_space
        return self.observation_space

    def __get_observation(self) -> np.ndarray:
        self.building_repair_times = np.array([b.current_repair_time for b in self.buildings_objs], dtype=self.dtype)
        self.road_repair_times = np.array([r.current_repair_time for r in self.road_objs], dtype=self.dtype)

        obs = np.concatenate((self.building_repair_times, self.road_repair_times))

        return obs

    def __init__(self,
        verbose: bool=False,
    ):
        self.dtype = np.float32
        self.verbose = verbose
        self.time_step_duration = 100
        self.trucks_per_building_per_day = 0.5
        self.n_agents = 30
        self.n_crews = 10
        self.time_horizon = 30
        self.possible_ds = DamageStates
        self.baselines = {}

        self.simulation = self.__init_quake_sim()
        self.num_buildings = self.simulation.num_buildings
        self.num_roads = self.simulation.num_roads
        self.num_components = self.num_buildings + self.num_roads


        self.__get_observation_space()
        self.__get_action_space()



        self.buildings_gdf = self.simulation.ds_to_int(
            self.simulation.buildings_study(), StudyBuildingSchema.DAMAGE_STATE
        )
        self.roads_gdf = self.simulation.ds_to_int(
            self.simulation.roads_study(), StudyRoadSchema.DAMAGE_STATE
        )
        self.buildings_objs = None
        self.road_objs = None
        self.earthquake_choices = [7.5, 8.0, 8.5, 9.0]
        self.info = {}

    def reset(self, seed: int = None, options: dict = None):
        self.simulation.traffic_calc_net.reset()
        self.initial_mtt, self.initial_traffic_results_df = self.__get_traffic_performance()
        self.current_mtt, self.current_traffic_results_df = self.initial_mtt, self.initial_traffic_results_df
        ## TODO make seed using probabilistic quake scenarios
        self.seed(seed=seed)

        self.simulation.earthquake.predict_building_DS(
            # use_random_IMs=True,
            save_directory=PathUtils.earthquake_model_folder,
            use_saved_IMs=True,
            base_name=self.quake_IM_bldg_save_prefix,
            eq_magnitude=self.eq_magnitude
        )
        self.simulation.earthquake.predict_road_DS(
            # use_random_IMs=True,
            save_directory=PathUtils.earthquake_model_folder,
            use_saved_IMs=True,
            base_name=self.quake_IM_road_save_prefix,
            eq_magnitude=self.eq_magnitude
        )
        self.simulation.earthquake.predict_bridge_DS(
            # use_random_IMs=True,
            save_directory=PathUtils.earthquake_model_folder,
            use_saved_IMs=True,
            base_name=self.quake_IM_road_save_prefix,
            eq_magnitude=self.eq_magnitude
        )
        self.simulation.earthquake.predict_building_RT()
        self.simulation.earthquake.predict_road_RT()
        self.simulation.earthquake.predict_bridge_RT()
        self.building_gdf = self.simulation.buildings_study()
        self.roads_gdf = self.simulation.roads_study()

        self.buildings_gdf, self.roads_gdf = self.simulation.traffic.update_capacities(
            buildings_study_gdf=self.buildings_gdf,
            roads_study_gdf=self.roads_gdf,
            recalculate=True
        )
        self.buildings_objs = make_building_objects(
            buildings_study_gdf=self.buildings_gdf,
            time_step_duration=self.time_step_duration,
            trucks_per_day=self.trucks_per_building_per_day
        )

        self.road_objs = make_road_objects(
            roads_study_gdf=self.roads_gdf,
            time_step_duration=self.time_step_duration,
            traffic_net_df=self.simulation.curr_traffic_net_df
        )
        self.buildings_objs, self.road_objs = map_capacity_reduction_debris(
            buildings=self.buildings_objs,
            roads=self.road_objs
        )
        # print([r.capacity_reduction for r in self.road_objs])
        # self.simulation.buildings_study.get_debris()
        self.simulation.traffic.step_traffic_calc_net(self.road_objs)
        self.current_mtt, self.current_traffic_results_df = self.__get_traffic_performance()


        self.__update_road_value()
        self.__update_building_value()
        # Compute observation for buildings (damage state, repair time)
        self.building_repair_times = [b.current_repair_time for b in self.buildings_objs]
        self.road_repair_times = [r.current_repair_time for r in self.road_objs]
        self.building_damage_states = [b.current_damage_state for b in self.buildings_objs]
        self.road_damage_states = [r.current_damage_state for r in self.road_objs]
        self.time = 0
        self.norm_time = self.time / self.time_horizon
        self.__init_resilience()
        # Return formatted observation matching the defined space
        obs = self.__get_observation()
        info = self.__get_info()
        return obs, info

    def step(self, actions: Tuple[int]):
        actions = np.array(actions)
        actions = self.__prioritise_actions(actions=actions)


        self.dones_buildings = [building.is_functional for building in self.buildings_objs]
        self.info_buildings = [{} for _ in self.buildings_objs]
        self.dones_roads = [building.is_functional for building in self.buildings_objs]
        self.info_roads = [{} for _ in self.buildings_objs]
        self.rewards_buildings = [0.0 for _ in range(len(self.buildings_objs))]

        for agent, action in enumerate(list(actions)):
            if agent < self.num_buildings:
                building = self.buildings_objs[agent]
                state, funcs, done, info = building.step(BuildingAction(action))
                self.rewards_buildings[agent] = funcs
                self.dones_buildings[agent] = done
                self.info_buildings[agent] = info
            else:
                road = self.road_objs[agent - self.num_buildings]
                road_id = road.id
                dependant_buildings = [
                    b for b in self.buildings_objs if b.access_road_id == road_id
                ]

                state, cost, done, info = road.step(RoadAction(action), dependant_buildings=dependant_buildings)
                self.dones_roads[agent - self.num_buildings] = done
                self.info_roads[agent - self.num_buildings] = info

        self.buildings_objs, self.road_objs = map_capacity_reduction_debris(
            buildings=self.buildings_objs,
            roads=self.road_objs
        )

        self.simulation.traffic.step_traffic_calc_net(self.road_objs)

        self.__update_road_value()
        self.__update_building_value()

        self.time += 1

        obs = self.__get_observation()
        reward, econ_r, crit_r, health_r  = self.__get_reward()

        reward = np.float32(reward)
        cost = np.float32(-reward)
        terminated = self.__is_terminated()
        truncated = self.__is_truncated()
        info = self.__get_info()
        info["reward"] = {
            "total": reward,
            "econ": econ_r,
            "crit": crit_r,
            "health": health_r,
        }

        # track if certain repair stages were completed in this time step.
        info["repairs_completed"] = [d["repair_has_finished"] for d in self.info_buildings]
        info["debris_cleared"] = [d["debris_has_cleared"] for d in self.info_buildings]
        info["functionality_restored"] = [d["functionality_has_restored"] for d in self.info_buildings]
        info["q_econ_components"] = self.resilience.q_econ_components
        info["road_repairs_completed"] = [d["road_has_repaired"] for d in self.info_roads]
        return obs, -reward, terminated, truncated, info

    def __prioritise_actions(
        self,
        actions: np.ndarray
    ) -> np.ndarray:
        # Split actions
        building_actions = actions[:self.num_buildings]
        road_actions = actions[self.num_buildings:]

        # Collect active (non-null) actions with their original index and value
        indexed_actions = []

        for i, action in enumerate(building_actions):
            if BuildingAction(action) != BuildingAction.DO_NOTHING:
                value = self.buildings_objs[i].value
                indexed_actions.append((i, value))

        for i, action in enumerate(road_actions):
            if RoadAction(action) not in  [RoadAction.DO_NOTHING, RoadAction.DO_NOTHING_temp] :
                value = self.road_objs[i].value
                indexed_actions.append((self.num_buildings + i, value))

        # Sort by value descending
        indexed_actions.sort(key=lambda x: x[1], reverse=True)

        # Keep only top `n_crews` indices
        top_indices = set(idx for idx, _ in indexed_actions[:self.n_crews])

        # Zero out or set NULL for actions not in top_indices
        pruned_actions = actions.copy()
        for i in range(len(pruned_actions)):
            if i not in top_indices:
                if i < self.num_buildings:
                    pruned_actions[i] = BuildingAction.DO_NOTHING.value
                else:
                    pruned_actions[i] = RoadAction.DO_NOTHING.value

        return pruned_actions

    def __update_road_value(self):
        max_capacity = max([road.capacity for road in self.road_objs])
        for road in self.road_objs:
            value = get_road_value(
                capacity=road.capacity,
                damage_state=road.current_damage_state,
                max_capacity=max_capacity,
                max_damage_state=max([ds.value for ds in DamageStates])
            )
            road.value = value

    def __update_building_value(self):
        nominal_income = max([building.max_income for building in self.buildings_objs])
        nominal_sqft = max([building.sqft for building in self.buildings_objs])
        for building in self.buildings_objs:
            value = get_building_value(
                undisturbed_income=building.max_income,
                nominal_income=nominal_income,
                sqft=building.sqft,
                nominal_sqft=nominal_sqft,
                is_essential=building.is_essential,
                damage_state=building.current_damage_state,
            )
            building.value = value

    def __get_traffic_performance(self):
        # print('traffic_net_df_runtime_capacities' + str([x for x in self.current_traffic_net_df['capacity']]))
        traffic_calc_net = self.simulation.traffic_calc_net

        MTT, results_df = self.simulation.traffic.user_equilibrium(
            traffic_calc_net,
            return_traffic_links_res=True
        )
        if hasattr(self, 'initial_mtt'):
            MTT = max(MTT, self.initial_mtt)

        return MTT, results_df

    def __get_traffic_reward(self):
        delay_time = max(0, self.current_mtt - self.initial_mtt)
        self.__log("Delay Time: " + str(delay_time))
        yearly_delay_cost = TrafficMonetaryValues.compute_yearly_delay_cost(delay_time=delay_time, sample=False)
        if yearly_delay_cost > sum([b.max_income for b in self.buildings_objs]):
            ## cap traffic cost to total community income
            yearly_delay_cost = sum([b.max_income for b in self.buildings_objs])
        return yearly_delay_cost

    def __get_reward(self):
        self.current_mtt = min(self.__get_traffic_performance()[0], self.current_mtt)
        r_bldgs = self.rewards_buildings
        ### ---------- COSTS
        ## repair cost for all building
        cost_buildings = np.sum([r[0] for r in r_bldgs])
        ## repair cost for all roads
        cost_roads = np.sum([r.current_repair_cost for r in self.road_objs])
        ## traffic delay cost
        cost_traffic = float(self.__get_traffic_reward())
        ## relocation cost for all buildings
        cost_relocation = np.sum([r[1] for r in r_bldgs])
        ## summed costs
        costs = np.array([cost_buildings, cost_roads, cost_traffic, cost_relocation])

        ## ----------- BENEFITS
        ## current yearly income for all buildings
        sum_income = np.sum([r[2] for r in r_bldgs])
        ## current critical functionality for all buildings
        sum_critical = np.sum([r[3] for r in r_bldgs])
        ## current hospital beds for all buildings
        sum_beds = np.sum([r[4] for r in r_bldgs])
        ## current number of doctors for all buildings
        sum_doctors = np.sum([r[5] for r in r_bldgs])

        ## community functionality based on costs.
        self.resilience.step(
            sum_income=sum_income,
            sum_critical_funcs=sum_critical,
            sum_beds=sum_beds,
            sum_doctors=sum_doctors,
            costs=costs
        )
        return self.resilience.q_community_decomp

    def __is_terminated(self):
        return all(self.dones_buildings) and all(self.dones_roads)

    def __is_truncated(self):
        return self.time >= self.time_horizon

    def __get_info(self):
        env_info = {
            "system_failure": False,
            "state": self.__get_observation(),
            "obs": self.__get_observation()
        }
        return env_info

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
        title: str = None,
        plot_econ_bldg_repair=False,
        plot_econ_road_repair=False,
        plot_econ_traffic=False,
        plot_econ_relocation=False,
        plot_cost=False,
        figsize=(10, 6),
        agent=None,
        padding_length=10
    ):

        # Reset environment and set up padding
        obs, info = self.reset()

        building_damage_states = [b.current_damage_state for b in self.buildings_objs]
        road_damage_states = [r.current_damage_state for r in self.road_objs]

        total_rewards = [1.0] * padding_length
        econ_rewards = [1.0] * padding_length
        econ_building_repairs_rewards = [1.0] * padding_length
        econ_road_repairs_rewards = [1.0] * padding_length
        econ_traffic_rewards = [1.0] * padding_length
        econ_relocation_rewards = [1.0] * padding_length
        crit_rewards = [1.0] * padding_length
        health_rewards = [1.0] * padding_length
        repairs_counts = []
        debris_clear_counts = []
        func_rest_counts = []
        road_repair_counts = []
        time = list(range(-padding_length, 0))  # Time steps from -100 to -1

        current_time = 0
        has_terminated, has_truncated = False, False

        def has_rollout_ended(term, trunc):
            return trunc or term

        while not has_rollout_ended(has_terminated, has_truncated):
            if agent:
                action = agent.select_action(obs, training=False)
            else:
                action = self.action_space.sample()

            obs, reward, has_terminated, has_truncated, info = self.step(action)
            if plot_cost:
                reward = -reward

            total_reward = reward
            econ_reward = info["reward"]["econ"]
            crit_reward = info["reward"]["crit"]
            health_reward = info["reward"]["health"]
            econ_building_repair = info["q_econ_components"]["building_repair"]
            econ_road_repair = info["q_econ_components"]["road_repair"]
            econ_traffic = info["q_econ_components"]["traffic_delay"]
            econ_relocation = info["q_econ_components"]["relocation"]
            repairs_completed = [
                False if comp is None else comp for comp in info["repairs_completed"]
            ]
            repairs_count = sum(repairs_completed)

            debris_cleared = [
                False if comp is None else comp for comp in info["debris_cleared"]
            ]
            debris_clear_count = sum(debris_cleared)

            functionalities_restored = [
                False if comp is None else comp for comp in info["functionality_restored"]
            ]
            func_rest_count = sum(functionalities_restored)

            road_repairs_completed = [
                False if comp is None else comp for comp in info["road_repairs_completed"]
            ]
            road_repair_count = sum(road_repairs_completed)


            total_rewards.append(total_reward)
            econ_rewards.append(econ_reward)
            econ_building_repairs_rewards.append(econ_building_repair)
            econ_road_repairs_rewards.append(econ_road_repair)
            econ_traffic_rewards.append(econ_traffic)
            econ_relocation_rewards.append(econ_relocation)
            crit_rewards.append(crit_reward)
            health_rewards.append(health_reward)
            repairs_counts.append(repairs_count)
            debris_clear_counts.append(debris_clear_count)
            func_rest_counts.append(func_rest_count)
            road_repair_counts.append(road_repair_count)
            time.append(current_time)
            current_time += 1

        # Add padding at the end
        total_rewards.extend([total_rewards[-1]] * padding_length)
        econ_rewards.extend([econ_rewards[-1]] * padding_length)
        econ_building_repairs_rewards.extend([econ_building_repairs_rewards[-1]] * padding_length)
        econ_road_repairs_rewards.extend([econ_road_repairs_rewards[-1]] * padding_length)
        econ_traffic_rewards.extend([econ_traffic_rewards[-1]] * padding_length)
        econ_relocation_rewards.extend([econ_relocation_rewards[-1]] * padding_length)
        crit_rewards.extend([crit_rewards[-1]] * padding_length)
        health_rewards.extend([health_rewards[-1]] * padding_length)
        repairs_counts.extend([0] * padding_length)
        time.extend(range(current_time, current_time + padding_length))

        # Set up Seaborn style
        sns.set_theme(style="whitegrid", palette="pastel")

        # Create the plot
        plt.figure(figsize=figsize)

        # Set figure background color to white
        plt.gcf().set_facecolor('white')  # Ensure the whole figure is white

        # Set axis background to white
        plt.gca().set_facecolor('white')

        # Define colors
        normal_op_color = 'forestgreen'  # Color for normal operation
        normal_op_alpha = 0.1           # Transparency level for normal operation
        recovery_color = 'orange'       # Color for recovery phase
        recovery_alpha = 0.1            # Transparency level for recovery

        # Add green background for pre-recovery phase (before timestep 0)
        plt.axvspan(xmin=-padding_length, xmax=0, color=normal_op_color, alpha=normal_op_alpha, zorder=-10)
                # Plot Functional Restoration markers with a circle 'o'
        finished_repair = 0
        recovery_ranges = []
        for t, count in zip(time[padding_length:], func_rest_counts):
            if count > 0:
                finished_repair = t
        normal_op_ranges = []
        # Group continuous ranges of recovery (total_reward != 1.0) and normal operation (total_reward == 1.0)
        start_time = None
        for t, total_reward in zip(time[padding_length:], total_rewards[padding_length:]):
            if total_reward != total_rewards[-1]:  # Recovery phase
                if start_time is None:
                    start_time = t  # Start of a new recovery range
            else:  # Normal operation phase
                if start_time is not None:
                    recovery_ranges.append((start_time, t))  # End of the previous recovery range
                    start_time = None
                normal_op_ranges.append((t, t + 1))  # Single time point where reward is 1.0

        # If recovery phase continues until the end, add it to recovery_ranges
        if start_time is not None:
            recovery_ranges.append((start_time, time[-1]))

        # Plot the recovery phase background for the entire range of recovery periods
        plt.axvspan(recovery_ranges[0][0], finished_repair, color=recovery_color, alpha=recovery_alpha, zorder=-10)


        # Plot the normal operation phase background for the entire range of normal operation periods
        plt.axvspan(finished_repair, normal_op_ranges[-1][1], color=normal_op_color, alpha=normal_op_alpha, zorder=-10)

        def plot_sub_functionalities():
            # Main reward curves
            sns.lineplot(x=time, y=econ_rewards, label='Economic Functionality', color='teal', linewidth=1.5, zorder=10)
            sns.lineplot(x=time, y=crit_rewards, label='Critical Functionality', color='darkorange', linewidth=1.5, zorder=10)
            sns.lineplot(x=time, y=health_rewards, label='Healthcare Functionality', color='crimson', linewidth=1.5, zorder=10)

        plot_sub_functionalities()

        def plot_sub_econ_functionalities():
            # Component reward curves
            if plot_econ_bldg_repair:
                sns.lineplot(x=time, y=econ_building_repairs_rewards, label='Building Repair Costs', color='darkgreen', linestyle=(0, (1, 3)), linewidth=0.75, zorder=10)
            if plot_econ_road_repair:
                sns.lineplot(x=time, y=econ_road_repairs_rewards, label='Road Repair Costs', color='darkorange', linestyle=(0, (1, 3)), linewidth=0.75, zorder=10)
            if plot_econ_traffic:
                sns.lineplot(x=time, y=econ_traffic_rewards, label='Traffic Delay Costs', color='dimgray', linestyle=(0, (3, 1, 1, 1)), linewidth=0.75, zorder=10)
            if plot_econ_relocation:
                sns.lineplot(x=time, y=econ_relocation_rewards, label='Relocation Costs', color='rosybrown', linestyle=(0, (3, 1, 1, 1, 1, 1)), linewidth=0.75, zorder=10)

        plot_sub_econ_functionalities()


        sns.lineplot(x=time, y=total_rewards, label='Community Functionality', linewidth=2.5, color='black', zorder=10)
        # Calculate the base position for repairs and adjust for other categories
        repairs_y = 1.1
        debris_y = repairs_y + 0.05
        func_y = repairs_y + 0.1
        road_repair_y = repairs_y + 0.2

        for t, count in zip(time[padding_length:], repairs_counts):
            if count > 0:
                size = 50 + 10 * count
                plt.scatter(t, repairs_y,
                            s=size, marker='>',
                            color='slateblue',
                            edgecolor='black',
                            alpha=0.7,
                            label='Repairs Completed' if t == time[padding_length] else "",
                            zorder=15)

        # Make sure these markers are visible
        lowest_y = min(total_rewards + econ_rewards + crit_rewards + health_rewards + econ_building_repairs_rewards + econ_road_repairs_rewards + econ_traffic_rewards + econ_relocation_rewards) - 0.05  # Adjusted for visual
        plt.ylim(lowest_y, 1.5)  # Adjust upper limit as needed

        # Plot Debris Clear markers with a square 's'
        for t, count in zip(time[padding_length:], debris_clear_counts):
            if count > 0:
                size = 50 + 10 * count
                plt.scatter(t, debris_y,  # Position slightly below repairs_y
                            s=size, marker=10,  # Square marker for debris clear
                            color='darkmagenta',  # Example color for debris
                            edgecolor='black',  # Black border around the markers
                            alpha=0.7,  # Set alpha for opacity of facecolor
                            label='Debris Cleared' if t == time[padding_length] else "",
                            zorder=14)


        # Plot Functional Restoration markers with a circle 'o'
        for t, count in zip(time[padding_length:], func_rest_counts):
            if count > 0:
                size = 50 + 10 * count
                plt.scatter(t, func_y,  # Position even further below
                            s=size, marker='d',  # Circle marker for functional restoration
                            color='green',  # Example color for functional restoration
                            edgecolor='black',  # Black border around the markers
                            alpha=0.7,  # Set alpha for opacity of facecolor
                            label='Functional Restoration' if t == time[padding_length] else "",
                            zorder=13)

                # Plot Functional Restoration markers with a circle 'o'
        for t, count in zip(time[padding_length:], road_repair_counts):
            if count > 0:
                size = 50 + 10 * count
                plt.scatter(t, road_repair_y,  # Position even further below
                            s=size, marker='4',  # Circle marker for functional restoration
                            color='green',  # Example color for functional restoration
                            edgecolor='black',  # Black border around the markers
                            alpha=0.7,  # Set alpha for opacity of facecolor
                            label='Road Repairs' if t == time[padding_length] else "",
                            zorder=13)





        # Labels and title
        plt.xlabel(f'Time Step / {self.time_step_duration} days ')
        plt.ylabel('Community Functionality')
        plt.title(f'Rollout Plot for toy-city-{self.num_components} with quake magnitude: {self.eq_magnitude}')


        # Create the legend and place it below the plot
        # Define custom markers for legend
        repairs_marker = plt.Line2D([0], [0], marker='>', color='w', markerfacecolor='slateblue', markersize=10, label='Repairs Completed', markeredgewidth=2)
        debris_marker = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkorange', markersize=10, label='Debris Cleared', markeredgewidth=2)
        func_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Functional Restoration', markeredgewidth=2)

        # Add a legend for the three sets of markers
        handles_marker = [repairs_marker, debris_marker, func_marker]
        labels_marker = ['Repairs Completed', 'Debris Cleared', 'Functional Restoration']
        handles, labels = plt.gca().get_legend_handles_labels()
        handles += handles_marker
        labels += labels_marker
        by_label = dict(zip(labels, handles))

        plt.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

        # Ensure everything fits
        plt.tight_layout()

        # Show the plot
        plt.show()
