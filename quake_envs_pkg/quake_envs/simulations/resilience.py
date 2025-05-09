from .building_funcs import Building
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from .building_funcs import *
from .road_funcs import *
from .utils import *
from .interdep_network import *
from .traffic_assignment import *
import pandas as pd
import geopandas as gpd
import json
import os
import re

class Resilience:
    def __init__(
        self,
        n_agents: int,
        n_crews: int,
        time_horizon: int,
        time_step_duration: int,
        truck_debris_per_day: float,
        w_econ: float = 0.1,
        w_crit: float = 0.45,
        w_health: float = 0.45,
        w_health_bed: float = 0.5,
        w_health_doc: float = 0.5

    ):
        self.n_agents = n_agents
        self.n_crews = n_crews
        self.time_step_duration = time_step_duration
        self.trucks_debris_per_day = truck_debris_per_day
        self.time_horizon = time_horizon
        self.w_econ = w_econ
        self.w_crit = w_crit
        self.w_health = w_health
        self.w_bed = w_health_bed
        self.w_doc = w_health_doc

        self.simulation = self.__init_simulation()
        self.num_buildings = self.simulation.num_buildings
        self.num_roads = self.simulation.num_roads


        self.buildings_gdf = self.simulation.ds_to_int(
            self.simulation.buildings_study(), StudyBuildingSchema.DAMAGE_STATE
        )
        self.roads_gdf = self.simulation.ds_to_int(
            self.simulation.roads_study(), StudyRoadSchema.DAMAGE_STATE
        )
        self.buildings_objs = None
        self.road_objs = None
        min_road_rt, max_road_rt = get_road_obs_bounds()
        min_bldg_rt, max_bldg_rt = get_building_obs_bounds()
        self._max_obs = max(max_road_rt, max_bldg_rt)

        self.earthquake_choices = [7.5, 8.0, 8.5, 9.0]
        self.info = {}

    def __init_simulation(self) -> InterdependentNetworkSimulation:
        env_data = PathUtils.env_data[str(self.n_agents)]
        in_buildings = gpd.read_file(env_data["buildings"])
        in_roads = gpd.read_file(env_data["roads"])
        in_traffic_gdf = gpd.read_file(env_data["traffic_links"])
        in_traffic_dem = pd.read_csv(env_data["traffic_dem"])
        in_traffic_net = pd.read_csv(env_data["traffic_net"])
        self.quake_IM_bldg_save_prefix = "toy_city_" + str(self.n_agents) + "_bldg_IM"
        self.quake_IM_road_save_prefix = "toy_city_" + str(self.n_agents) + "_road_IMs"

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

    def __simulate_earthquake(self):
        eq_magnitude = self.eq_magnitude
        self.simulation.earthquake.predict_building_DS(
            save_directory=PathUtils.earthquake_model_folder,
            use_saved_IMs=True,
            base_name=self.quake_IM_bldg_save_prefix,
            eq_magnitude=eq_magnitude
        )
        self.simulation.earthquake.predict_road_DS(
            save_directory=PathUtils.earthquake_model_folder,
            use_saved_IMs=True,
            base_name=self.quake_IM_road_save_prefix,
            eq_magnitude=eq_magnitude
        )
        self.simulation.earthquake.predict_bridge_DS(
            save_directory=PathUtils.earthquake_model_folder,
            use_saved_IMs=True,
            base_name=self.quake_IM_road_save_prefix,
            eq_magnitude=eq_magnitude
        )
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
            trucks_per_day=self.trucks_debris_per_day
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
        self.simulation.traffic.step_traffic_calc_net(self.road_objs)

    def __get_mean_travel_time(self):
        traffic_calc_net = self.simulation.traffic_calc_net

        MTT, _ = self.simulation.traffic.user_equilibrium(
            traffic_calc_net,
            return_traffic_links_res=True
        )
        if hasattr(self, 'initial_mtt'):
            MTT = max(MTT, self.initial_mtt)

        return MTT

    def __get_traffic_delay_cost(self) -> float:
        delay_time = max(0, self.current_mtt - self.initial_mtt)
        yearly_delay_cost = TrafficMonetaryValues.compute_yearly_delay_cost(delay_time=delay_time, sample=False)
        if yearly_delay_cost > sum([b.max_income for b in self.buildings_objs]):
            ## cap traffic cost to total community income
            yearly_delay_cost = sum([b.max_income for b in self.buildings_objs])
        return yearly_delay_cost

    def get_info(self):
        info = {
            "income": sum([b.current_income for b in self.buildings_objs]),
            "costs": {
                "buildings_repair_cost":  - sum([b.current_structural_repair_cost for b in self.buildings_objs]),
                "roads_repair_cost": - sum([r.current_repair_cost for r in self.road_objs]),
                "traffic_delay_cost": - self.__get_traffic_delay_cost(),
                "relocation_cost": - sum([b.get_relocation_cost() for b in self.buildings_objs])
            },
            "functionalities": {
                "critical_functionality": sum([b.get_critical_functionality() for b in self.buildings_objs]),
                "hospital_beds": sum([b.get_hosp_beds() for b in self.buildings_objs]),
                "doctors": sum([b.get_doctors() for b in self.buildings_objs])
            }
        }
        return info

    def state(self, dtype):
        return np.concatenate((
            np.array([b.current_repair_time for b in self.buildings_objs], dtype=dtype),
            np.array([r.current_repair_time for r in self.road_objs], dtype=dtype)
        ))
        # return np.concatenate((
        #     np.array([b.current_damage_state for b in self.buildings_objs], dtype=dtype),
        #     np.array([r.current_damage_state for r in self.road_objs], dtype=dtype)
        # ))

    def compute_down_time_delays(self, financing_method: str = "insurance"):
        delay = np.zeros(len(self.buildings_objs))
        for i, building in enumerate(self.buildings_objs):
            dt = building.get_downtime_delay(financing_method=financing_method)
            delay[i] = dt

        adjusted_delay = np.max(delay) /  self.time_step_duration
        return adjusted_delay

    def reset(
        self,
        eq_magnitude: int
    ):
        self.time = 0
        self.simulation.traffic_calc_net.reset()
        self.initial_mtt = self.__get_mean_travel_time()
        self.current_mtt = self.initial_mtt
        self.eq_magnitude = eq_magnitude
        self.__simulate_earthquake()
        # print(f"Log: Initial mean travel time: {self.initial_mtt}")
        # print(f"Log: Initial building repair times: {[b.current_repair_time for b in self.buildings_objs]}")
        # print(f"Log: Initial road repair times: {[r.current_repair_time for r in self.road_objs]}")
        # print(f"Log: Initial building damage states: {[b.current_damage_state for b in self.buildings_objs]}")
        # print(f"Log: Initial road damage states: {[r.current_damage_state for r in self.road_objs]}")
        self.current_mtt = self.__get_mean_travel_time()
        # self.__assign_random_normalized_ranks()

        self.initial_income = sum([b.max_income for b in self.buildings_objs])
        self.initial_critical_func = sum([b.initial_critical_func for b in self.buildings_objs])
        self.initial_beds = sum([b.initial_beds for b in self.buildings_objs])
        self.initial_doctors = sum([b.initial_doctors for b in self.buildings_objs])
        self.delay_time = self.compute_down_time_delays()

        return self.q_community_decomp

    def _save_env_config(self, folder_path: str):
        bldg = self.buildings_objs[0]
        road = self.road_objs[0]

        env_config = {
            "n_agents": self.n_agents,
            "n_crews": self.n_crews,
            "time_horizon": self.time_horizon,
            "time_step_duration": self.time_step_duration,
            "trucks_debris_per_day": self.trucks_debris_per_day,
            "w_econ": self.w_econ,
            "w_crit": self.w_crit,
            "w_health": self.w_health,
            "w_bed": self.w_bed,
            "w_doc": self.w_doc,
            "building": {
                "time_step_duration": bldg.time_step_duration,
                "trucks_per_day": bldg.trucks_per_day,
                "verbose": bldg.verbose,
                "stoch_ds": bldg.stoch_ds,
                "calc_debris": bldg.calc_debris,
                "stoch_rt": bldg.stoch_rt,
                "stoch_cost": bldg.stoch_cost,
                "stoch_inc_loss": bldg.stoch_inc_loss,
                "stoch_loss_of_function": bldg.stoch_loss_of_function,
                "stoch_relocation_cost": bldg.stoch_relocation_cost
            },
            "road": {
                "time_step_duration": road.time_step_duration,
                "verbose": road.verbose,
                "stoch_ds": road.stoch_ds,
                "calc_debris": road.calc_debris,
                "stoch_rt": road.stoch_rt,
                "stoch_cost": road.stoch_cost
            }
        }

        # Ensure folder exists
        os.makedirs(folder_path, exist_ok=True)

        # Find existing versioned files
        existing_files = os.listdir(folder_path)
        version_nums = [
            int(re.match(r"v(\d+)\.json", f).group(1))
            for f in existing_files
            if re.match(r"v(\d+)\.json", f)
        ]
        next_version = max(version_nums) + 1 if version_nums else 1

        file_name = f"v{next_version}.json"
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "w") as f:
            json.dump(env_config, f, indent=4)

    def step(self,
        actions: tuple
    ) -> None:
        # original_actions = np.array(actions)
        # actions = self.__prioritise_actions(original_actions)

        # Check if actions changed during prioritization (agents chose too many repair actions for n_crews available)
        # self.actions_changed = not np.array_equal(actions, original_actions)

        ## DEBUG: make all actions possible, even repair everything at all times
        actions = np.array(actions)
        self.actions_changed = False

        self.info_buildings = [{} for _ in self.buildings_objs]
        self.info_roads = [{} for _ in self.road_objs]

        for agent, action in enumerate(list(actions)):
            if agent < self.num_buildings:
                building = self.buildings_objs[agent]
                building_info = building.step(BuildingAction(action))
                self.info_buildings[agent] = building_info
            else:
                road = self.road_objs[agent - self.num_buildings]
                dependant_buildings = [
                    b for b in self.buildings_objs if b.access_road_id == road.id
                ]
                road_info = road.step(RoadAction(action), dependant_buildings=dependant_buildings)
                self.info_roads[agent - self.num_buildings] = road_info

        self.buildings_objs, self.road_objs = map_capacity_reduction_debris(
            buildings=self.buildings_objs,
            roads = self.road_objs
        )
        self.simulation.traffic.step_traffic_calc_net(self.road_objs)
        self.current_mtt = min(self.__get_mean_travel_time(), self.current_mtt)
        # self.__assign_random_normalized_ranks()
        self.time += 1

        q_community, q_econ, q_crit, q_health = self.q_community_decomp
        bldg_repairs_completed = [d["repair_has_finished"] for d in self.info_buildings]
        bldg_repairs_completed = [False if rep is None else rep for rep in bldg_repairs_completed]

        road_repairs_completed = [d["road_has_repaired"] for d in self.info_roads]
        road_repairs_completed = [False if rep is None else rep for rep in road_repairs_completed]

        bldg_debris_cleared = [d["debris_has_cleared"] for d in self.info_buildings]
        bldg_debris_cleared = [False if clear is None else clear for clear in bldg_debris_cleared]

        bldg_funcs_restored = [d["functionality_has_restored"] for d in self.info_buildings]
        bldg_funcs_restored = [False if clear is None else clear for clear in bldg_funcs_restored]

        q_econ_components = self.q_econ[1]
        if self.time == 1:
            self.post_quake_func = q_community
            self.post_quake_econ_func = q_econ
            self.post_quake_crit_func = q_crit
            self.post_quake_health_func = q_health

        return {
            "q": {
                "community_robustness": self.post_quake_func,
                "community_robustness_econ": self.post_quake_econ_func,
                "community_robustness_crit": self.post_quake_crit_func,
                "community_robustness_health": self.post_quake_health_func,
                "community": q_community,
                "econ": q_econ,
                "crit": q_crit,
                "health": q_health
            },
            "q_econ_components": q_econ_components,
            "completions": {
                "bldg_repairs": bldg_repairs_completed,
                "road_repairs": road_repairs_completed,
                "bldg_debris": bldg_debris_cleared,
                "bldg_funcs": bldg_funcs_restored
            }
        }

    @property
    def q_community_decomp(self) -> Tuple[float, float, float]:
        q_community = min(1.0, self.w_econ * self.q_econ[0] + self.w_crit * self.q_crit + self.w_health * self.q_health)
        return (
            q_community,
            self.w_econ * self.q_econ[0],
            self.w_crit * self.q_crit,
            self.w_health * self.q_health,
        )

    @property
    def q_econ(self) -> tuple:
        """
        Decomposes the economic functionality (BCR) into:
        - Income contribution
        - Cost contributions (negative)
        Returns a dict where all values sum up to q_econ.
        """
        info = self.get_info()

        # Income term: normalized by initial income
        curr_incom = info["income"]
        income_term = info["income"] / self.initial_income

        # Cost terms: each already negative in get_info, normalize by initial income
        costs = info["costs"]
        cost_terms = {
            name: cost / self.initial_income
            for name, cost in costs.items()
        }
        # Assemble components
        components = {'income': income_term}
        components.update(cost_terms)

        return (
            sum(components.values()),
            components
        )

    @property
    def q_crit(self) -> float:
        """Calculate the critical functionality"""
        info = self.get_info()
        return info["functionalities"]["critical_functionality"] / self.initial_critical_func

    @property
    def q_health(
        self,
    ) -> float:
        info = self.get_info()
        beds = info["functionalities"]["hospital_beds"]
        doctors = info["functionalities"]["doctors"]
        q_beds = 0.0 if self.initial_beds == 0 else beds / self.initial_beds
        q_doctors = 0.0 if self.initial_doctors == 0 else doctors / self.initial_doctors
        q_health = self.w_bed * q_beds + self.w_doc * q_doctors
        return q_health

    @property
    def terminated(self):
        dones_buildings = [b.is_functional for b in self.buildings_objs]
        dones_roads = [r.is_fully_repaired for r in self.road_objs]

        are_buildings_done = all(dones_buildings)
        are_roads_done = all(dones_roads)

        return are_buildings_done and are_roads_done

    @property
    def truncated(self):
        truncation_conditions_met = (
            self.time >= self.time_horizon, ## time horizong exceeded
            self.actions_changed            ## actions changed during prioritization
        )
        return truncation_conditions_met

    def __assign_random_normalized_ranks(self):
        """
        Assigns a unique random rank (normalized between 0 and 1) to each road and building.
        The highest rank gets a value of 1.0, lowest gets 0.0.
        """
        all_objects = self.road_objs + self.buildings_objs
        total_items = len(all_objects)

        if total_items == 0:
            return

        random.shuffle(all_objects)

        for i, obj in enumerate(all_objects):
            obj.value = i / (total_items - 1) if total_items > 1 else 0.0

    def __prioritise_actions(
        self,
        actions: np.ndarray
    ) -> np.ndarray:
        # Split actions
        building_actions = actions[:self.num_buildings]
        road_actions = actions[self.num_roads:]

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




