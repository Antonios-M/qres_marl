from typing import Union, Optional, List, Final
from scipy.stats import lognorm
import numpy as np
import geopandas as gpd
import json
from pathlib import Path
from dataclasses import dataclass
import scipy.stats as stats
from enum import Enum


HAZUS_QUAKE_TECH_MANUAL = "Federal Emergency Mangement Agency, Hazus 6.0 Earthquake Model Technical Manual. (2024b). from: https://www.fema.gov/sites/default/files/2020-10/fema_hazus_earthquake_technical_manual_4-2.pdf"
HAZUS_INVENTORY_TECH_MANUAL = "Federal Emergency Mangement Agency, Hazus 6.0 Inventory Technical Manual. (2024b). from: https://www.fema.gov/sites/default/files/documents/fema_hazus-6-inventory-technical-manual.pdf"

class DamageStates(Enum):
    """Represents different states of damage as an Enum."""
    UNDAMAGED = 0
    SLIGHT = 1
    MODERATE = 2
    EXTENSIVE = 3
    COMPLETE = 4

    @classmethod
    def to_int(cls, damage_state: str) -> int:
        """Converts a damage state string to its corresponding integer value."""
        try:
            return cls[damage_state.upper()].value
        except KeyError:
            return 0  # Default to 0 if invalid

    @classmethod
    def to_str(cls, damage_state: int) -> str:
        """Converts a damage state integer to its corresponding string value."""
        for state in cls:
            if state.value == damage_state:
                return state.name.capitalize()
        return "Invalid"

def sample_lognormal(
        median: float,
        dispersion: float,
        size: int = 1,
        random_state: int = None
    ) -> Union[float, List[float]]:
    """
    Utility function to sample from a lognormal distribution

    Parameters:
    - median (float): median taken from empirical data
    - dispersion (float): standard deviation taken from empirical data
    - size (int) (optional): number of samples to take
    - random_state (int) (optional): random seed used for testing
    Returns:
    - Sampled list or item from specified distribution

    """
    # Convert median and dispersion to shape (σ) and scale for lognormal
    shape = np.log(dispersion**2 + 1)  # Shape parameter
    scale = median
    if size == 1:                 # Median is the scale parameter
        res = lognorm(s=shape, scale=scale).rvs(size=size, random_state=random_state).item()
    else:
        res = lognorm(s=shape, scale=scale).rvs(size=size, random_state=random_state)

    return res

def sample_repair_time(
    mean: float,
    std_dev: float,
    random_seed: int,
    size: int = 1
) -> Union[int, List[int]]:
    """
    Samples repair times based on a normal distribution around the given mean.

    Parameters:
    - mean (float): The mean repair time in days.
    - std_dev (float): The standard deviation for the repair time distribution.
    - random_seed (int): The seed for reproducibility.
    - size (int, optional): The number of samples to draw. Default is 1.

    Returns:
    - Union[int, List[int]]: A single sampled repair time or a list of repair times.
    """

    if mean <= 0 or std_dev <= 0:
        return 0

    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed=random_seed)

    # Sample from a normal distribution around the mean with the given standard deviation
    samples = rng.normal(loc=mean, scale=std_dev, size=size)

    # Round to nearest integer and ensure non-negative values
    samples = np.maximum(np.round(samples), 0).astype(int)

    return samples[0] if size == 1 else samples.tolist()


def get_project_root(folder: Path = None) -> Path:
    """
    Get the absolute path of the given folder or the script's directory.

    Parameters:
        folder (Path): The folder to get the absolute path for. Defaults to None, which
                    will return the directory of the current script.

    Returns:
        Path: The absolute path of the provided folder or the script's directory.
    """
    if folder is None:
        return Path(__file__).resolve().parent.absolute()

    else:
        return folder.resolve().parent.absolute()

class PathUtils:
    """
    Utils for file names
    """

    # FOLDERS

    # Input folders
    building_geo_data_folder = get_project_root() / "buildings_geo_data"
    roads_geodata_folder = get_project_root() / "roads_geo_data"
    traffic_input_networks_folder = get_project_root() / "traffic_tntp_networks"
    traffic_processed_networks_folder = get_project_root() / "traffic_processed_networks"
    traffic_geo_neworks_folder = get_project_root() / "traffic_geo_networks"

    bridges_folder = get_project_root() / "bridges"

    earthquake_model_folder = get_project_root() / "earthquake"

    # FILES

    # Building files / Roads files
    buildings_nsi_anaheim_shp = building_geo_data_folder / "nsi_anaheim.shp"
    buildings_study_anaheim = building_geo_data_folder / "anaheim_buildings.shp"
    roads_anaheim_shp =  roads_geodata_folder / "anaheim_roads.shp"
    buildings_toy_shp = building_geo_data_folder / "toy_city_buildings.shp"
    buildings_toy_shp_2 = building_geo_data_folder / "toy_city_buildings_v_2.shp"
    buildings_toy_shp_3 = building_geo_data_folder / "toy_city_buildings_v_3.geojson"
    buildings_toy_shp_4 = building_geo_data_folder / "toy_city_buildings_v_4.geojson"
    roads_toy_shp =  roads_geodata_folder / "toy_city_roads.shp"
    roads_toy_shp_2 =  roads_geodata_folder / "toy_city_roads_v_2.shp"
    roads_toy_shp_3 =  roads_geodata_folder / "toy_city_roads_v_3.geojson"
    roads_toy_shp_4 =  roads_geodata_folder / "toy_city_roads_v_4.geojson"




    # Traffic network files
    traffic__anaheim_tntp = traffic_input_networks_folder / "Anaheim_net.tntp"
    traffic_anaheim_demand_tntp = traffic_input_networks_folder / "Anaheim_trips.tntp"
    traffic_anaheim_net_csv = traffic_processed_networks_folder / "Anaheim_net.csv"
    traffic_anaheim_demand_csv = traffic_processed_networks_folder / "Anaheim_trips.csv"
    traffic_nodes_geojson_file = traffic_geo_neworks_folder / "anaheim_nodes.geojson"
    traffic_links_geojson_file = traffic_geo_neworks_folder / "anaheim.geojson"

    traffic_toy_city_geojson = traffic_geo_neworks_folder / "toy_city_traffic.geojson"
    traffic_toy_city_geojson_2 = traffic_geo_neworks_folder / "toy_city_traffic_v_2.geojson"
    traffic_toy_city_geojson_3 = traffic_geo_neworks_folder / "toy_city_traffic_v_3.geojson"
    traffic_toy_city_geojson_4 = traffic_geo_neworks_folder / "toy_city_traffic_v_4.geojson"
    traffic_toy_city_network = traffic_processed_networks_folder / "toy_city_net.csv"
    traffic_toy_city_demand = traffic_processed_networks_folder / "toy_city_demand.csv"
    traffic_toy_city_network_2 = traffic_processed_networks_folder / "toy_city_net_v_2.csv"
    traffic_toy_city_network_3 = traffic_processed_networks_folder / "toy_city_net_v_3.csv"
    traffic_toy_city_network_4 = traffic_processed_networks_folder / "toy_city_net_v_4.csv"
    traffic_toy_city_demand_2 = traffic_processed_networks_folder / "toy_city_demand_v_2.csv"
    traffic_toy_city_demand_3 = traffic_processed_networks_folder / "toy_city_demand_v_3.csv"
    traffic_toy_city_demand_4 = traffic_processed_networks_folder / "toy_city_demand_v_4.csv"


    env_data = {
        "30": {
                "buildings": buildings_toy_shp_2,
                "roads": roads_toy_shp_2,
                "traffic_links": traffic_toy_city_geojson_2,
                "traffic_net": traffic_toy_city_network_2,
                "traffic_dem": traffic_toy_city_demand_2
        },
        "10": {
                "buildings": buildings_toy_shp_3,
                "roads": roads_toy_shp_3,
                "traffic_links": traffic_toy_city_geojson_3,
                "traffic_net": traffic_toy_city_network_3,
                "traffic_dem": traffic_toy_city_demand_3
        },
        "4": {
                "buildings": buildings_toy_shp_4,
                "roads": roads_toy_shp_4,
                "traffic_links": traffic_toy_city_geojson_4,
                "traffic_net": traffic_toy_city_network_4,
                "traffic_dem": traffic_toy_city_demand_4
        }
    }


    # Bridges Files
    bridges_shp_file = bridges_folder / "national_bridge_inventorty_2024.shp"

    # Earthquake files
    earthquake_model_file = earthquake_model_folder / "incore_eq_model.json"




