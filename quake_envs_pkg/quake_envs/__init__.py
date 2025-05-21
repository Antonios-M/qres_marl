from gymnasium import register
import sys
from pathlib import Path
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import geopandas as gpd
import os

from .simulations.utils import *
from .simulations.building_config import *
from .simulations.building_funcs import *
from .simulations.road_config import *
from .simulations.road_funcs import *
from .simulations.interdep_network import *
from .simulations.interdependencies import *
from .simulations.traffic_assignment import *


## example game environment ( 2 agents / grid world / agents get reward when reaching individual goal square), use to test multi-agent RL algorithms
register(
  id="ma-grid-world-v0",
  entry_point="quake_envs.ma_grid_world:SimpleGridWorld",
  kwargs={
    "grid_size": 5,
  }
)

## quake response environment: 30 agents (15 roads + 15 buildings)
register(
    id="quake-res-30-v1",
    entry_point="quake_envs.qres_env_wrapper:Qres_env_wrapper",
    kwargs={
        "verbose": False,
        "time_step_duration": 150,
        "trucks_per_building_per_day": 0.1,
        "n_agents": 30,
        "n_crews": 10,
        "time_horizon": 20,
        "w_econ": 0.2,
        "w_crit": 0.5,
        "w_health": 0.5,
        "quake_choices": [7.5, 8.0, 8.5]
    }
)

## quake reponse environment: 4 agents (2 roads + 2 buildings)
register(
    id="quake-res-4-v1",
    entry_point="quake_envs.qres_env_wrapper:Qres_env_wrapper",
    kwargs={
        "verbose": False,
        "time_step_duration": 10,
        "trucks_per_building_per_day": 0.1,
        "n_agents": 4,
        "n_crews": 2,
        "time_horizon": 100,
        "w_econ": 0.2,
        "w_crit": 0.5,
        "w_health": 0.3,
        "quake_choices": [7.5, 8.0, 8.5]
    }
)

register(
    id="quake-res-6000-v1",
    entry_point="quake_envs.qres_env_wrapper:Qres_env_wrapper",
    kwargs={
        "verbose": False,
        "time_step_duration": 40,
        "trucks_per_building_per_day": 0.1,
        "n_agents": 6000,
        "n_crews": 3000,
        "time_horizon": 50,
        "w_econ": 0.2,
        "w_crit": 0.5,
        "w_health": 0.3,
        "quake_choices": [7.5, 8.0, 8.5]
    }
)


