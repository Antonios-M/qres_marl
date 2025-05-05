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


## example game environment ( 2 agents / grid world / agents get reward when reaching individual goal square)
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
        "time_step_duration": 30,
        "trucks_per_building_per_day": 0.1,
        "n_agents": 30,
        "n_crews": 25,
        "time_horizon": 20,
        "w_econ": 0.2,
        "w_crit": 0.4,
        "w_health": 0.4
    }
)

## quake reponse environment: 4 agents (2 roads + 2 buildings)
register(
    id="quake-res-4-v1",
    entry_point="quake_envs.qres_env_wrapper:Qres_env_wrapper",
    kwargs={
        "verbose": False,
        "time_step_duration": 20,
        "trucks_per_building_per_day": 0.1,
        "n_agents": 4,
        "n_crews": 4,
        "time_horizon": 20,
        "w_econ": 0.2,
        "w_crit": 0.4,
        "w_health": 0.4
    }
)



# import gymnasium as gym
# env = gym.make("quake-res-30-v1")
# actions_buildings = np.random.randint(0, 4, 15)  # Assuming 4 possible actions per building
# actions_roads = np.random.randint(0, 3, 15)  # Assuming 3 possible actions per road


# actions = (actions_buildings, actions_roads)
# env.reset(seed=random.choice([0,1,2,3]))
# observation, reward, done, info = env.step(actions)
# done = False