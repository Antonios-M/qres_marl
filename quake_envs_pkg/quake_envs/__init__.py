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



register(
    id="quake-res-30-v1",
    entry_point="quake_envs.city_quake_res_30:Quake_Res_30",
    kwargs={
        "verbose": False,
    }
)

register(
    id="quake-res-10-v1",
    entry_point="quake_envs.city_quake_res_10:Quake_Res_10",
    kwargs={
        "verbose": False,
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