import gymnasium as gym
from gym import register
import quake_envs
import sys
from enum import Enum
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from gymnasium import spaces
from gymnasium.spaces import flatdim
import seaborn as sns
import copy
import itertools

import matplotlib.colors as mcolors
import geopandas as gpd
import math

from quake_envs.simulations.utils import *
# print(get_project_root())
env = gym.make("quake-res-30-v1")


# # traffic_gdf = env.simulation._traffic_links_gdf
# env.simulation.viz_environment("test", figsize=(15,15), show_road_ids=True)
env.plot_rollout(figsize=(15,10), plot_econ_income=True)