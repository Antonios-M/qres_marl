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
from imprl.post_process.inference import HeuristicInference

# print(get_project_root())
env = gym.make("quake-res-4-v1")
save_env_config = 0
_baselines = ["random", "do_nothing", "importance_based"]
baseline = _baselines[0]
plot_rollout = 1
plot_3d = 0
viz_environment = 0
plot_buildings = 1

if plot_buildings:
    inference = HeuristicInference(name=baseline, env=env)
    inference.plot_components()

if save_env_config:
    env.reset()
    env.resilience._save_env_config("tests\\environment_testing_vdn_ps")

if viz_environment:
    env.resilience.simulation.viz_environment("test", show_road_ids=True, show_traffic_ids=True)

if plot_rollout:
    inference = HeuristicInference(name=baseline, env=env)
    inference.plot_rollout(figsize=(20,10), plot_econ_traffic=False, plot_delay=True)

if plot_3d:
    inference = HeuristicInference(name=baseline, env=env)
    inference.plot_3d(figsize=(20,10))
