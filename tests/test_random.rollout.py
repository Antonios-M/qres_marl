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
# env.reset()
# env.resilience._save_env_config("tests\\environment_testing_vdn_ps")
# random_inference = HeuristicInference(name="random", env=env)
random_inference = HeuristicInference(name="importance_based", env=env)
# env.resilience.simulation.viz_environment("test")
random_inference.plot_rollout(figsize=(20,10), plot_econ_traffic=True, plot_delay=True)
# random_inference.plot_3d(figsize=(20,10))

