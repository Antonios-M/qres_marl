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

random_inference = HeuristicInference(name="importance_based", env=env)
# random_inference = HeuristicInference(name="random", env=env)
# env.resilience.simulation.viz_environment("test")
random_inference.plot_rollout(figsize=(10,10), title_intensity="max")

