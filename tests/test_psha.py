import numpy as np
import pandas as pd
import geopandas as gpd
from quake_envs.simulations.utils import *
from quake_envs.simulations.psha2 import SeismicSourceZone
from openquake.hazardlib.imt import PGA, SA
import quake_envs
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import collections

env = gym.make("quake-res-4-v1").unwrapped
env.reset()
bldg_centers = env.resilience.buildings_objs
road_centers = env.resilience.road_objs

centers = []
for bldg in bldg_centers:
    centers.append(bldg.centroid)

for road in road_centers:
    centers.append(road.centroid)


ucerf3 = gpd.read_file(PathUtils.faults)

psha = SeismicSourceZone(ucerf3, centers)
max_return_period = psha.get_max_return_period()


src_model = psha.select_seismic_source(return_period=5000)
src = src_model
site_wise_res = psha.attenuate(src)
np.set_printoptions(suppress=True, precision=6)
print(site_wise_res)

print(src)
