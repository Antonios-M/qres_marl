import numpy as np
import matplotlib.pyplot as plt
import quake_envs
from quake_envs.simulations.building_funcs import Building, BuildingAction
from quake_envs.simulations.resilience import Resilience
import seaborn as sns
import gymnasium as gym
from shapely.geometry import Polygon

geo = Polygon([(0,0), (100,0), (100, 100), (0,100)])

bldg = Building(
  id="test_building",
  geometry=geo,
  occtype="COM2",
  str_type="S5L",
  num_stories=2,
  sqft=1000,
  is_essential=False,
  verbose=False,
  calc_debris=True,
  stoch_ds=True,
  stoch_rt=True,
  stoch_cost=True,
  stoch_inc_loss=True,
  stoch_loss_of_function=True,
  stoch_relocation_cost=True
)
bldg.reset(damage_state_probs=np.array([0.0, 0.0, 0.0, 1.0, 0.0]), debris_capacity_reduction=0.0)
print(bldg.current_damage_state)
print(bldg.current_repair_time)
print(bldg.current_structural_repair_cost)
print(bldg.max_rep_cost)
print(bldg.current_income)
print(bldg.current_relocation_cost)
print(bldg.max_reloc_cost)
