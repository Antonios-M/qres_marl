import numpy as np
import matplotlib.pyplot as plt
import quake_envs
from quake_envs.simulations.road_funcs import Road, RoadAction
import seaborn as sns

# Initialize the road environment
road = Road(
  id="test_road",
  init_node=0,
  term_node=1,
  flow=0.0,
  damage_state=4,
  capacity=1000,
  length_miles=1.0,
  hazus_road_class="HRD1",
  hazus_bridge_class="HWB8",
  is_bridge=False,
  capacity_red_debris=0.0,
  capacity_red_damage_state=1.0,
  time_step_duration=20,
  traffic_idx=0,
  verbose=False,
  stoch_ds=False,
  calc_debris=False,
  stoch_rt=False,
  stoch_cost=False
)
def normalize_to_range_0_4(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        # Avoid division by zero; all values are the same
        return [0 for _ in values]
    return [4 * (v - min_val) / (max_val - min_val) for v in values]
rt = []
rc = []
ds = []
time = []
cr = []
i = -1
t_rep = -1
while not road.is_fully_repaired:
    i += 1
    info = road.step(RoadAction.REPAIR, [])
    rt.append(road.current_repair_time)
    rc.append(road.current_repair_cost)
    ds.append(road.current_damage_state)
    cr.append(road.capacity_red_damage_state)
    time.append(i)

rt = normalize_to_range_0_4(rt)
rc = normalize_to_range_0_4(rc)
ds = normalize_to_range_0_4(ds)
cr = normalize_to_range_0_4(cr)

# Set seaborn style
sns.set_theme(style="whitegrid", context="notebook")

# Create the plot
plt.figure(figsize=(20, 10))

# Background shading
plt.axvspan(xmin=0, xmax=max(time), color='red', alpha=0.1, label='_nolegend_')    # Between debris cleared and repair


# Plot each line with seaborn color palette
palette = sns.color_palette("tab10", 6)
plt.plot(time, ds, label="Damage State", marker='o', color=palette[1])
plt.plot(time, rt, label="Repair Time", marker='s', color=palette[2])
plt.plot(time, rc, label="Repair Cost", marker='^', color=palette[3])
plt.plot(time, cr, label="Relocation Cost", marker='x', color=palette[4])

# Add labels, legend, title
plt.title(f"Bridge: {road.bridge_class}, Performance Over Repair Progress", fontsize=16)
plt.xlabel("Step", fontsize=14)
plt.ylabel("Performance", fontsize=14)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()