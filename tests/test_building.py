import numpy as np
import matplotlib.pyplot as plt
import quake_envs
from quake_envs.simulations.building_funcs import Building, BuildingAction
from quake_envs.simulations.resilience import Resilience
import seaborn as sns

# Initialize the building environment
building = Building(
  id="test_building",
  damage_state_probs=np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
  occtype="RES3A",
  str_type="PC2H",
  num_stories=2,
  sqft=1000,
  time_step_duration=20,
  trucks_per_day=0.05,
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
print(f"Repair Time: {building.initial_repair_time}")
# List to store rewards
total_rewards = []
econ_rewards = []
crit_rewards = []
health_rewards = []
time = []
ds = []
rt = []
rc = []
reloc = []
inc = []
i = -1
deb = []
t_repair = -1
t_debris = -1
while not building.is_functional:
  # print(building.current_relocation_cost)
  i += 1
  # if building.has_debris:
  #   action = BuildingAction.CLEAR_DEBRIS
  # actions = [BuildingAction.DO_NOTHING, BuildingAction.REPAIR]
  # action = np.random.choice(actions)
  action = BuildingAction.REPAIR
  if building.has_debris:
    deb.append(4.0)
  else:
    deb.append(0.0)
  if t_repair == -1:
    if building.is_fully_repaired:
        t_repair = i-1
  if t_debris == -1:
    if not building.has_debris:
        t_debris = i-1
  info = building.step(action)
  time.append(i)
  inc.append(building.current_income)
  ds.append(building.current_damage_state)
  rt.append(building.current_repair_time)
  rc.append(building.current_structural_repair_cost)
  reloc.append(building.current_relocation_cost)



def normalize_to_range_0_4(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        # Avoid division by zero; all values are the same
        return [0 for _ in values]
    return [4 * (v - min_val) / (max_val - min_val) for v in values]

rt = normalize_to_range_0_4(rt)
rc = normalize_to_range_0_4(rc)
reloc = normalize_to_range_0_4(reloc)
inc = normalize_to_range_0_4(inc)
deb = normalize_to_range_0_4(deb)

# Set seaborn style
sns.set_theme(style="whitegrid", context="notebook")

# Create the plot
plt.figure(figsize=(20, 10))

# Background shading
plt.axvspan(xmin=time[0], xmax=t_debris, color='orange', alpha=0.1, label='_nolegend_')  # Pre-debris
plt.axvspan(xmin=t_debris, xmax=t_repair, color='red', alpha=0.1, label='_nolegend_')    # Between debris cleared and repair
plt.axvspan(xmin=t_repair, xmax=time[-1], color='green', alpha=0.1, label='_nolegend_')  # Post-repair


# Plot each line with seaborn color palette
palette = sns.color_palette("tab10", 6)
plt.plot(time, deb, label="Debris", marker='*', color=palette[0])
plt.plot(time, ds, label="Damage State", marker='o', color=palette[1])
plt.plot(time, rt, label="Repair Time", marker='s', color=palette[2])
plt.plot(time, rc, label="Repair Cost", marker='^', color=palette[3])
plt.plot(time, reloc, label="Relocation Cost", marker='x', color=palette[4])
plt.plot(time, inc, label="Income", marker='d', color=palette[5])


plt.axvline(x=t_repair, color='red', linestyle='--', linewidth=1, label='Repair Completed')
plt.axvline(x=t_debris, color="orange", linestyle="--", linewidth=1, label='Debris Cleared')

# Add labels, legend, title
plt.title(f"Building: {building.occtype}, Performance Over Repair Progress", fontsize=16)
plt.xlabel("Step", fontsize=14)
plt.ylabel("Performance", fontsize=14)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()