import numpy as np
import matplotlib.pyplot as plt
import quake_envs
from quake_envs.simulations.building_funcs import Building, BuildingAction
from quake_envs.simulations.resilience import Resilience

# Initialize the building environment
building = Building(
  id="test_building",
  damage_state_probs=np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
  occtype="COM6",
  str_type="S5L",
  num_stories=2,
  sqft=1000,
  is_essential=False,
  verbose=False
)

res = Resilience(
  sum_initial_income=building.max_income,
  sum_current_income=building.current_income,
  sum_current_critical_funcs=building.current_critical_func,
  sum_initial_critical_funcs=building.initial_critical_func,
  sum_current_beds=building.current_beds,
  sum_initial_beds=building.initial_beds,
  sum_current_doctors=building.current_doctors,
  sum_initial_doctors=building.initial_doctors,
  costs=np.array([0, 0, 0, 0]),
)

# List to store rewards
total_rewards = []
econ_rewards = []
crit_rewards = []
health_rewards = []
time = []
ds = []
rt = []
# print(building)
# print(building._get_reward())
# Simulate 20 steps with random actions
# rewards.append(building.max_income)
for i in range(100):
  action = np.random.randint(0, 3)  # Random action for testing
  action = BuildingAction(action)

  state, r, done, info = building.step(action)
  res.step(r[2], r[3], r[4], r[5], r[0])
  total, econ, crit, health = res.q_community_decomp
  total_rewards.append(total)
  econ_rewards.append(econ)
  crit_rewards.append(crit)
  health_rewards.append(health)
  ds.append(building.current_damage_state)
  time.append(len(total_rewards))


plt.figure(figsize=(10, 6))  # Optional: make the plot larger

plt.plot(time, total_rewards, label='Total Reward', marker='o')
plt.plot(time, econ_rewards, label='Economic Reward', marker='s')
plt.plot(time, crit_rewards, label='Critical Function Reward', marker='^')
plt.plot(time, health_rewards, label='Health Reward', marker='x')
plt.plot(time, ds, label="Damage State")

plt.title(f"Building: {building.occtype},  functionality over time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
