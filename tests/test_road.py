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


# def plot_avoided_losses_matrix(inference="all", bins=20, n=10,
#                                figsize=(12, 6), save_path=None):
#     """
#     Plots a 2D heatmap matrix: CAL ratio (x-axis) vs earthquake magnitudes (y-axis).
#     Color intensity = ratio of (instances in bin) to number of rollouts.
#     Earthquake bins use 0.5 steps and include both low and high values.
#     """
#     policy_list = ["random", "importance_based", "DCMAC"]
#     titles = ["Random", "Importance Based", "DCMAC"]

#     if inference != "all":
#         policy_list = inference if isinstance(inference, list) else [inference]
#         titles = policy_list

#     cmap = cm.get_cmap("viridis")  # perceptually uniform

#     fig, axes = plt.subplots(1, len(policy_list), figsize=figsize, sharey=True)
#     if len(policy_list) == 1:
#         axes = [axes]

#     for i, name in enumerate(policy_list):
#         ax = axes[i]

#         # --- inference code --------------------------------------
#         if name == "DCMAC":
#             agent = AgentInference(name, env)
#             checkpoint_path, episode = get_trained_agent_dir()
#             agent.load_weights(checkpoint_path, episode)
#         else:
#             agent = HeuristicInference(name=name, env=env)

#         agent.get_n_rollouts(n=n)
#         losses      = agent.plotter.batch_losses
#         resilience  = agent.plotter.batch_resilience
#         quake_mags  = agent.plotter.batch_mags
#         # ---------------------------------------------------------

#         if not losses or not resilience or not quake_mags:
#             print(f"Skipping '{name}' due to invalid data.")
#             continue

#         cal_vals = []
#         mags = []
#         for l, r, q in zip(losses, resilience, quake_mags):
#             total = sum(l) + sum(r)
#             if total > 0:
#                 cal = sum(r) / total
#                 cal_vals.append(cal)
#                 mags.append(q)

#         if not cal_vals or not mags:
#             print(f"No valid CAL or magnitude data for '{name}'")
#             continue

#         # Define fixed bin edges for CAL and quake magnitude
#         cal_bins = np.linspace(0, 1, bins + 1)

#         mag_min, mag_max = np.floor(min(mags)), np.ceil(max(mags))
#         mag_bins = np.arange(mag_min, mag_max + 0.5, 0.5)  # 0.5 step, inclusive

#         # Create 2D histogram
#         hist, xedges, yedges = np.histogram2d(cal_vals, mags,
#                                               bins=[cal_bins, mag_bins])

#         # Normalize by rollouts
#         hist /= n

#         # Plot
#         mesh = ax.pcolormesh(xedges, yedges, hist.T, cmap=cmap,
#                              shading='auto')
#         cbar = fig.colorbar(mesh, ax=ax)
#         cbar.set_label("Ratio to Rollouts")

#         ax.set_xlabel("CAL")
#         if i == 0:
#             ax.set_ylabel("Earthquake Magnitude")
#         ax.set_title(titles[i])
#         ax.set_xlim(0, 1)

#     fig.suptitle(
#         rf"$\bf{{CAL\ Matrix}}$: CAL vs Quake Magnitude (0.5 steps), Normalized by {n} Rollouts, toy-city-{env.n_agents}",
#         fontsize=14, fontfamily="serif"
#     )
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if save_path:
#         fig.savefig(save_path, dpi=300)

#     plt.show()