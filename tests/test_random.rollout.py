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
import matplotlib.cm as cm

import matplotlib.colors as mcolors
import geopandas as gpd
import math
from scipy.stats import ttest_ind

from quake_envs.simulations.utils import *
from imprl.post_process.inference import HeuristicInference
from imprl.post_process.inference import AgentInference

env_4 = gym.make("quake-res-4-v1")
env_30 = gym.make("quake-res-30-v1")
save_env_config = 1
_baselines = ["random", "do_nothing", "importance_based"]
baseline = _baselines[2]
viz_environment = 0
plot_rollout = 0
plot_3d = 0
plot_buildings = 0
run_n = 0
run_avoided_losses = 1
eval_single = True

curr = "v8-cal-loss-reward-env-fixed" ## 45000
curr_30 = "v8-cal-loss-reward-30-components"
curr_30_rfix = "v9-30-reward-fix"  #60000
v8 = "v8_DCMAC" #45000
v9 = "v9-DCMAC" #70000
v7_qmix = "v7_QMIX_PS" #99999
v7_vdn_ps = "v7" #99999
v9_loss_reward = "v9-30-loss-reward-DCMAC" #99999

env = env_30

# env.resilience.simulation.viz_environment(plot_name="test")
def get_trained_agent_dir(name=None,v=v9_loss_reward, env=env_30, checkpt=45000):
    error = False
    if name == "DCMAC":
        if env.n_agents == 4:
            v = curr
            checkpt = 45000
        else:
            v = v9
            checkpt = 70000
    elif name == "QMIX_PS":
        if env.n_agents == 4:
            v = v7_qmix
            checkpt = 99999
        else:
            error = True
    elif name == "VDN_PS":
        if env.n_agents == 4:
            v = v7_vdn_ps
            checkpt = 99999
        else:
            error = True

    if error:
        print(f"Error: The configuration for '{name}' with {env.n_agents} agents is not supported.")
        return None, None # Example: return None to indicate failure

    base_path = "wandb"
    return base_path + "\\" + v + "\\files", checkpt



def plot_avoided_losses(inference="all", bins=50, n=10,
                        figsize=(12, 10), save_path=None):
    """
    Plots cumulative-loss (CL) ratio distributions for any number of policies.
     • One subplot per policy for its histogram + KDE.
     • A final, separate subplot for the KDE overlay of all policies.
    """
    # This setup allows the function to run standalone for testing

    # --- Change 1: Create a robust mapping from policy name to title ---
    # This prevents errors if policies are skipped or reordered.
    policy_title_map = {
        "random": "Random",
        "importance_based": "IMPB ",
        "DCMAC": "DCMAC",
        "QMIX_PS": "QMIX_PS",
        "VDN_PS": "VDN_PS"
    }

    if inference == "all":
        policy_list = list(policy_title_map.keys())
    else:
        policy_list = [inference] if isinstance(inference, str) else inference

    # Separate lists for magma and cividis policies
    magma_policies = {"random", "importance_based"}
    cividis_policies = [p for p in policy_list if p not in magma_policies]
    magma_policies = [p for p in policy_list if p in magma_policies]

    # Get colormaps
    cmap_cividis = cm.get_cmap("cividis")
    cmap_magma = cm.get_cmap("magma")

    # Assign positions in the colormap
    positions_cividis = np.linspace(0.25, 0.75, len(cividis_policies)) if cividis_policies else []
    positions_magma = np.linspace(0.25, 0.75, len(magma_policies)) if magma_policies else []

    # Create color mappings
    policy_colors = {
        **{name: cmap_cividis(p) for name, p in zip(cividis_policies, positions_cividis)},
        **{name: cmap_magma(p) for name, p in zip(magma_policies, positions_magma)},
    }

    all_ratios = {}
    # First pass: collect all valid ratios for bin range calculation
    for name in policy_list:
        print(f"Running inference for policy: {name}")
        if name in ["DCMAC", "QMIX_PS", "VDN_PS"]:
            agent = AgentInference(name, env)
            checkpoint_path, episode = get_trained_agent_dir(name=name)
            agent.load_weights(checkpoint_path, episode)
        else:
            agent = HeuristicInference(name=name, env=env)

        agent.get_n_rollouts(n=n)
        losses = agent.plotter.batch_losses
        resilience = agent.plotter.batch_resilience

        if not losses or not resilience or len(losses) != len(resilience):
            print(f"Skipping '{name}' due to invalid rollout data.")
            continue

        ratios = [-sum(l) for l, r in zip(losses, resilience) if (sum(l) + sum(r)) > 0]
        if ratios:
            all_ratios[name] = ratios
        else:
            print(f"No valid data for '{name}'")

    if not all_ratios:
        print("No valid data across all policies.")
        return

    # Filter the policy list to only include those with data
    active_policies = list(all_ratios.keys())
    num_policies = len(active_policies)

    # Determine global min and max for consistent bin ranges
    all_values = [v for ratios in all_ratios.values() for v in ratios]
    min_val, max_val = min(all_values), max(all_values)
    bins_range = np.linspace(min_val, max_val, bins + 1)

    # --- Change 2: Dynamically calculate grid size ---
    # We need one plot for each policy, plus one for the KDE overlay.
    total_plots = num_policies + 1
    ncols = 2 # Let's keep 2 columns for a neat layout
    nrows = math.ceil(total_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    sns.set_style("whitegrid")

    # Plot each individual histogram + KDE
    for i, name in enumerate(active_policies):
        ax = axes[i]
        ratios = all_ratios[name]
        mean_val, std_val = np.mean(ratios), np.std(ratios)

        sns.histplot(ratios, bins=bins_range, ax=ax, stat="density", kde=True,
                     edgecolor="black", color=policy_colors[name], alpha=0.7)
        ax.axvline(mean_val, color=policy_colors[name], linestyle="--", linewidth=1)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.1)
        ax.set_xlim(min_val, max_val)

        ax.text(mean_val, ymax * 1.05, f"μ={mean_val:.2f}\nσ={std_val:.2f}",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))

        ax.set_xlabel("Losses")
        ax.set_ylabel("Density")
        ax.set_title(policy_title_map.get(name, name)) # Use the map for the title

    # --- Change 3: Plot KDE overlay on the correct, dedicated subplot ---
    kde_ax = axes[num_policies]
    for name, ratios in all_ratios.items():
        sns.kdeplot(ratios, label=policy_title_map.get(name, name), ax=kde_ax,
                    color=policy_colors[name], fill=True, alpha=0.3)

    kde_ax.set_xlim(min_val, max_val)
    kde_ax.set_xlabel("Losses")
    kde_ax.set_ylabel("Density")
    kde_ax.set_title("Policy-wise Losses KDE")
    kde_ax.legend()
    kde_ax.grid(True, linestyle="--", alpha=0.6)

    # --- Change 4: Hide any unused subplots for a clean look ---
    for i in range(total_plots, len(axes)):
        axes[i].axis('off')

    fig.suptitle(
        r"$\bf{{Cumulative\ Losses\ (CL)}}$"
        f" Distributions per Policy, over {n} Rollouts, toy-city-{env.n_agents}",
        fontsize=14, fontfamily="serif"
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=300)

    plt.show()
# def plot_avoided_losses(env, inference="all", bins=20, n=10,
#                              figsize=(14, 12), save_path=None):
#     """
#     Full visualization: CAL histograms, KDEs, and statistical bar comparison.
#     """
#     # ---- Policies ----------------------------------------------------------
#     policy_list = ["random", "importance_based", "DCMAC"]
#     titles = ["Random",
#               "Importance Based",
#               "Deep Centralised Multi-Agent Actor-Critic (DCMAC)"]
#     if inference != "all":
#         policy_list = inference if isinstance(inference, list) else [inference]

#     cmap = cm.get_cmap("magma")
#     positions = np.linspace(0.25, 0.75, len(policy_list))
#     policy_colors = {n: cmap(p) for n, p in zip(policy_list, positions)}
#     bins_range = np.linspace(0, 1, bins + 1)

#     # ---- Run Inference + Collect CAL ---------------------------------------
#     all_ratios = {}
#     for name in policy_list:
#         if name == "DCMAC":
#             agent = AgentInference(name, env)
#             checkpoint_path, episode = get_trained_agent_dir()
#             agent.load_weights(checkpoint_path, episode)
#         else:
#             agent = HeuristicInference(name=name, env=env)

#         agent.get_n_rollouts(n=n)
#         losses = agent.plotter.batch_losses
#         resilience = agent.plotter.batch_resilience

#         if not losses or not resilience or len(losses) != len(resilience):
#             print(f"Skipping '{name}' due to invalid rollout data.")
#             continue

#         ratios = [sum(r) / (sum(l) + sum(r))
#                   for l, r in zip(losses, resilience)
#                   if (sum(l) + sum(r)) > 0]

#         if not ratios:
#             print(f"No valid data for '{name}'")
#             continue

#         all_ratios[name] = ratios

#     # ---- Plot Histograms + KDEs --------------------------------------------
#     fig, axes = plt.subplots(3, 2, figsize=figsize)
#     sns.set_style("whitegrid")

#     for i, name in enumerate(policy_list):
#         ax = axes[i][0]
#         ratios = all_ratios.get(name, [])
#         if not ratios:
#             continue
#         mean_val, std_val = np.mean(ratios), np.std(ratios)

#         sns.histplot(ratios, bins=bins_range, ax=ax, stat="density", kde=True,
#                      color=policy_colors[name], edgecolor="black", alpha=0.7)

#         ax.axvline(mean_val, color=policy_colors[name], linestyle="--", linewidth=1)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(top=ax.get_ylim()[1] * 1.5)

#         ax.text(mean_val, ax.get_ylim()[1] * 0.95,
#                 f"μ={mean_val:.2f}\nσ={std_val:.2f}",
#                 ha="center", fontsize=9,
#                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

#         ax.set_title(f"{titles[i]} — CAL")
#         ax.set_xlabel("CAL")
#         ax.set_ylabel("Density")

#     # ---- KDE Overlay -------------------------------------------------------
#     ax_kde = axes[0][1]
#     for name in policy_list:
#         ratios = all_ratios.get(name, [])
#         if ratios:
#             sns.kdeplot(ratios, label=name, ax=ax_kde,
#                         color=policy_colors[name], shade=True, alpha=0.5)
#     ax_kde.set_xlim(0, 1)
#     ax_kde.set_title("KDE Overlay — All Policies")
#     ax_kde.legend()
#     ax_kde.set_xlabel("CAL")
#     ax_kde.set_ylabel("Density")
#     ax_kde.grid(True, linestyle="--", alpha=0.6)

#     # ---- Bar Plot + T-Test Comparisons -------------------------------------
#     ax_bar = axes[1][1]

#     means = [np.mean(all_ratios[name]) for name in policy_list]
#     stds = [np.std(all_ratios[name]) for name in policy_list]
#     x_pos = np.arange(len(policy_list))

#     bars = ax_bar.bar(x_pos, means, yerr=stds, capsize=8,
#                       color=[policy_colors[n] for n in policy_list], alpha=0.8)

#     ax_bar.set_xticks(x_pos)
#     ax_bar.set_xticklabels(titles, rotation=20, ha="right")
#     ax_bar.set_ylabel("Mean CAL")
#     ax_bar.set_ylim(0, 1)
#     ax_bar.set_title("Policy Comparison — Mean CAL with Std Dev")
#     ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)

#     # Pairwise T-Test Annotations
#     pairs = [("random", "importance_based"),
#              ("importance_based", "DCMAC"),
#              ("random", "DCMAC")]

#     y_offset = 0.02
#     for i, (a, b) in enumerate(pairs):
#         x1, x2 = policy_list.index(a), policy_list.index(b)
#         y = max(means[x1], means[x2]) + 0.08 + i * y_offset

#         t_stat, p_val = ttest_ind(all_ratios[a], all_ratios[b], equal_var=False)
#         delta = np.mean(all_ratios[b]) - np.mean(all_ratios[a])
#         annotation = f"Δ={delta:.2f}, p={p_val:.3f}"

#         # ax_bar.plot([x1, x1, x2, x2], [y-0.005, y, y, y-0.005], lw=1.2, c='gray')
#         ax_bar.text((x1 + x2) / 2, y + 0.005, annotation,
#                     ha='center', va='bottom', fontsize=8,
#                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

#     # ---- Final Layout ------------------------------------------------------
#     fig.suptitle(
#         r"$\bf{Avoided\ Loss\ Ratio\ (CAL)}$"
#         f" — Distribution and Comparison over {n} Rollouts per Policy",
#         fontsize=15, fontfamily="serif"
#     )
#     plt.tight_layout(rect=[0, 0.04, 1, 0.96])

#     if save_path:
#         fig.savefig(save_path, dpi=300)

#     plt.show()


def plot_relative_performance(inference="all", n=10, figsize=(10, 6), save_path=None):
    """
    Plots relative performance ((x - H)/H * 100) of each policy compared to the 'importance_based' baseline.
    Also shows the mean cumulative return per policy.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    policy_title_map = {
        "random": "Random",
        "importance_based": "IMPB ",
        "DCMAC": "DCMAC",
        "QMIX_PS": "QMIX_PS",
        "VDN_PS": "VDN_PS"
    }

    if inference == "all":
        policy_list = list(policy_title_map.keys())
    else:
        policy_list = [inference] if isinstance(inference, str) else inference

    # Colormap setup
    magma_policies = {"random", "importance_based"}
    cividis_policies = [p for p in policy_list if p not in magma_policies]
    magma_policies = [p for p in policy_list if p in magma_policies]

    cmap_cividis = cm.get_cmap("viridis")
    cmap_magma = cm.get_cmap("magma")

    positions_cividis = np.linspace(0.25, 0.75, len(cividis_policies)) if cividis_policies else []
    positions_magma = np.linspace(0.25, 0.75, len(magma_policies)) if magma_policies else []

    policy_colors = {
        **{name: cmap_cividis(p) for name, p in zip(cividis_policies, positions_cividis)},
        **{name: cmap_magma(p) for name, p in zip(magma_policies, positions_magma)},
    }

    # Collect returns
    policy_returns = {}
    for name in policy_list:
        print(f"Evaluating policy: {name}")
        if name in ["DCMAC", "QMIX_PS", "VDN_PS"]:
            agent = AgentInference(name, env)
            checkpoint_path, episode = get_trained_agent_dir(name=name)
            agent.load_weights(checkpoint_path, episode)
        else:
            agent = HeuristicInference(name=name, env=env)

        agent.get_n_rollouts(n=n)
        losses = agent.plotter.batch_losses
        resilience = agent.plotter.batch_resilience

        if not losses or not resilience or len(losses) != len(resilience):
            print(f"Skipping '{name}' due to invalid rollout data.")
            continue
        # print(losses)
        returns = [-sum(l) for l, r in zip(losses, resilience) if (sum(l) + sum(r)) > 0]
        if returns:
            policy_returns[name] = returns
        else:
            print(f"No valid returns for {name}")

    # Check baseline
    if "importance_based" not in policy_returns:
        print("Missing 'importance_based' baseline. Cannot compute relative performance.")
        return

    baseline_mean = np.mean(policy_returns["importance_based"])

    # Prepare DataFrame for plotting
    data = []
    for i, (name, values) in enumerate(policy_returns.items()):
        mean_return = np.mean(values)
        rel_diff = ((mean_return - baseline_mean) / baseline_mean) * 100 if name != "importance_based" else 0
        data.append({
            "Policy": policy_title_map.get(name, name),
            "Relative (%)": rel_diff,
            "Mean Return": mean_return,
            "Color": policy_colors[name]
        })

    df = pd.DataFrame(data)
    df = df[df["Policy"] != policy_title_map["importance_based"]]  # Exclude baseline from bars
    df.sort_values("Relative (%)", inplace=True)

    # Plotting
    sns.set(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=figsize)

    # Build data with min/max/mean relative returns
    summary = []
    for i, (name, values) in enumerate(policy_returns.items()):
        if name == "importance_based":
            continue
        rel_values = [((v - baseline_mean) / baseline_mean) * 100 for v in values]
        summary.append({
            "Policy": policy_title_map.get(name, name),
            "Min": np.min(rel_values),
            "Max": np.max(rel_values),
            "Mean": np.mean(rel_values),
            "Color": policy_colors[name]
        })

    df = pd.DataFrame(summary)
    df.sort_values("Mean", inplace=True)

    y_positions = np.arange(len(df))
    bar_height = 0.6

    for i, row in df.iterrows():
        # Draw range bar
        ax.barh(
            y=y_positions[i],
            width=row["Max"] - row["Min"],
            left=row["Min"],
            height=bar_height,
            color=row["Color"],
            edgecolor="black"
        )
        # Draw vertical line at the mean
        ax.plot(
            [row["Mean"], row["Mean"]],
            [y_positions[i] - bar_height / 2, y_positions[i] + bar_height / 2],
            color="black",
            linewidth=2,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["Policy"])
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Relative Performance vs IMPB", fontsize=14, fontweight="bold")
    ax.set_xlabel("Normalised Relative Returns (x-H)/H")
    ax.set_ylabel("")
    ax.set_xlim(-100, 100)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()
# agent.run_n(2)
# plot_avoided_losses(inference=["random", "importance_based", "DCMAC"], n=1000)
# plot_relative_performance(inference=["random", "importance_based", "DCMAC"], n=1000, figsize=(12, 8))

def plot_single_rollout(inference="all", plot_components=True):
    if inference == "all":
        policy_list = ["random", "importance_based", "DCMAC", "QMIX_PS"]
    elif isinstance(inference, list):
        policy_list = inference
    else:
        policy_list = [inference]

    for name in policy_list:
        if name in ["DCMAC", "QMIX_PS"]:
            agent = AgentInference(name, env)
            checkpoint_path, episode = get_trained_agent_dir()
            # checkpoint_path = r"C:\Users\Anton\OneDrive\Desktop\tudelft_thesis\qres_marl\wandb\v8-cal-loss-reward-env-fixed\files"
            # episode = 40000
            agent.load_weights(checkpoint_path, episode)
        else:
            agent = HeuristicInference(name=name, env=env)

        agent.get_rollout(save=True)
        agent.plot_rollout(figsize=(20,10), plot_econ_traffic=True, plot_delay=True,
    plot_econ_relocation=True)
        if plot_components:
            agent.plot_components(figsize=(20,20))

plot_single_rollout(inference="DCMAC", plot_components=True)


if save_env_config:
    env.reset()
    env.resilience._save_env_config("tests\\environment_testing_vdn_ps")

if viz_environment:
    env.resilience.simulation.viz_environment("test", show_road_ids=True, show_traffic_ids=True)

