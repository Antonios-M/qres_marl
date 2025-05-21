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

env = gym.make("quake-res-4-v1")
env_2 = gym.make("quake-res-30-v1")
save_env_config = 0
_baselines = ["random", "do_nothing", "importance_based"]
baseline = _baselines[2]
viz_environment = 0
plot_rollout = 0
plot_3d = 0
plot_buildings = 0
run_n = 0
run_avoided_losses = 1
eval_single = True

curr = "v8-cal-loss-reward-env-fixed"
curr_30 = "v8-cal-loss-reward-30-components"
v8 = "v8_DCMAC"

env = env_2

def get_trained_agent_dir(v=curr_30, checkpt=45000):
    base_path = "wandb"
    return base_path + "\\" + v + "\\files", checkpt

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

# plot_avoided_losses_matrix(inference="all", n=10)


# def plot_avoided_losses(inference="all", bins=20, n=10,
#                         figsize=(12, 10), save_path=None):
#     """
#     Plots avoided‑loss (CAL) ratio distributions:
#       • Three subplots: histogram + KDE per policy
#       • Fourth subplot: KDE overlay of all policies
#     """

#     policy_list = ["random", "importance_based", "DCMAC"]
#     titles = ["Random",
#               "Importance Based",
#               "Deep Centralised Multi-Agent Actor-Critic (DCMAC)"]
#     if inference != "all":
#         if isinstance(inference, list):
#             policy_list = inference
#         else:
#             policy_list = [inference]

#     cmap = cm.get_cmap("magma")
#     positions = np.linspace(0.25, 0.75, len(policy_list))
#     policy_colors = {n: cmap(p) for n, p in zip(policy_list, positions)}

#     fig, axes = plt.subplots(2, 2, figsize=figsize)
#     axes = axes.flatten()
#     sns.set_style("whitegrid")

#     all_ratios = {}
#     bins_range = np.linspace(0, 1, bins + 1)          # 0.0 → 1.0

#     for i, name in enumerate(policy_list):
#         ax = axes[i]

#         # --- inference code (unchanged) -------------------------------------
#         if name == "DCMAC":
#             agent = AgentInference(name, env)
#             checkpoint_path, episode = get_trained_agent_dir()
#             agent.load_weights(checkpoint_path, episode)
#         else:
#             agent = HeuristicInference(name=name, env=env)

#         agent.get_n_rollouts(n=n)
#         losses      = agent.plotter.batch_losses
#         resilience  = agent.plotter.batch_resilience
#         # --------------------------------------------------------------------

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
#         mean_val, std_val = np.mean(ratios), np.std(ratios)

#         sns.histplot(ratios,
#                      bins=bins_range,            # fixed 0–1 bins
#                      ax=ax, stat="density", kde=True,
#                      edgecolor="black",
#                      color=policy_colors[name],
#                      alpha=0.7)

#         ax.axvline(mean_val, color=policy_colors[name],
#                    linestyle="--", linewidth=1)

#         ymin, ymax = ax.get_ylim()
#         ax.set_ylim(ymin, ymax * 1.1)
#         ax.set_xlim(0, 1)                       # <-- lock x‑axis 0–1

#         ax.text(mean_val, ymax * 1.05,
#                 f"μ={mean_val:.2f}\nσ={std_val:.2f}",
#                 ha="center", va="top", fontsize=9,
#                 bbox=dict(boxstyle="round,pad=0.3",
#                           facecolor="white",
#                           edgecolor="gray", alpha=0.7))

#         ax.set_xlabel("CAL")
#         ax.set_ylabel("Density")
#         ax.set_title(titles[i])

#     # ---- Fourth subplot: KDE overlay ----------------------------------------
#     ax = axes[3]
#     for name, ratios in all_ratios.items():
#         sns.kdeplot(ratios, label=name, ax=ax,
#                     color=policy_colors[name],
#                     shade=True, alpha=0.5)

#     ax.set_xlim(0, 1)                            # lock x‑axis 0–1
#     ax.set_xlabel("CAL")
#     ax.set_ylabel("Density")
#     ax.set_title("Policy‑wise Avoided Loss Ratio KDE")
#     ax.legend()
#     ax.grid(True, linestyle="--", alpha=0.6)
#     # -------------------------------------------------------------------------

#     fig.suptitle(
#         r"$\bf{{Cumulative\ Avoided\ Losses\ (CAL)}}$"
#         f" Distributions per Policy, over {n} Rollouts, toy-city-{env.n_agents}",
#         fontsize=14, fontfamily="serif"
#     )
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#     if save_path:
#         fig.savefig(save_path, dpi=300)

#     plt.show()




def plot_avoided_losses(env, inference="all", bins=20, n=10,
                             figsize=(14, 12), save_path=None):
    """
    Full visualization: CAL histograms, KDEs, and statistical bar comparison.
    """
    # ---- Policies ----------------------------------------------------------
    policy_list = ["random", "importance_based", "DCMAC"]
    titles = ["Random",
              "Importance Based",
              "Deep Centralised Multi-Agent Actor-Critic (DCMAC)"]
    if inference != "all":
        policy_list = inference if isinstance(inference, list) else [inference]

    cmap = cm.get_cmap("magma")
    positions = np.linspace(0.25, 0.75, len(policy_list))
    policy_colors = {n: cmap(p) for n, p in zip(policy_list, positions)}
    bins_range = np.linspace(0, 1, bins + 1)

    # ---- Run Inference + Collect CAL ---------------------------------------
    all_ratios = {}
    for name in policy_list:
        if name == "DCMAC":
            agent = AgentInference(name, env)
            checkpoint_path, episode = get_trained_agent_dir()
            agent.load_weights(checkpoint_path, episode)
        else:
            agent = HeuristicInference(name=name, env=env)

        agent.get_n_rollouts(n=n)
        losses = agent.plotter.batch_losses
        resilience = agent.plotter.batch_resilience

        if not losses or not resilience or len(losses) != len(resilience):
            print(f"Skipping '{name}' due to invalid rollout data.")
            continue

        ratios = [sum(r) / (sum(l) + sum(r))
                  for l, r in zip(losses, resilience)
                  if (sum(l) + sum(r)) > 0]

        if not ratios:
            print(f"No valid data for '{name}'")
            continue

        all_ratios[name] = ratios

    # ---- Plot Histograms + KDEs --------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    sns.set_style("whitegrid")

    for i, name in enumerate(policy_list):
        ax = axes[i][0]
        ratios = all_ratios.get(name, [])
        if not ratios:
            continue
        mean_val, std_val = np.mean(ratios), np.std(ratios)

        sns.histplot(ratios, bins=bins_range, ax=ax, stat="density", kde=True,
                     color=policy_colors[name], edgecolor="black", alpha=0.7)

        ax.axvline(mean_val, color=policy_colors[name], linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

        ax.text(mean_val, ax.get_ylim()[1] * 0.95,
                f"μ={mean_val:.2f}\nσ={std_val:.2f}",
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

        ax.set_title(f"{titles[i]} — CAL")
        ax.set_xlabel("CAL")
        ax.set_ylabel("Density")

    # ---- KDE Overlay -------------------------------------------------------
    ax_kde = axes[0][1]
    for name in policy_list:
        ratios = all_ratios.get(name, [])
        if ratios:
            sns.kdeplot(ratios, label=name, ax=ax_kde,
                        color=policy_colors[name], shade=True, alpha=0.5)
    ax_kde.set_xlim(0, 1)
    ax_kde.set_title("KDE Overlay — All Policies")
    ax_kde.legend()
    ax_kde.set_xlabel("CAL")
    ax_kde.set_ylabel("Density")
    ax_kde.grid(True, linestyle="--", alpha=0.6)

    # ---- Bar Plot + T-Test Comparisons -------------------------------------
    ax_bar = axes[1][1]

    means = [np.mean(all_ratios[name]) for name in policy_list]
    stds = [np.std(all_ratios[name]) for name in policy_list]
    x_pos = np.arange(len(policy_list))

    bars = ax_bar.bar(x_pos, means, yerr=stds, capsize=8,
                      color=[policy_colors[n] for n in policy_list], alpha=0.8)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(titles, rotation=20, ha="right")
    ax_bar.set_ylabel("Mean CAL")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_title("Policy Comparison — Mean CAL with Std Dev")
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Pairwise T-Test Annotations
    pairs = [("random", "importance_based"),
             ("importance_based", "DCMAC"),
             ("random", "DCMAC")]

    y_offset = 0.02
    for i, (a, b) in enumerate(pairs):
        x1, x2 = policy_list.index(a), policy_list.index(b)
        y = max(means[x1], means[x2]) + 0.08 + i * y_offset

        t_stat, p_val = ttest_ind(all_ratios[a], all_ratios[b], equal_var=False)
        delta = np.mean(all_ratios[b]) - np.mean(all_ratios[a])
        annotation = f"Δ={delta:.2f}, p={p_val:.3f}"

        ax_bar.plot([x1, x1, x2, x2], [y-0.005, y, y, y-0.005], lw=1.2, c='gray')
        ax_bar.text((x1 + x2) / 2, y + 0.005, annotation,
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # ---- Final Layout ------------------------------------------------------
    fig.suptitle(
        r"$\bf{Avoided\ Loss\ Ratio\ (CAL)}$"
        f" — Distribution and Comparison over {n} Rollouts per Policy",
        fontsize=15, fontfamily="serif"
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=300)

    plt.show()



plot_avoided_losses(env=env, inference="all", n=100)

def plot_single_rollout(inference="all", plot_components=True):
    if inference == "all":
        policy_list = ["random", "importance_based", "DCMAC"]
    else:
        policy_list = [inference]

    for name in policy_list:
        if name == "DCMAC":
            agent = AgentInference(name, env)
            checkpoint_path, episode = get_trained_agent_dir()
            # checkpoint_path = r"C:\Users\Anton\OneDrive\Desktop\tudelft_thesis\qres_marl\wandb\v8-cal-loss-reward-env-fixed\files"
            # episode = 40000
            agent.load_weights(checkpoint_path, episode)
        else:
            agent = HeuristicInference(name=name, env=env)

        agent.get_rollout()
        agent.plot_rollout(figsize=(20,10), plot_econ_traffic=True, plot_delay=True,
    plot_econ_relocation=True)
        if plot_components:
            agent.plot_components(figsize=(20,20))

# plot_single_rollout(inference="DCMAC", plot_components=True )


if save_env_config:
    env.reset()
    env.resilience._save_env_config("tests\\environment_testing_vdn_ps")

if viz_environment:
    env.resilience.simulation.viz_environment("test", show_road_ids=True, show_traffic_ids=True)

