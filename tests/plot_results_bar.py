import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 1. Data and Style Setup ---

# Create the data
data = {
    'Policy': ['Random', 'IMPB', 'VDN-PS', 'QMIX-PS', 'DCMAC',
               'Random', 'IMPB', 'DCMAC'],
    'Components': [4, 4, 4, 4, 4, 30, 30, 30],
    'CL_Mean': [275.34, 250.91, 281.41, 287.33, 249.65,
                420.42, 311.93, 359.6],
    'CL-70_Mean': [197.3, 169.2, 169.7, 152.56, 168.45,
                   250.44, 264.99, 187.12],
    'CL_Sigma': [144.18, 120.11, 132.12, 129.28, 123.75,
                 147.42, 167.76, 126.22],
    'CL-70_Sigma': [122.72, 94.98, 90.84, 85.49, 91.05,
                    113.95, 146.63, 81.5]
}
df = pd.DataFrame(data)

# Set a clean and professional style
sns.set_style("white")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Define the custom color palette from your previous request
viridis = plt.get_cmap('viridis')
magma = plt.get_cmap('magma')
policy_colors = {
    'Random': "lightgray", 'IMPB': "dimgray",
    'VDN-PS': magma(0.25), 'QMIX-PS': magma(0.5), 'DCMAC': magma(0.75)
}

# Separate data for plotting
df_4 = df[df['Components'] == 4]
df_30 = df[df['Components'] == 30]

# --- 2. Create the Plot ---

# Use subplots with shared axes for a cleaner look
fig, axes = plt.subplots(2, 2, figsize=(12,8), sharex=True)
fig.suptitle('Policy Performance Comparison Across Environments', fontsize=20, fontweight='bold')

# --- Helper function for plotting to avoid repeating code ---
def plot_bars(ax, data, y_metric, y_error, title):
    """Plots bars, error bars, and data labels on a given axis."""
    colors = [policy_colors[p] for p in data['Policy']]
    prcnt = 1.96  # 95% confidence interval multiplier
    CI = 1.96 * data[y_error] / (1000 ** 0.5)  # Calculate confidence intervals
    bars = ax.bar(data['Policy'], data[y_metric], yerr=CI,
                  color=colors, capsize=5, alpha=0.85)

    # Add data labels on top of each bar
    ax.bar_label(bars, fmt='%.1f', padding=8, fontsize=10, color='black')

    # Professional formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)
    # Set y-limit to give space for labels
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

# --- Plotting on each subplot ---
plot_bars(axes[0, 0], df_4, 'CL_Mean', 'CL_Sigma', '4 Components')
plot_bars(axes[0, 1], df_30, 'CL_Mean', 'CL_Sigma', '30 Components')
plot_bars(axes[1, 0], df_4, 'CL-70_Mean', 'CL-70_Sigma', '') # No title for bottom row
plot_bars(axes[1, 1], df_30, 'CL-70_Mean', 'CL-70_Sigma', '')

# --- 3. Final Touches ---

# Set shared axis labels to reduce redundancy
fig.text(0.06, 0.7, 'CL Mean', va='center', rotation='vertical', fontsize=14)
fig.text(0.06, 0.3, 'CL-70 Mean', va='center', rotation='vertical', fontsize=14)

# Rotate bottom x-tick labels
for ax in axes[1]:
    ax.tick_params(axis='x', rotation=45)

# Remove top and right borders (spines) for a cleaner look
sns.despine(fig)

# Create a centralized legend
legend_patches = [mpatches.Patch(color=color, label=label) for label, color in policy_colors.items()]
fig.legend(handles=legend_patches,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.95),
           ncol=len(policy_colors),
           fontsize=12,
           title="Policy",
           title_fontsize=16)

# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0.06, 0, 1, 0.92]) # rect=[left, bottom, right, top]
plt.show()