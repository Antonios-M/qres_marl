import numpy as np
import pandas as pd
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt
import quake_envs

def analyze_earthquake_data(env_name="quake-res-30-v1", use_return_periods=True,
                          return_periods=None, num_samples=2):
    """
    Analyze earthquake simulation data and create a professional plot.

    Parameters:
    -----------
    env_name : str
        Name of the gymnasium environment
    use_return_periods : bool
        If True, use return periods. If False, use random magnitudes
    return_periods : list
        List of return periods to sample from. If None, uses default list
    num_samples : int
        Number of simulations to run by randomly sampling from return_periods or using random magnitudes
    """

    env = gym.make(env_name).unwrapped

    # Set default return periods if not provided
    if return_periods is None:
        return_periods = [475, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]

    # --- Data Generation from Simulation ---
    x_values = []
    robustness_values = []

    if use_return_periods:
        print(f"Running {num_samples} simulations by randomly sampling from return periods...")
        # Randomly sample return periods for the specified number of simulations
        np.random.seed(42)  # For reproducibility
        sampled_return_periods = np.random.choice(return_periods, size=num_samples, replace=True)

        for i, rp in enumerate(sampled_return_periods):
            obs, info = env.reset(rp)
            # Use return period for x-axis
            x_values.append(env.return_period)
            # Append the community robustness for the y-axis
            robustness_values.append(info["q"]["community_robustness"])
            if info["q"]["community_robustness"] > 0.99:
                print(f"Simulation {i+1} (RP={rp}): Community robustness is {info['q']['community_robustness']}")

        x_label = "Return Period (years)"
        title_var = "Return Period"
        data_label = "Return Periods"

    else:
        print(f"Running {num_samples} simulations with random magnitudes...")
        for i in range(num_samples):
            obs, info = env.reset()
            # Use earthquake magnitude for x-axis
            x_values.append(env.resilience.eq_magnitude)
            # Append the community robustness for the y-axis
            robustness_values.append(info["q"]["community_robustness"])
            if info["q"]["community_robustness"] > 0.99:
                print(f"Simulation {i+1}: Community robustness is {info['q']['community_robustness']}")

        x_label = "Earthquake Magnitude"
        title_var = "Earthquake Magnitude"
        data_label = "Magnitudes"

    print("Simulations complete.")

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'x_values': x_values,
        'robustness': robustness_values
    })

    # Check the actual range of robustness values
    print(f"Robustness range: {min(robustness_values):.3f} to {max(robustness_values):.3f}")

    # --- Data Aggregation for Error Bars ---
    # Group by x-values and calculate the mean and standard deviation of robustness
    agg_data = df.groupby('x_values')['robustness'].agg(['mean', 'std']).reset_index()

    # Handle cases where std is NaN (single data point per group)
    agg_data['std'] = agg_data['std'].fillna(0)

    # --- Non-Linear Regression (2nd Degree Polynomial) ---
    # Fit the curve to the MEAN of the data for a stable fit
    coeffs = np.polyfit(agg_data['x_values'], agg_data['mean'], 2)
    polynomial_fit = np.poly1d(coeffs)

    # Generate a smooth x-axis range for plotting the curve
    x_fit = np.linspace(agg_data['x_values'].min(), agg_data['x_values'].max(), 200)
    y_fit = polynomial_fit(x_fit)

    # --- Create envelope based on standard deviation ---
    # Interpolate standard deviations for smooth envelope
    std_interp = np.interp(x_fit, agg_data['x_values'], agg_data['std'])
    y_upper = y_fit + std_interp
    y_lower = y_fit - std_interp

    # --- Plotting with Seaborn and Matplotlib ---
    # Set a professional style for an academic paper
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Plot individual datapoints as semi-transparent circles
    plt.scatter(df['x_values'], df['robustness'],
               alpha=0.1,
               s=20,
               color='midnightblue',
               edgecolors='none',
               label='Data Points',
               zorder=1)

    # Plot the envelope (confidence band)
    plt.fill_between(x_fit, y_lower, y_upper,
                     alpha=0.2,
                     color='darkred',
                     label='±1 Standard Deviation',
                     zorder=5)

    # Plot the non-linear regression curve
    plt.plot(x_fit, y_fit,
             color='darkred',
             linewidth=2.5,
             label='Polynomial Fit (2nd Degree)',
             zorder=10)

    # Plot the mean values with error bars (NO OFFSET to avoid going above 1)
    plt.errorbar(
        x=agg_data['x_values'],
        y=agg_data['mean'],  # No offset - plot at actual mean values
        yerr=agg_data['std'],
        fmt='s',
        color='darkcyan',
        markersize=5,
        markerfacecolor='black',
        markeredgewidth=1.5,
        markeredgecolor='black',
        elinewidth=1,
        capsize=10,
        ecolor='dimgray',
        label='Mean Robustness ± Std Dev',
        linestyle='None',
        zorder=50
    )

    # --- Final Plot Customization ---
    plt.title(f"Community Robustness vs. {title_var}", fontsize=16, weight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Community Robustness ($Q_{com}(t_m)$)", fontsize=12)

    # Set appropriate x-axis limits
    x_range = max(x_values) - min(x_values)
    x_padding = x_range * 0.05
    plt.xlim(min(x_values) - x_padding, max(x_values) + x_padding)

    # Set y-axis limits to ensure all data is visible and not above 1
    y_max = max(1.05, max(robustness_values) + 0.05)
    plt.ylim(0, y_max)

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, agg_data

# Example usage:
# For return periods with random sampling (500 samples from predefined return periods):
df_rp, agg_rp = analyze_earthquake_data(use_return_periods=False, num_samples=1000)

# For magnitudes (random magnitudes):
# df_mag, agg_mag = analyze_earthquake_data(use_return_periods=False, num_samples=100)