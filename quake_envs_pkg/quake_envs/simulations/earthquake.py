import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

def plot_pga_at_target_rps_no_faults(hazard_gdf, polygon_gdf, target_rps, center=None):
    """
    Plot PGA interpolated at given Return Periods (RPs) over polygons without outlines,
    with a shared color scale, km axes centered on plot, and optional center point label.

    Parameters:
    - hazard_gdf: GeoDataFrame with Points and columns representing IM values (like PGA in g) as strings,
      where cell values are probabilities of exceedance (PE).
    - polygon_gdf: GeoDataFrame with polygons to be colored by interpolated PGA.
    - target_rps: dict, keys = column names for plots, values = target return periods (years).
    - center: shapely.geometry.Point or None, coordinate to center axes on (in the polygons CRS).

    Returns:
    - joined GeoDataFrame (polygons with interpolated PGA columns added).
    """

    # Use projected CRS for USA (Albers Equal Area)
    projected_crs = "EPSG:5070"

    # Reproject inputs if needed
    if hazard_gdf.crs.to_epsg() == 4326:
        hazard_gdf = hazard_gdf.to_crs(projected_crs)
    if polygon_gdf.crs.to_epsg() == 4326:
        polygon_gdf = polygon_gdf.to_crs(projected_crs)

    def interpolate_gm(row, target_rps):
        exclude_cols = ['lon', 'lat', 'geometry', 'index_right', 'point_geom']
        im_cols = [c for c in row.index if c not in exclude_cols and str(c).replace('.', '', 1).isdigit()]
        if len(im_cols) == 0:
            return {label: np.nan for label in target_rps.keys()}

        im_levels = np.array([float(c) for c in im_cols])
        pe_values = pd.to_numeric(row[im_cols], errors='coerce').values
        mask = ~np.isnan(pe_values)
        im_levels = im_levels[mask]
        pe_values = pe_values[mask]

        if len(pe_values) < 2:
            return {label: np.nan for label in target_rps.keys()}

        sort_idx = np.argsort(pe_values)
        pe_values_sorted = pe_values[sort_idx]
        im_levels_sorted = im_levels[sort_idx]

        results = {}
        for label, rp in target_rps.items():
            target_pe = 1.0 / rp
            gm_at_pe = np.interp(target_pe, pe_values_sorted, im_levels_sorted)
            results[label] = gm_at_pe
        return results

    # Interpolate hazard GM values at target RPs
    interpolated_df = hazard_gdf.apply(lambda r: interpolate_gm(r, target_rps), axis=1, result_type='expand')
    hazard_gdf = hazard_gdf.join(interpolated_df)

    # Ensure same CRS before spatial join
    if hazard_gdf.crs != polygon_gdf.crs:
        hazard_gdf = hazard_gdf.to_crs(polygon_gdf.crs)

    # Use polygon centroids for join
    polygon_gdf = polygon_gdf.copy()
    polygon_gdf['centroid'] = polygon_gdf.geometry.centroid

    joined = gpd.sjoin_nearest(
        polygon_gdf.set_geometry('centroid'),
        hazard_gdf.set_geometry('geometry'),
        how='left',
        distance_col='dist'
    )

    # Determine center of plot axes in projected coords
    if center is not None:
        center_x, center_y = center.x, center.y
    else:
        bounds = joined.geometry.total_bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

    n = len(target_rps)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # Find global min/max for color scale
    all_vals = []
    for label in target_rps.keys():
        if label in joined.columns:
            vals = joined[label].dropna().values
            if len(vals) > 0:
                all_vals.append(vals)
    if all_vals:
        all_vals = np.concatenate(all_vals)
        vmin, vmax = all_vals.min(), all_vals.max()
    else:
        vmin, vmax = 0, 1

    cmap = 'viridis'

    for ax, (label, rp) in zip(axes, target_rps.items()):
        if label not in joined.columns or joined[label].isnull().all():
            ax.set_title(f"{label} (RP={rp} yrs)\nNo data")
            ax.axis('off')
            continue

        joined.plot(
            column=label,
            ax=ax,
            cmap=cmap,
            linewidth=0,
            edgecolor=None,
            vmin=vmin,
            vmax=vmax,
            legend=(ax == axes[-1]),
            legend_kwds={'label': 'PGA (g)', 'shrink': 0.7}
        )

        # Draw polygon boundaries
        joined.boundary.plot(ax=ax, color='black', linewidth=0.5)

        ax.set_title(f"{label} (RP = {rp} years)")
        ax.set_xlabel("X (km from center)")
        ax.set_ylabel("Y (km from center)")

        ax.set_aspect('equal', adjustable='datalim')

        # Fix axis limits if flipped
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if xmin > xmax:
            ax.set_xlim(xmax, xmin)
        if ymin > ymax:
            ax.set_ylim(ymax, ymin)

        # Convert ticks to km from center
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{(x - center_x) / 1000:.0f}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{(y - center_y) / 1000:.0f}"))

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), ha='right')

        # Mark the center point if given
        if center is not None:
            ax.plot(center_x, center_y, marker='x', color='red', markersize=10, label='Study Site Center')
            ax.legend()

    plt.show()
    return joined


# from shapely.geometry import Point

# # Your hazard and polygon GeoDataFrames here:
# # pga_map = ...
# # vs30_map = ...

# target_rps = {
#     'GM_475yr': 475,
#     'GM_975yr': 975,
#     'GM_2475yr': 2475,
# }


# # If your polygons are originally in 4326, project center point to projected CRS
# import pyproj
# from shapely.ops import transform

# project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform
# center_point_proj = transform(project, center)

# joined_gdf = plot_pga_at_target_rps_no_faults(pga_map, vs30_map, target_rps, center=center_point_proj)





def plot_hazard_curve(row, im_type: str):
  # Extract ground motion levels and probabilities
  gm_cols = [col for col in row.index if isinstance(col, float) or col.replace('.', '', 1).isdigit()]
  gm_levels = np.array([float(gm) for gm in gm_cols])
  probs = np.array([row[gm] for gm in gm_cols])

  # Sort by ground motion level
  sorted_indices = np.argsort(gm_levels)
  gm_levels = gm_levels[sorted_indices]
  probs = probs[sorted_indices]

  # Plot
  plt.figure(figsize=(8, 5))
  plt.plot(gm_levels, probs, marker='o')
  # plt.gca().set_xscale('log')
  plt.gca().set_yscale('log')
  plt.xlabel(f'Ground Motion,  {im_type}')
  plt.ylabel('Annual Probability of Exceedance')
  plt.title('Hazard Curve at Location (%.2f, %.2f)' % (row['lon'], row['lat']))
  plt.grid(True, which="both", ls="--", linewidth=0.5)
  plt.tight_layout()
  plt.show()