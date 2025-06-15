import json
import os
import numpy as np
from pathlib import Path

# Define file paths
base_dir = Path("quake_envs_pkg/quake_envs/simulations/earthquake")
file_5_0 = base_dir / "toy_city_30_road_IMs_5.0.json"
file_5_5 = base_dir / "toy_city_30_road_IMs_5.5.json"

# Load both JSON files
with open(file_5_0, 'r') as f:
    data_5_0 = json.load(f)

with open(file_5_5, 'r') as f:
    data_5_5 = json.load(f)

# Find minimum values from 5.5 file for each position in the arrays
min_values = {}
for iteration in data_5_5:
    for road_id, values in data_5_5[iteration].items():
        for i in range(len(values)):
            if i not in min_values:
                min_values[i] = float('inf')
            if values[i] > 0:  # Only consider positive values
                min_values[i] = min(min_values[i], values[i])

# Replace 0 values in 5.0 file with minimum values from 5.5 file
for iteration in data_5_0:
    for road_id, values in data_5_0[iteration].items():
        for i in range(len(values)):
            if values[i] == 0 and i in min_values and min_values[i] != float('inf'):
                data_5_0[iteration][road_id][i] = min_values[i] * 0.5  # Use 50% of min value from 5.5

# Save the updated 5.0 file
with open(file_5_0, 'w') as f:
    json.dump(data_5_0, f, indent=2)

print("Replacement complete. Null values in 5.0 file have been replaced with 50% of minimum values from 5.5 file.")