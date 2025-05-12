from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Final, Union
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points, transform
from pyproj import CRS, Transformer
from shapely import affinity, wkt
import random
import pyproj
from geopy.distance import geodesic
import numpy as np
import json
import requests
from enum import Enum
from scipy.stats import lognorm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from .building_config import *
from .road_config import *
from .road_funcs import *

from pyincore import HazardService,IncoreClient
from pyincore_viz.geoutil import GeoUtil as viz
from .utils import DamageStates, sample_repair_time



import numpy as np
from scipy.stats import lognorm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class FragilityCurve:
    """
    Calculate probabilities of being in specific damage states given an intensity measure
    and state parameters using lognormal fragility curves.
    """
    def __init__(self, states_dict):
        """
        Initialize with damage states and their lognormal distribution parameters.

        Parameters
        ----------
        states_dict : dict
            Dictionary where keys are state names and values are tuples of (median, beta)
            Example: {'minor': (0.3, 0.3), 'moderate': (0.6, 0.3), 'severe': (0.9, 0.3)}
            States should be ordered from least to most severe
        """
        self.states = dict(sorted(states_dict.items(), key=lambda x: x[1][0]))
        self.state_names = list(self.states.keys())

    def exceedance_probabilities(self, im_value):
        """
        Calculate the probability of exceeding each damage state.

        Parameters
        ----------
        im_value : float or np.array
            Intensity measure value(s)

        Returns
        -------
        pd.Series
            Probabilities of exceeding each damage state
        """
        probs = {}
        for state, (median, beta) in self.states.items():
            probs[state] = lognorm.cdf(im_value, beta, scale=median)
        return pd.Series(probs)

    def state_probabilities(self, im_value):
        """
        Calculate the probability of being in each damage state
        (including no damage state).

        Parameters
        ----------
        im_value : float or np.array
            Intensity measure value(s)

        Returns
        -------
        pd.Series
            Probabilities of being in each state (including no damage)
        """
        # Get exceedance probabilities
        exceed_probs = self.exceedance_probabilities(im_value)

        # Calculate discrete state probabilities
        state_probs = {}

        # Probability of no damage
        state_probs[DamageStates.UNDAMAGED.value] = 1 - exceed_probs.iloc[0]

        # Probabilities of intermediate states
        for i in range(len(self.state_names)-1):
            current_state = self.state_names[i]
            state_probs[current_state] = exceed_probs[current_state] - exceed_probs[self.state_names[i+1]]

        # Probability of most severe state
        state_probs[self.state_names[-1]] = exceed_probs[self.state_names[-1]]

        return pd.Series(state_probs)

    def sample_damage_state(self, im_value, n_samples=1, seed=None):
        """
        Sample damage states based on their probabilities.

        Parameters
        ----------
        im_value : float
            Intensity measure value
        n_samples : int
            Number of samples to generate
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        list
            Sampled damage states
        """
        if seed is not None:
            np.random.seed(seed)

        probs = self.state_probabilities(im_value)

        # print(probs)

        states = [DamageStates.UNDAMAGED.value] + self.state_names
        samples = np.random.choice(states, size=n_samples, p=probs.values)
        # res = samples if n_samples > 1 else samples[0]
        # print('Sample are: ' + str(samples))
        # print('Sampled ds is: ' + str(res))
        return samples if n_samples > 1 else samples[0]

    def plot_fragility_analysis(self, im_range=None, figsize=(15, 8),
                            colors=None, im_label='Intensity Measure (g)'):
        """
        Create a comprehensive visualization of fragility curves and state probabilities.

        Parameters
        ----------
        im_range : tuple, optional
            Tuple of (min, max) for intensity measure range.
            If None, defaults to (0, 2*highest_median)
        figsize : tuple, optional
            Figure size in inches (width, height)
        colors : dict, optional
            Dictionary mapping states to colors
        im_label : str, optional
            Label for the intensity measure axis

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        (ax1, ax2) : tuple
            Tuple of the two axes objects
        """
        # Set up IM range if not provided
        if im_range is None:
            max_median = max(median for median, _ in self.states.values())
            im_range = (0, 2 * max_median)

        im_values = np.linspace(im_range[0], im_range[1], 100)

        # Create default colors if not provided
        if colors is None:
            default_colors = plt.cm.viridis(np.linspace(0, 1, len(self.states) + 1))
            colors = {'no_damage': default_colors[0]}
            colors.update({state: color for state, color
                        in zip(self.states.keys(), default_colors[1:])})

        # Create figure
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[1, 1])

        # Plot 1: Exceedance probabilities (fragility curves)
        ax1 = fig.add_subplot(gs[0])
        exceedance_probs = np.array([self.exceedance_probabilities(im) for im in im_values])

        for i, state in enumerate(self.states.keys()):
            ax1.plot(im_values, exceedance_probs[:, i],
                    label=state, color=colors[state])

        ax1.grid(True)
        ax1.legend()
        ax1.set_xlabel(im_label)
        ax1.set_ylabel('Probability of Exceedance')
        ax1.set_title('Fragility Curves')

        # Plot 2: State probabilities
        ax2 = fig.add_subplot(gs[1])
        state_probs = np.array([self.state_probabilities(im) for im in im_values])

        for i, state in enumerate(['no_damage'] + list(self.states.keys())):
            ax2.plot(im_values, state_probs[:, i],
                    label=state, color=colors[state])

        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel(im_label)
        ax2.set_ylabel('Probability of Being in State')
        ax2.set_title('State Probabilities')

        plt.tight_layout()
        return fig, (ax1, ax2)

@dataclass(frozen=True)
class AttenuationModels:
    ASK2014: Final[str] = 'AbrahamsonSilvaKamai2014'
    AB1995: Final[str] = 'AtkinsonBoore1995' # currently only this one works :(
    CB2014: Final[str] = 'CampbellBozorgnia2014'
    CY2014: Final[str] = 'ChiouYoungs2014'
    T1997: Final[str] = 'Toro1997'

@dataclass
class INCOREEarthquakeModelConfig:
    """
    Dataclass to represent an earthquake model configuration.
    Requires specific parameters to be provided.
    """
    name: str
    description: str
    src_latitude: float
    src_longitude: float
    magnitude: float
    depth: float
    demand_type: str
    bounds: Tuple[float, float, float, float]

    # Optional parameters with more specific defaults
    attenuation_models: Dict[str, str] = field(default_factory=lambda: {"AtkinsonBoore1995": "1.0"})
    demand_units: str = "g"
    num_points: int = 1025
    amplify_hazard: bool = True

    def __post_init__(self):
        """
        Validate input parameters after initialization.
        Raises ValueError for missing or invalid parameters.
        """
        # Check for None or empty string values in required fields
        required_fields = [
            ('name', self.name),
            ('description', self.description),
            ('demand_type', self.demand_type)
        ]

        for field_name, field_value in required_fields:
            if field_value is None or (isinstance(field_value, str) and field_value.strip() == ""):
                raise ValueError(f"{field_name.capitalize()} cannot be None or an empty string")

        # Validate numeric parameters
        numeric_fields = [
            ('src_latitude', self.src_latitude),
            ('src_longitude', self.src_longitude),
            ('magnitude', self.magnitude),
            ('depth', self.depth)
        ]

        for field_name, field_value in numeric_fields:
            if field_value is None:
                raise ValueError(f"{field_name.capitalize()} must be provided")

        # Validate bounds
        if self.bounds is None or len(self.bounds) != 4:
            raise ValueError("Bounds must be a tuple of 4 float values (minX, minY, maxX, maxY)")

        # Additional optional validations
        if not (-90 <= self.src_latitude <= 90):
            raise ValueError("Source latitude must be between -90 and 90 degrees")

        if not (-180 <= self.src_longitude <= 180):
            raise ValueError("Source longitude must be between -180 and 180 degrees")

        if self.magnitude < 0:
            raise ValueError("Magnitude cannot be negative")

        if self.depth < 0:
            raise ValueError("Depth cannot be negative")

    def to_dict(self) -> Dict:
        """
        Convert the dataclass to a dictionary matching the specified format.

        :return: Formatted dictionary for earthquake model
        """
        min_x, min_y, max_x, max_y = self.bounds

        return {
            "name": self.name,
            "description": self.description,
            "eqType": "model",
            "attenuations": self.attenuation_models,
            "eqParameters": {
                "srcLatitude": str(self.src_latitude),
                "srcLongitude": str(self.src_longitude),
                "magnitude": str(self.magnitude),
                "depth": str(self.depth)
            },
            "visualizationParameters": {
                "demandType": self.demand_type,
                "demandUnits": self.demand_units,
                "minX": str(min_x),
                "minY": str(min_y),
                "maxX": str(max_x),
                "maxY": str(max_y),
                "numPoints": str(self.num_points),
                "amplifyHazard": str(self.amplify_hazard).lower()
            }
        }

class EarthquakeAccessor:
    def __init__(self, parent_instance):
        self._parent = parent_instance
        self._validate_parent_instance()
        self.magnitude = None

    def _validate_parent_instance(self):
        """
        Validate that the parent instance has all required attributes.

        Raises:
            AttributeError: If required attributes are missing
        """
        required_attrs = [
            'verbose',
            'bounds'
        ]

        for attr in required_attrs:
            if not hasattr(self._parent, attr):
                raise AttributeError(f"Parent instance missing required attribute: {attr}")

    def __call__(self) -> gpd.GeoDataFrame:
        """
        Return the currently set study roads.

        Returns:
            gpd.GeoDataFrame: Currently stored study roads
        """
        return self._parent._earthquae_model

    def create_earthquake_model(self,
        *,  # Force keyword arguments
        name: str,
        description: str,
        src_point: Point,
        magnitude: float,
        depth: float,
        demand_type: str,
        bounds: Tuple[float, float, float, float],
        attenuation_models: Dict[AttenuationModels, int],
        demand_units: str = "g",
        num_points: int = 1025,
        amplify_hazard: bool = True
    ) -> Dict:
        """
        Convenience function to create an earthquake model configuration dictionary.
        Requires all key parameters to be explicitly provided.

        :return: Formatted earthquake model dictionary
        """

        model = INCOREEarthquakeModelConfig(
            name=name,
            description=description,
            src_latitude=src_point.y,
            src_longitude=src_point.x,
            magnitude=magnitude,
            depth=depth,
            demand_type=demand_type,
            bounds=bounds,
            attenuation_models=attenuation_models,
            demand_units=demand_units,
            num_points=num_points,
            amplify_hazard=amplify_hazard
        )
        self._log(f'Successfully created earthquake model dict: {model.to_dict()}')
        return model.to_dict()

    def generate_random_point_in_ring(
        self,
        project_centre: Point,
        min_r: float,
        max_r: float
    ) -> Point:
        """
        Generate a random point within an annular region around the project centre.

        :param project_centre: Central point in CRS 4326
        :param min_r: Minimum radius in kilometers
        :param max_r: Maximum radius in kilometers
        :return: Randomly generated point in CRS 4326
        """
        # Create a geodesic transformer
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",  # Input CRS (WGS84)
            "EPSG:3857",  # Projected CRS for calculations (Web Mercator)
            always_xy=True
        )

        # Transform the project centre to Web Mercator
        centre_x, centre_y = transformer.transform(
            project_centre.x,
            project_centre.y
        )

        # Generate random angle and distance
        angle = np.random.uniform(0, 2 * np.pi)

        # Generate random distance within the specified ring
        radius = np.sqrt(np.random.uniform(min_r**2, max_r**2))

        # Calculate new point coordinates
        new_x = centre_x + radius * 1000 * np.cos(angle)  # convert km to meters
        new_y = centre_y + radius * 1000 * np.sin(angle)

        # Transform back to WGS84
        reverse_transformer = pyproj.Transformer.from_crs(
            "EPSG:3857",  # Projected CRS
            "EPSG:4326",  # Output CRS (WGS84)
            always_xy=True
        )

        lon, lat = reverse_transformer.transform(new_x, new_y)

        return Point(lon, lat)

    def create_distance_circles(
        self,
        project_centre: Point,
        min_r: float,
        max_r: float,
        num_points: int = 100
    ) -> Tuple[LineString, LineString]:
        """
        Create two circular LineStrings representing the min and max radii from the project centre.

        :param project_centre: Central point in CRS 4326
        :param min_r: Minimum radius in kilometers
        :param max_r: Maximum radius in kilometers
        :param num_points: Number of points to use in creating the circle LineStrings
        :return: Tuple of (min_radius_linestring, max_radius_linestring)
        """
        # Create a geodesic transformer
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326",  # Input CRS (WGS84)
            "EPSG:3857",  # Projected CRS for calculations (Web Mercator)
            always_xy=True
        )

        # Transform the project centre to Web Mercator
        centre_x, centre_y = transformer.transform(
            project_centre.x,
            project_centre.y
        )

        # Create reverse transformer
        reverse_transformer = pyproj.Transformer.from_crs(
            "EPSG:3857",  # Projected CRS
            "EPSG:4326",  # Output CRS (WGS84)
            always_xy=True
        )

        # Generate points for min and max radius circles
        def create_circle_points(radius_km):
            # Generate evenly spaced angles
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

            # Calculate circle points
            circle_points_wm = []
            for angle in angles:
                # Convert radius to meters
                radius_m = radius_km * 1000

                # Calculate point coordinates
                x = centre_x + radius_m * np.cos(angle)
                y = centre_y + radius_m * np.sin(angle)

                # Transform back to WGS84
                lon, lat = reverse_transformer.transform(x, y)
                circle_points_wm.append((lon, lat))

            # Close the circle by repeating the first point
            circle_points_wm.append(circle_points_wm[0])

            return circle_points_wm

        # Create LineStrings for min and max radius circles
        min_circle_points = create_circle_points(min_r)
        max_circle_points = create_circle_points(max_r)

        min_radius_linestring = LineString(min_circle_points)
        max_radius_linestring = LineString(max_circle_points)

        return min_radius_linestring, max_radius_linestring

    def save_earthquake_json(
        self,
        eq_model_dict: Dict,
        save_path: str
    ) -> str:
        with open(save_path, 'w') as json_file:
            json.dump(eq_model_dict, json_file, indent=4)

    def read_earthquake_json(
        self,
        save_path:str
    ) -> str:
        with open(save_path, 'r') as file:
            eq_model_json = file.read()
            self._log(json.dumps(json.loads(eq_model_json), indent=4))

        return eq_model_json

    def POST_incore_earthquake(
        self,
        client: IncoreClient,
        eq_str: str
    )-> str:
        ## In-CORE Hazard Service
        hazardsrvc = HazardService(client)
        model_response = hazardsrvc.create_earthquake(eq_str)
        model_id = model_response['id']
        self._log(f'Created Earthquake with id: {model_id}')
        return model_id

    def DEL_incore_earthquake(
        self,
        bearer_token,
        eq_id
    ) -> None:
        url = "https://incore.ncsa.illinois.edu/hazard/api/earthquakes/" +  eq_id

        # IN-CORE Account Bearer Token
        headers = {
            "Authorization": "Bearer "  + bearer_token,
            "Content-Type": "application/json"
        }

        # Send a DELETE request
        response = requests.delete(url, headers=headers)

        # Check the status code of the response
        if response.status_code == 200 or response.status_code == 204:
            self._log(f"Deleted Earthquake with id: {eq_id}")
        else:
            self._log(f"Error: {response.status_code} - {response.text}")

    def POST_incore_eq_building_hazard_values(self,
        client: IncoreClient,
        eq_id : str
    ) -> None:
        hazardsrvc = HazardService(client)
        buildings_nsi = self._parent._buildings_study_gdf
        payload = []
        building_PGA = []

        for idx, row in buildings_nsi.iterrows():
            point = row['geom']

            x = point.x #long
            y = point.y #lat

            payload.append(
                {
                    "demands": ["PGA"],
                    "units": ["g"],
                    "loc": str(y) + "," + " " + str(x)
                }
            )

        eq_model_vals = hazardsrvc.post_earthquake_hazard_values(eq_id, payload)
        for i, hazard_dict in enumerate(eq_model_vals):
            pga = hazard_dict['hazardValues'][0]
            building_PGA.append(pga)

        self._parent._buildings_study_gdf[StudyBuildingSchema.PGA] = building_PGA

    def threshold_PGA_liquefaction(self, pga, M)->float:
        """
        Hazus manual table 4-10 and Figure 4-12
        """
        ## equation 4-9
        map_units = np.array([0.0,0.02,0.05,0.1,0.2,0.25]) ## table 4-10
        threshold_pgas = np.array([-1, 0.26, 0.21, 0.15, 0.12, 0.09]) ## table 4-12
        cat_probs = np.array([ ## table 4-11
            max(0.0, 9.09 * pga - 0.82),
            max(0.0, 7.67 * pga - 0.92),
            max(0.0, 6.67 * pga - 1.0),
            max(0.0, 5.57 * pga - 1.18),
            max(0.0, 4.16 * pga - 1.08),
            0.0
        ])
        k_M = 0.0027 * (M**3) - (0.0267 * M**2) - (0.2055 * M) + 2.9188 ## eq 4-10
        k_w = 0.022 * 20 + 0.93 ## eq 4-11, assumed 20ft depth to groundwater
        liquefaction_probs = np.zeros(6)

        for i in range(len(liquefaction_probs)):
            liquefaction_probs[i] = ( cat_probs[i] / (k_M * k_w) ) * map_units[i]

        ## normalise liquefaction probabilities
        total = np.sum(liquefaction_probs)
        if total > 0:
            liquefaction_probs /= total
        else:
            liquefaction_probs[-1] = 1.0

        liquefaction_cat = np.random.choice(len(liquefaction_probs), p=liquefaction_probs)
        pga_threshold = threshold_pgas[liquefaction_cat]

        return pga_threshold

    def lateral_spreading(self, pga_ratio):
        if pga_ratio < 1.0:
            return 0.0
        elif 1.0 <= pga_ratio < 2.0:
            return 12 * pga_ratio - 12
        elif 2.0 <= pga_ratio < 3.0:
            return 18 * pga_ratio - 24
        else:
            return 70 * pga_ratio - 180

    def POST_incore_eq_road_hazard_values(self,
        client: IncoreClient,
        eq_id : str,
        M: float
    ) -> None:
        hazardsrvc = HazardService(client)
        roads = self._parent._roads_study_gdf
        points = []
        road_0_3_SA = []
        road_1_0_SA = []
        road_pgd = []

        for idx, row in roads.iterrows():
            midpoint = row.geometry.interpolate(0.5, normalized=True)

            x = midpoint.x
            y = midpoint.y

            points.append(
                {
                    "demands": ["0.3 SA", "1.0 SA", "PGA"],
                    "units": ["g", "g", "g"],
                    "loc": str(y) + "," + " " + str(x)
                }
            )

        eq_model_vals = hazardsrvc.post_earthquake_hazard_values(eq_id, points)
        for i, hazard_dict in enumerate(eq_model_vals):
            sa0_3 = hazard_dict['hazardValues'][0]
            sa1_0 = hazard_dict['hazardValues'][1]
            pga = hazard_dict['hazardValues'][2]

            pga_threshold = self.threshold_PGA_liquefaction(pga=pga, M=M)
            if pga == 0.0:
                pgd = 0.0
            else:
                k_delta = (0.0086 * M**3) - (0.0914 * (M**2)) + (0.4698 * M) - 0.9835 ## eq 4-13, hazus quake manual 2024
                pga_ratio = pga / pga_threshold
                pgd = self.lateral_spreading(pga_ratio) * k_delta



            road_0_3_SA.append(sa0_3)
            road_1_0_SA.append(sa1_0)
            road_pgd.append(pgd)

        self._parent._roads_study_gdf[StudyRoadSchema.SA03SEC] = road_0_3_SA ## spectral acceleration in m/s2
        self._parent._roads_study_gdf[StudyRoadSchema.SA1SEC] = road_1_0_SA ## spectral acceleration in m/s2
        self._parent._roads_study_gdf[StudyRoadSchema.PGD] = road_pgd ## permanent ground deformation in inches

    def predict_building_DS(
        self,
        save_directory: str,
        base_name: str,
        eq_magnitude: float,
        use_random_IMs: bool = False,
        use_saved_IMs: bool = False
    ) -> None:
        """
        Predict damage states for buildings based on PGA (Peak Ground Acceleration) values.

        Args:
            use_random_pga (bool): If True, generates random PGA values between 0.1 and 0.7.
                                If False, uses existing PGA values from the dataset.
        """
        if use_random_IMs:
            self._parent._buildings_study_gdf[StudyBuildingSchema.PGA] = np.random.uniform(
                low=0.1,
                high=0.7,
                size=len(self._parent._buildings_study_gdf)
            )
        elif use_saved_IMs:
            # Get list of matching JSON files
            json_files = list(Path(save_directory).glob(f"{base_name}_{str(eq_magnitude)}.json"))
            if not json_files:
                raise FileNotFoundError(f"No files matching {base_name}_*.json found in {save_directory}")
            # Pick a random file
            selected_file = json_files[0]
            # print(f"Using IMs from: {selected_file}")

            # Load the JSON data
            with open(selected_file, "r") as f:
                im_data = json.load(f)
            self._parent._buildings_study_gdf[StudyBuildingSchema.PGA] = list(im_data.values())


        fragility = FragilityBuildingPGA_low_code()
        self._parent._buildings_study_gdf[StudyBuildingSchema.PLS0] = 0.0
        self._parent._buildings_study_gdf[StudyBuildingSchema.PLS1] = 0.0
        self._parent._buildings_study_gdf[StudyBuildingSchema.PLS2] = 0.0
        self._parent._buildings_study_gdf[StudyBuildingSchema.PLS3] = 0.0
        self._parent._buildings_study_gdf[StudyBuildingSchema.PLS4] = 0.0

        for idx, row in self._parent._buildings_study_gdf.iterrows():
            str_type = row[StudyBuildingSchema.STR_TYP2]
            pga = row[StudyBuildingSchema.PGA]

            ds_distributions = fragility.get_distribution(str_type[:4])
            fragility_curve = FragilityCurve(ds_distributions)

            probs = fragility_curve.state_probabilities(pga)
            ## probs is used for testing, make sure it sums to 1

            ## limit state probabilities
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.PLS0] = probs.values[0]
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.PLS1] = probs.values[1]
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.PLS2] = probs.values[2]
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.PLS3] = probs.values[3]
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.PLS4] = probs.values[4]

            damage_state = fragility_curve.sample_damage_state(pga)
            # print(damage_state)
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.DAMAGE_STATE] = damage_state

    def predict_road_DS(
        self,
        save_directory: str,
        base_name: str,
        eq_magnitude: float,
        use_random_IMs: bool = False,
        use_saved_IMs: bool = False
    ) -> None:
        none_bridge_mask = self._parent._roads_study_gdf[StudyRoadSchema.HAZUS_BRIDGE_CLASS] == 'None'

        if use_random_IMs:
            # Only apply random PGD to rows where bridge class is 'None'
            self._parent._roads_study_gdf.loc[none_bridge_mask, StudyRoadSchema.PGD] = np.random.uniform(
                low=5,
                high=30,
                size=none_bridge_mask.sum()  # Only generate values for matching rows
            )

        elif use_saved_IMs:
            # Get list of matching JSON files
            json_files = list(Path(save_directory).glob(f"{base_name}_{str(eq_magnitude)}.json"))
            if not json_files:
                raise FileNotFoundError(f"No files matching {base_name}_*.json found in {save_directory}")

            # Pick a random file
            selected_file = json_files[0]
            # print(f"Using IMs from: {selected_file}")

            # Load the JSON data
            with open(selected_file, "r") as f:
                im_data = json.load(f)

            # Convert indices to integers and filter based on `not_none_bridge_mask`
            matching_indices = [int(idx) for idx in im_data.keys()]
            valid_indices = self._parent._roads_study_gdf.index.intersection(matching_indices)
            valid_mask = self._parent._roads_study_gdf.index.isin(valid_indices) & none_bridge_mask

            # Ensure valid_mask and values have the same length before assignment
            if valid_mask.any():
                valid_indices_list = list(valid_indices)  # Convert to list to ensure consistent ordering

                # Convert values to Series with correct index alignment
                sa03_series = pd.Series([im_data[str(idx)][0] for idx in valid_indices_list], index=valid_indices_list)
                sa1_series = pd.Series([im_data[str(idx)][1] for idx in valid_indices_list], index=valid_indices_list)
                pgd_series = pd.Series([im_data[str(idx)][2] for idx in valid_indices_list], index=valid_indices_list)

                # Assign values using the correct index
                self._parent._roads_study_gdf.loc[valid_indices_list, StudyRoadSchema.SA03SEC] = sa03_series
                self._parent._roads_study_gdf.loc[valid_indices_list, StudyRoadSchema.SA1SEC] = sa1_series
                self._parent._roads_study_gdf.loc[valid_indices_list, StudyRoadSchema.PGD] = pgd_series

            self._log("Road IMs updated successfully.")

        fragility = FragilityRoadPGD()
        for idx, row in self._parent._roads_study_gdf.iterrows():
            if row[StudyRoadSchema.HAZUS_BRIDGE_CLASS] == 'None':
                pgd = row[StudyRoadSchema.PGD]
                hazus_road_class = row[StudyRoadSchema.HAZUS_ROAD_CLASS]


                ds_distributions = fragility.get_distribution(hazus_road_class)
                fragility_curve = FragilityCurve(ds_distributions)

                probs = fragility_curve.state_probabilities(pgd)
                ## probs is used for testing, make sure it sums to 1

                damage_state = fragility_curve.sample_damage_state(pgd)

                self._parent._roads_study_gdf.loc[idx, StudyRoadSchema.DAMAGE_STATE] = damage_state

    def predict_bridge_DS(
        self,
        save_directory: str,
        base_name: str,
        eq_magnitude: float,
        use_random_IMs: bool = False,
        use_saved_IMs: bool = False
    ) -> None:
        # Create boolean mask for rows where bridge class is NOT 'None'
        not_none_bridge_mask = self._parent._roads_study_gdf[StudyRoadSchema.HAZUS_BRIDGE_CLASS] != 'None'

        if use_random_IMs:
            self._parent._roads_study_gdf.loc[not_none_bridge_mask, StudyRoadSchema.PGD] = np.random.uniform(
                low=5,
                high=30,
                size=not_none_bridge_mask.sum()
            )
            self._parent._roads_study_gdf.loc[not_none_bridge_mask, StudyRoadSchema.SA03SEC] = np.random.uniform(
                low=3.0,
                high=14,
                size=not_none_bridge_mask.sum()
            )
            self._parent._roads_study_gdf.loc[not_none_bridge_mask, StudyRoadSchema.SA1SEC] = np.random.uniform(
                low=0.3,
                high=2.0,
                size=not_none_bridge_mask.sum()
            )

        elif use_saved_IMs:
            # Get list of matching JSON files
            json_files = list(Path(save_directory).glob(f"{base_name}_{str(eq_magnitude)}.json"))
            if not json_files:
                raise FileNotFoundError(f"No files matching {base_name}_*.json found in {save_directory}")

            # Pick a random file
            selected_file = json_files[0]
            # print(f"Using IMs from: {selected_file}")

            # Load the JSON data
            with open(selected_file, "r") as f:
                im_data = json.load(f)

            # Convert indices to integers and filter based on `not_none_bridge_mask`
            matching_indices = [int(idx) for idx in im_data.keys()]
            valid_indices = self._parent._roads_study_gdf.index.intersection(matching_indices)
            valid_mask = self._parent._roads_study_gdf.index.isin(valid_indices) & not_none_bridge_mask

            # Ensure valid_mask and values have the same length before assignment
            if valid_mask.any():
                valid_indices_list = list(valid_indices)  # Convert to list to ensure consistent ordering

                # Convert values to Series with correct index alignment
                sa03_series = pd.Series([im_data[str(idx)][0] for idx in valid_indices_list], index=valid_indices_list)
                sa1_series = pd.Series([im_data[str(idx)][1] for idx in valid_indices_list], index=valid_indices_list)
                pgd_series = pd.Series([im_data[str(idx)][2] for idx in valid_indices_list], index=valid_indices_list)

                # Assign values using the correct index
                self._parent._roads_study_gdf.loc[valid_indices_list, StudyRoadSchema.SA03SEC] = sa03_series
                self._parent._roads_study_gdf.loc[valid_indices_list, StudyRoadSchema.SA1SEC] = sa1_series
                self._parent._roads_study_gdf.loc[valid_indices_list, StudyRoadSchema.PGD] = pgd_series

            self._log("Bridge IMs updated successfully.")

        fragility = FragilityBridgeSA_PGD()
        for idx, row in self._parent._roads_study_gdf.iterrows():
            if row[StudyRoadSchema.HAZUS_BRIDGE_CLASS] != 'None':
                pgd = row[StudyRoadSchema.PGD]
                sa1_0 = row[StudyRoadSchema.SA1SEC]
                sa0_3 = row[StudyRoadSchema.SA03SEC]
                bridge_shape = row[StudyRoadSchema.BRIDGE_SHAPE]
                skew_angle = row[StudyRoadSchema.SKEW_ANGLE]
                num_spans = row[StudyRoadSchema.NUM_SPANS]
                k3d_coefficients = (row[StudyRoadSchema.K3D_A], row[StudyRoadSchema.K3D_B])

                hazus_bridge_class = row[StudyRoadSchema.HAZUS_BRIDGE_CLASS]
                ds_medians = fragility.get_medians(hazus_bridge_class)

                sa_modifier = FragilityBridgeSAModifier(
                    bridge_fragility=ds_medians,
                    sa_0_3=sa0_3,
                    sa_1_0=sa1_0,
                    bridge_shape=bridge_shape,
                    skew_angle=skew_angle,
                    num_spans=num_spans,
                    k3d_coefficients=k3d_coefficients
                )

                modified_sa_ds_distributions = sa_modifier.modify_spectral_accelerations()
                fragility_curve = FragilityCurve(modified_sa_ds_distributions)

                probs = fragility_curve.state_probabilities(pgd)

                damage_state = fragility_curve.sample_damage_state(pgd)
                self._parent._roads_study_gdf.loc[idx, StudyRoadSchema.DAMAGE_STATE] = damage_state

    def predict_building_RT(self, seed: int=None) -> None:
        bldg_repair_data = BuildingRecoveryData()
        for idx, row in self._parent._buildings_study_gdf.iterrows():
            occtype = row[StudyBuildingSchema.OCC_TYPE]
            # damage_state = row[StudyBuildingSchema.DAMAGE_STATE]

            mean_repair_times = bldg_repair_data.get_repair_time(occtype)
            damage_state_probs = np.array([
                row[StudyBuildingSchema.PLS0],
                row[StudyBuildingSchema.PLS1],
                row[StudyBuildingSchema.PLS2],
                row[StudyBuildingSchema.PLS3],
                row[StudyBuildingSchema.PLS4]
            ])
            mean_repair_time = np.sum(mean_repair_times * damage_state_probs)
            random_seed = np.random.default_rng(seed=seed).integers(low=1, high=1000, size=1)
            predicted_repair_time = sample_repair_time(
                mean_repair_time,
                random_seed=random_seed,
                size=1,
                std_dev=(0.25 * mean_repair_time)
            )
            self._parent._buildings_study_gdf.loc[idx, StudyBuildingSchema.INIT_REPAIR_TIME] = predicted_repair_time

    def predict_road_RT(self, seed: int=None) -> None:
        road_repair_distributions = RoadRepairDistributions()

        for idx, row in self._parent._roads_study_gdf.iterrows():
            if row[StudyRoadSchema.HAZUS_BRIDGE_CLASS] == 'None':
                damage_state = row[StudyRoadSchema.DAMAGE_STATE]

                if isinstance(damage_state, str):
                    # print(damage_state)
                    damage_state = DamageStates.to_int(damage_state=damage_state)
                    # print(damage_state)

                mean_repair_time, std_dev = road_repair_distributions.get_distribution(damage_state=damage_state)
                random_seed = np.random.default_rng(seed=seed).integers(low=1, high=1000, size=1)
                predicted_repair_time = sample_repair_time(
                    mean_repair_time,
                    random_seed=random_seed,
                    size=1,
                    std_dev=std_dev
                )
                self._parent._roads_study_gdf.loc[idx, StudyRoadSchema.INIT_REPAIR_TIME] = predicted_repair_time

    def predict_bridge_RT(self, seed: int=None) -> None:
        bridge_repair_distributions = BridgeRepairDistributions()

        for idx, row in self._parent._roads_study_gdf.iterrows():
            if row[StudyRoadSchema.HAZUS_BRIDGE_CLASS] != 'None':
                damage_state = row[StudyRoadSchema.DAMAGE_STATE]
                if isinstance(damage_state, str):
                    # print(damage_state)
                    damage_state = DamageStates.to_int(damage_state=damage_state)
                    # print(damage_state)

                mean_repair_time, std_dev = bridge_repair_distributions.get_distribution(damage_state=damage_state)
                random_seed = np.random.default_rng(seed=seed).integers(low=1, high=1000, size=1)
                predicted_repair_time = sample_repair_time(
                    mean_repair_time,
                    random_seed=random_seed,
                    size=1,
                    std_dev=std_dev
                )
                self._parent._roads_study_gdf.loc[idx, StudyRoadSchema.INIT_REPAIR_TIME] = predicted_repair_time

    def save_building_IM(self,
        folder_directory,
        json_file_name
    ) -> None:
        os.makedirs(folder_directory, exist_ok=True)
        study_buildings_gdf = self._parent._buildings_study_gdf
        data_dict = {
            idx: row[StudyBuildingSchema.PGA]
            for idx, row in study_buildings_gdf.iterrows()
        }

        file_path = os.path.join(folder_directory, json_file_name)
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)

        self._log(f'JSON file saved at {file_path}')

    def save_road_IM(self,
        folder_directory,
        json_file_name
    ) -> None:
        os.makedirs(folder_directory, exist_ok=True)
        study_roads_gdf = self._parent._roads_study_gdf
        data_dict = {
            idx: [
                row[StudyRoadSchema.SA03SEC],
                row[StudyRoadSchema.SA1SEC],
                row[StudyRoadSchema.PGD]
            ]
            for idx, row in study_roads_gdf.iterrows()
        }

        file_path = os.path.join(folder_directory, json_file_name)
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4)

        self._log(f'JSON file saved at {file_path}')

    def _log(
        self,
        message: str
    ) -> None:
        """
        Log messages if verbose mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._parent.verbose:
            print(message)



