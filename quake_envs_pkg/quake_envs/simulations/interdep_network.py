from dataclasses import dataclass
from typing import List, Optional, Final, Tuple, Dict
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.affinity import rotate
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from .building_config import StudyBuildingSchema

from .building_funcs import (
    validate_NSI,
    get_NSI_bounds,
    StudyBuildingsAccessor
)
from .earthquake_funcs import EarthquakeAccessor
from .road_funcs import StudyRoadsAccessor
from .traffic_assignment import TrafficAccessor
from .interdependencies import *
import matplotlib.patches as mpatches

class InterdependentNetworkSimulation:
    """
    Simulates interdependent infrastructure networks under earthquake scenarios.

    This class processes and integrates multiple datasets, including buildings, roads, and traffic networks,
    to model interactions between different infrastructure components. The simulation framework allows
    for earthquake impact assessment and network resilience analysis.

    Features:
    - Processes geospatial datasets (GeoDataFrames) for buildings and roads.
    - Supports integration with pyIncore-compatible datasets.
    - Enables network simulation for earthquake impact analysis.
    - Provides visualization methods for infrastructure networks.

    Citations:
    - Traffic Assignment:
        - @misc{Bettini2021Traffic,
            author =       {Matteo Bettini},
            title =        {Static traffic assignment using user equilibrium and system optimum},
            howpublished = {GitHub},
            year =         {2021},
            url =          {https://github.com/MatteoBettini/Traffic-Assignment-Frank-Wolfe-2021}
            }`

    """

    ## TODO finish this ^^
    @staticmethod
    def __validate_premade(
        buildings_study_gdf,
        roads_study_gdf,
        traffic_net_df,
        traffic_dem_df,
        traffic_links_gdf,
    ) -> None:
        # Required parameters when use_premade=True
        required_params = {
            "buildings_study_gdf": buildings_study_gdf,
            "roads_study_gdf": roads_study_gdf,
            "traffic_net_df": traffic_net_df,
            "traffic_dem_df": traffic_dem_df,
            "traffic_links_gdf": traffic_links_gdf,
        }

        # Find missing parameters
        missing_params = [
            param for param, value in required_params.items() if value is None
        ]

        # Raise error if any required parameter is missing
        if missing_params:
            raise ValueError(
                f"Missing required parameters when use_premade=True: {', '.join(missing_params)}. "
                "Provide these gdfs/dfs."
                "\nInformation on each input: "
                "\n-- buildings_study_gdf: "
                "\n---- see base gdf schema: (req. log-in)https://tools.in-core.org/semantics/api/types/ergo:buildingInventoryVer4"
                "\n---- see extended gdf schema: -> building_config.py:StudyBuildingSchema "
                "\n "
                "\n-- roads_study_gdf: "
                "\n---- see base gdf schema: (req. login)https://tools.in-core.org/semantics/api/types/incore:roads"
                "\n---- see extended gdf schema: -> road_config.py:StudyRoadSchema"
                "\n-- traffic_net_df: "
                "\n---- see df schema: https://github.com/matteobettini/Traffic-Assignment-Frank-Wolfe-2021"
                # TODO finish this ^^
            )

    def __init__(
        self,
        ## Init choice 1: --------
        # Used for premade study gdfs made either manually or with Choice 2
        use_premade: bool = True,
        buildings_study_gdf: gpd.GeoDataFrame = None,
        roads_study_gdf: gpd.GeoDataFrame = None,
        traffic_net_df: pd.DataFrame = None,
        traffic_dem_df: pd.DataFrame = None,
        traffic_links_gdf: gpd.GeoDataFrame = None,
        # Init choice 2 : ----------
        # Used with an nsi gdf from https://nsi.sec.usace.army.mil/downloads/
        # ATTENTION: it is recommended to pick a sub-area of the state-downloads
        # in the link provided, simulation can be very slow for large environments
        building_nsi_gdf: gpd.GeoDataFrame = None,
        roads_traffic_net_tntp_f: str = None,
        roads_traffic_demand_tntp_f: str=None,
        avg_dwelling_size: float = 2.5,
        osm_search_radius: int = 50,
        osm_call_limit : int = 3,
        verbose : bool = True
    ) -> None:
        self.verbose = verbose
        self._avg_dwell_size = avg_dwelling_size
        self.bounds = None
        self.center = None

        buildings_study_gdf = buildings_study_gdf.copy()
        roads_study_gdf= roads_study_gdf.copy()
        traffic_net_df= traffic_net_df.copy()
        traffic_dem_df= traffic_dem_df.copy()
        traffic_links_gdf = traffic_links_gdf.copy()

        if use_premade:
            self.__validate_premade(
                buildings_study_gdf=buildings_study_gdf,
                roads_study_gdf=roads_study_gdf,
                traffic_net_df=traffic_net_df,
                traffic_dem_df=traffic_dem_df,
                traffic_links_gdf=traffic_links_gdf
            )
            self.buildings_study.set_local(
                buildings_study_gdf=buildings_study_gdf
            )
            self.__log(f"Set {len(buildings_study_gdf)} study buildings with bounds: {self.bounds}")
            self.roads_study.set_local(
                roads_study_gdf=roads_study_gdf
            )
            self.__log(f"Set local road network with {len(roads_study_gdf)} links.")

            self._traffic_links_gdf = traffic_links_gdf
            self.interdep_road_to_building()
            self.traffic.map_traffic_links()
            self.init_traffic_net_df = traffic_net_df.copy()
            self.init_traffic_dem_def = traffic_dem_df.copy()
            self.curr_traffic_net_df = traffic_net_df.copy()
            self.curr_traffic_dem_df = traffic_dem_df.copy()
            self.traffic_calc_net = self.traffic.make_traffic_net(
                net_df=self.init_traffic_net_df,
                demand_df=self.init_traffic_dem_def
            )

        else:
            # project utility variables
            self.osm_call_limit = osm_call_limit

            # study building inputs
            self._osm_search_radius = osm_search_radius
            self._buildings_nsi_gdf = building_nsi_gdf
            self._buildings_incore_gdf = None
            self._buildings_study_gdf = None

            # study traffic network inputs
            self._traffic_links_gdf = traffic_links_gdf
            self._roads_traffic_net_tntp_f = roads_traffic_net_tntp_f
            self._roads_traffic_demand_tntp_f = roads_traffic_demand_tntp_f
            self._roads_osm_graph = None
            self._roads_incore_gdf = None
            self._roads_study_gdf = None

            # road-building interdependency matrix [road,building]=1 if building accesses road
            self._dep_road_to_building = None

    @property
    def buildings_nsi(self) -> gpd.GeoDataFrame:
        return self._buildings_nsi_gdf

    @buildings_nsi.setter
    def building_nsi(
        self,
        nsi_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        validates column names as per NSISchema and removes dup building centres, keeping largest area centre

        Parameters:
            nsi_gdf (gpd.GeoDataFrame): A GeoDataFrame containing building
            centres with associated attributes such as square footage.

        Raises:
            ValueError: If the GeoDataFrame does not contain the required schema
            (NSI Structures Inventory).

        Returns:
            None
        """
        validated_nsi_gdf = validate_NSI(nsi_gdf)
        self._buildings_nsi_gdf = validated_nsi_gdf
        self.bounds, self.center = get_NSI_bounds(self._buildings_nsi_gdf) ## get bounds and centre

    @property
    def buildings_study(self):
        """
        Provides access to study buildings with methods for retrieval and setting.

        Usage:
        - Get OSM buildings: sim.buildings_study.get_osm()
        - Set local buildings: sim.buildings_study.set_local(local_gdf)
        - Clear buildings: sim.buildings_study.clear()
        - Access current buildings: sim.buildings_study
        """
        return StudyBuildingsAccessor(self)

    @property
    def num_buildings(self):
        return len(self._buildings_study_gdf)

    @property
    def roads_study(self):
        """
        Provides access to study roads with methods for retrieval and setting.

        Usage:
        - Get OSM roads: sim.STUDY_ROADS.set_osm()
        - Set local buildings: sim.STUDY_ROADS.set_local(local_gdf)
        - Clear buildings: sim.STUDY_BUILDINGS.clear()
        - Access current buildings: sim.STUDY_BUILDINGS
        """
        return StudyRoadsAccessor(self)

    @property
    def num_roads(self):
        return len(self._roads_study_gdf)

    @property
    def roads_graph(self):
        return self._roads_osm_graph

    @property
    def traffic(self):
        return TrafficAccessor(self)

    @property
    def earthquake(self):
        return EarthquakeAccessor(self)

    def interdep_road_to_building(self):
        roads_gdf = self._roads_study_gdf
        buildings_gdf = self._buildings_study_gdf

        self._buildings_study_gdf = road_to_building(
            roads_gdf=roads_gdf,
            buildings_gdf=buildings_gdf
        )

    @staticmethod
    def ds_to_int(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
        """Converts damage states to integers only if the column is not already of type int."""

        if col not in gdf.columns:
            gdf[col] = 0  # Initialize column if it does not exist

        # Only convert if the column is not already of integer type
        if not pd.api.types.is_integer_dtype(gdf[col]):
            gdf[col] = [DamageStates.to_int(damage_state=ds) for ds in gdf[col]]

        return gdf

    @staticmethod
    def ds_to_str(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
        """Converts integers back to damage states, but only if the column is not already of type str."""
        if col not in gdf.columns:
            gdf[col] = 0
        gdf[col] = [DamageStates.to_str(damage_state=ds) for ds in gdf[col]]
        return gdf

    def viz_environment(
        self,
        plot_name: str,
        figsize: Tuple[int, int] = None,
        show_traffic_ids: bool = False,
        show_road_ids: bool = False,
    ) -> None:
        roads = self._roads_study_gdf
        buildings = self._buildings_study_gdf
        traffic_roads = self._traffic_links_gdf

        # Set professional style
        sns.set_theme(style="white", palette="muted")

        # Create figure and axes with an optional figsize
        if not figsize:
            fig, ax = plt.subplots(figsize=(14, 14))
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # Color buildings based on 'occ_type' column
        def get_building_color(occ_type):
            if occ_type.startswith("COM") and occ_type != 'COM6':
                return 'purple', 'Commercial'
            elif occ_type.startswith("RES"):
                return 'blue', 'Residential'
            elif occ_type in ['COM6', 'GOV2']:
                return 'orange', 'Essential'
            else:
                return 'gray', 'Other'

        # Plot buildings with conditional colors based on 'occ_type'
        for idx, row in buildings.iterrows():
            color, label = get_building_color(row[StudyBuildingSchema.OCC_TYPE])
            buildings.iloc[[idx]].plot(ax=ax, color=color, edgecolor='black', alpha=0.5, label=label)

        # Plot building debris rectangles
        for idx, row in buildings.iterrows():
            geom = row[StudyBuildingSchema.DEBRIS_GEOM]
            if geom:
                gpd.GeoDataFrame(geometry=[geom], crs=buildings.crs).plot(
                    ax=ax,
                    edgecolor='black',    # solid black outline
                    facecolor='none',     # fully transparent fill
                    linewidth=0.5,
                    alpha=1.0             # fully opaque edge
                )

        # Plot traffic roads with dashed red lines and more prominent styling
        traffic_roads.plot(ax=ax, color='darkred', linewidth=1, linestyle='--', label='Traffic Routes')

        # Plot roads with conditional line thickness based on road classification
        for idx, row in roads.iterrows():
            # Check if road is labeled as HRD1 or a bridge
            road_label = row.get(StudyRoadSchema.HAZUS_ROAD_CLASS) or row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS)
            if road_label == ["HRD1"] or road_label[:3] == "HWB":
                line_width = 10
            else:
                line_width = 4
            # Plot each road with the appropriate line width
            roads.iloc[[idx]].plot(ax=ax, color='black', linewidth=line_width, alpha = 0.4)
        def bounds(gdf_list):
            x_min, y_min, x_max, y_max = None, None, None, None
            for gdf in gdf_list:
                if StudyBuildingSchema.DEBRIS_GEOM in gdf.columns:
                    _x_min, _y_min, _x_max, _y_max = gpd.GeoSeries(gdf[StudyBuildingSchema.DEBRIS_GEOM]).total_bounds

                else:
                    _x_min, _y_min, _x_max, _y_max = gdf.total_bounds
                if x_min is None:
                    x_min, y_min, x_max, y_max = _x_min, _y_min, _x_max, _y_max
                else:
                    x_min, y_min, x_max, y_max = min(x_min, _x_min), min(y_min, _y_min), max(x_max, _x_max), max(y_max, _y_max)
            return x_min, y_min, x_max, y_max


        x_min, y_min, x_max, y_max = bounds([buildings, roads, traffic_roads])
        padding = 0.3
        x_range = x_max - x_min
        y_range = y_max - y_min
        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        # Label buildings with their structure type ('str_typ2')
        for idx, row in buildings.iterrows():
            x, y = row.geometry.centroid.x, row.geometry.centroid.y
            ax.text(
                x, y, str(row[StudyBuildingSchema.STR_TYP2]),
                fontsize=10, color='black', ha='center', va='center', fontweight='bold', rotation=45
            )

        # Optionally show traffic IDs at midpoints
        if show_traffic_ids:
            for idx, row in traffic_roads.iterrows():
                midpoint = row.geometry.interpolate(0.5, normalized=True)
                ax.text(
                    midpoint.x, midpoint.y, str(idx),
                    fontsize=12, color='darkred',
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkred', boxstyle='round,pad=0.3')
                )

        # Optionally show road IDs with their classification at midpoints
        if show_road_ids:
            for idx, row in roads.iterrows():
                midpoint = row.geometry.interpolate(0.5, normalized=True)
                label = row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS) if row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS) != "None" else row.get(StudyRoadSchema.HAZUS_ROAD_CLASS)
                if label:
                    ax.text(
                        midpoint.x, midpoint.y, str(label),
                        fontsize=10, color='darkslateblue', ha='center', va='center', fontweight='bold', rotation=45
                    )

        # Set a more descriptive and professional title
        ax.set_title(plot_name, fontsize=18, fontweight='bold')

        # Create a custom legend to describe building types
        purple_patch = mpatches.Patch(color='purple', label='Commercial')
        blue_patch = mpatches.Patch(color='blue', label='Residential')
        orange_patch = mpatches.Patch(color='orange', label='Essential')
        gray_patch = mpatches.Patch(color='gray', label='Other')

        ax.legend(handles=[purple_patch, blue_patch, orange_patch, gray_patch,
                        mpatches.Patch(color='black', label='Roads', linewidth=0.25),
                        mpatches.Patch(color='darkred', label='Traffic Routes', linewidth=0.25, linestyle='--')],
                loc='lower right', fontsize=12, frameon=True, edgecolor='black', ncol=2)

        # Improve the grid appearance (lighter and dotted)
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray')

        # Tight layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.show()

    def __log(self,msg):
        if self.verbose:
            print(msg)










