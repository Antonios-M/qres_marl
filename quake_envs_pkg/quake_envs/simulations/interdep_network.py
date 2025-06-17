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
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon, MultiPolygon

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

    # def viz_environment(
    #     self,
    #     plot_name: str,
    #     figsize: Tuple[int, int] = None,
    #     show_traffic_ids: bool = False,
    #     show_road_ids: bool = False,
    # ) -> None:
    #     roads = self._roads_study_gdf
    #     buildings = self._buildings_study_gdf
    #     traffic_roads = self._traffic_links_gdf

    #     # Set professional style
    #     sns.set_theme(style="white", palette="muted")

    #     # Create figure and axes with an optional figsize
    #     if not figsize:
    #         fig, ax = plt.subplots(figsize=(14, 14))
    #     else:
    #         fig, ax = plt.subplots(figsize=figsize)

    #     # Color buildings based on 'occ_type' column
    #     def get_building_color(occ_type):
    #         if occ_type.startswith("COM") and occ_type != 'COM6':
    #             return 'purple', 'Commercial'
    #         elif occ_type.startswith("RES"):
    #             return 'blue', 'Residential'
    #         elif occ_type in ['COM6', 'GOV2']:
    #             return 'orange', 'Essential'
    #         else:
    #             return 'gray', 'Other'

    #     # Plot buildings with conditional colors based on 'occ_type'
    #     for idx, row in buildings.iterrows():
    #         color, label = get_building_color(row[StudyBuildingSchema.OCC_TYPE])
    #         buildings.iloc[[idx]].plot(ax=ax, color=color, edgecolor='black', alpha=0.5, label=label)

    #     # Plot building debris rectangles with transparent fill and dotted black outline
    #     for idx, row in buildings.iterrows():
    #         geom = row[StudyBuildingSchema.DEBRIS_GEOM]
    #         if geom:
    #             if isinstance(geom, (Polygon, MultiPolygon)):
    #                 geoms = [geom] if isinstance(geom, Polygon) else geom.geoms
    #                 for poly in geoms:
    #                     patch = MplPolygon(
    #                         list(poly.exterior.coords),
    #                         facecolor='orange',          # fully transparent fill
    #                         alpha=0.2,
    #                         edgecolor='black',         # solid black outline
    #                         linewidth=0.5,
    #                         linestyle=':',
    #                         zorder=10             # dotted line
    #                     )
    #                     ax.add_patch(patch)


    #     # Plot traffic roads with dashed red lines and more prominent styling
    #     traffic_roads.plot(ax=ax, color='darkred', linewidth=1, linestyle='--', label='Traffic Routes')

    #     # Plot roads with conditional line thickness based on road classification
    #     for idx, row in roads.iterrows():
    #         # Check if road is labeled as HRD1 or a bridge
    #         road_label = row.get(StudyRoadSchema.HAZUS_ROAD_CLASS) or row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS)
    #         if road_label == ["HRD1"] or road_label[:3] == "HWB":
    #             line_width = 10
    #         else:
    #             line_width = 4
    #         # Plot each road with the appropriate line width
    #         roads.iloc[[idx]].plot(ax=ax, color='black', linewidth=line_width, alpha = 0.4)

    #     def bounds(gdf_list):
    #         x_min, y_min, x_max, y_max = None, None, None, None
    #         for gdf in gdf_list:
    #             if StudyBuildingSchema.DEBRIS_GEOM in gdf.columns:
    #                 _x_min, _y_min, _x_max, _y_max = gpd.GeoSeries(gdf[StudyBuildingSchema.DEBRIS_GEOM]).total_bounds

    #             else:
    #                 _x_min, _y_min, _x_max, _y_max = gdf.total_bounds
    #             if x_min is None:
    #                 x_min, y_min, x_max, y_max = _x_min, _y_min, _x_max, _y_max
    #             else:
    #                 x_min, y_min, x_max, y_max = min(x_min, _x_min), min(y_min, _y_min), max(x_max, _x_max), max(y_max, _y_max)
    #         return x_min, y_min, x_max, y_max


    #     x_min, y_min, x_max, y_max = bounds([buildings, roads, traffic_roads])
    #     padding = 0.2
    #     x_range = x_max - x_min
    #     y_range = y_max - y_min
    #     ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    #     ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    #     # Label buildings with their structure type ('str_typ2')
    #     for idx, row in buildings.iterrows():
    #         x, y = row.geometry.centroid.x, row.geometry.centroid.y
    #         ax.text(
    #             x, y, str(row[StudyBuildingSchema.STR_TYP2]),
    #             fontsize=10, color='black', ha='center', va='center', fontweight='bold', rotation=45
    #         )

    #     # Optionally show traffic IDs at midpoints
    #     if show_traffic_ids:
    #         for idx, row in traffic_roads.iterrows():
    #             midpoint = row.geometry.interpolate(0.5, normalized=True)
    #             ax.text(
    #                 midpoint.x, midpoint.y, str(idx),
    #                 fontsize=12, color='darkred',
    #                 ha='center', va='center', fontweight='bold',
    #                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkred', boxstyle='round,pad=0.3')
    #             )

    #     # Optionally show road IDs with their classification at midpoints
    #     if show_road_ids:
    #         for idx, row in roads.iterrows():
    #             line = row.geometry
    #             midpoint = line.interpolate(0.5, normalized=True)

    #             # Get label
    #             label = row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS)
    #             if label == "None":
    #                 label = row.get(StudyRoadSchema.HAZUS_ROAD_CLASS)

    #             label = label + "_" + str(idx)
    #             if label:
    #                 # Compute angle at midpoint
    #                 try:
    #                     # Take two points close to the midpoint for angle calculation
    #                     p1 = line.interpolate(0.49, normalized=True)
    #                     p2 = line.interpolate(0.51, normalized=True)
    #                     dx = p2.x - p1.x
    #                     dy = p2.y - p1.y
    #                     angle = np.degrees(np.arctan2(dy, dx))
    #                 except:
    #                     angle = 0

    #                 ax.text(
    #                     midpoint.x, midpoint.y, str(label),
    #                     fontsize=10, color='darkslateblue', ha='center', va='center',
    #                     fontweight='bold', rotation=angle,
    #                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none')
    #                 )

    #     # Set a more descriptive and professional title
    #     ax.set_title(plot_name, fontsize=18, fontweight='bold')

    #     # Create a custom legend to describe building types
    #     purple_patch = mpatches.Patch(color='purple', label='Commercial')
    #     blue_patch = mpatches.Patch(color='blue', label='Residential')
    #     orange_patch = mpatches.Patch(color='orange', label='Essential')
    #     gray_patch = mpatches.Patch(color='gray', label='Other')

    #     ax.legend(handles=[purple_patch, blue_patch, orange_patch, gray_patch,
    #                     mpatches.Patch(color='black', label='Roads', linewidth=0.25),
    #                     mpatches.Patch(color='darkred', label='Traffic Routes', linewidth=0.25, linestyle='--')],
    #             loc='lower right', fontsize=12, frameon=True, edgecolor='black', ncol=2)

    #     # Improve the grid appearance (lighter and dotted)
    #     ax.grid(True, linestyle=':', linewidth=0.5, color='gray')

    #     # Tight layout for better spacing
    #     plt.tight_layout()

    #     # Show the plot
    #     plt.show()

    def __log(self,msg):
        if self.verbose:
            print(msg)



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

        # Create figure and axes
        if not figsize:
            fig, ax = plt.subplots(figsize=(14, 14))
        else:
            fig, ax = plt.subplots(figsize=figsize)

        # --- (Keep Building plotting, Debris plotting, Traffic plotting, Road plotting logic as before) ---
        # Color buildings based on 'occ_type' column
        def get_building_color(occ_type):
            occ_type_str = str(occ_type)
            if occ_type_str.startswith("COM") and occ_type_str != 'COM6':
                return 'purple', 'Commercial'
            elif occ_type_str.startswith("RES"):
                return 'blue', 'Residential'
            elif occ_type_str in ESSENTIAL_FACILITY_OCC_TYPES:
                return 'orange', 'Essential'
            else:
                return 'gray', 'Other'

        building_handles = {}
        for idx, row in buildings.iterrows():
            color, label = get_building_color(row[StudyBuildingSchema.OCC_TYPE])
            buildings.iloc[[idx]].plot(ax=ax, color=color, edgecolor='black', alpha=0.5)
            if label not in building_handles:
                 building_handles[label] = mpatches.Patch(color=color, label=label)

        for idx, row in buildings.iterrows():
            geom = row.get(StudyBuildingSchema.DEBRIS_GEOM)
            if geom and not geom.is_empty: # Basic check before detailed check
                # Check if geom is a valid shapely geometry before accessing coords
                if hasattr(geom, 'exterior'):
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
                        for poly in geoms:
                            if poly and not poly.is_empty and hasattr(poly, 'exterior'):
                                patch = MplPolygon(
                                    list(poly.exterior.coords),
                                    facecolor='orange', alpha=0.2, edgecolor='black',
                                    linewidth=0.5, linestyle=':', zorder=10
                                )
                                # ax.add_patch(patch)

        traffic_handle = traffic_roads.plot(ax=ax, color='darkred', linewidth=1.5, linestyle='--', label='Traffic Routes', zorder=20, alpha=0.4)

        road_handle = None
        for idx, row in roads.iterrows():
            road_label = row.get(StudyRoadSchema.HAZUS_ROAD_CLASS) or row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS, "")
            road_label_str = str(road_label)
            is_hrd1 = "HRD1" in road_label_str
            is_bridge = road_label_str.startswith("HWB")
            line_width = 2.5 if is_hrd1 or is_bridge else 1.0
            plot_instance = roads.iloc[[idx]].plot(ax=ax, color='black', linewidth=line_width, alpha=0.6, zorder=5)
            if road_handle is None:
                 road_handle = mpatches.Patch(color='black', label='Roads', linewidth=0.25)


        # --- CORRECTED bounds function ---
        def bounds(gdf_list):
            all_bounds_arrays = []

            # Process standard geometry bounds first
            for gdf in gdf_list:
                 if not gdf.empty and gdf.geometry.is_valid.all(): # Check if GDF is not empty and geometries are valid
                      try:
                           # Ensure CRS is consistent if possible, otherwise ignore for bounds
                           b = gdf.total_bounds
                           if np.all(np.isfinite(b)): # Check if bounds are finite numbers
                                all_bounds_arrays.append(b)
                      except Exception as e:
                           print(f"Warning: Could not get total_bounds for a GeoDataFrame: {e}")


            # # Handle debris geometry bounds separately
            # for gdf in gdf_list:
            #      # Check if it's the buildings GDF (or any GDF with debris_geom)
            #      if StudyBuildingSchema.DEBRIS_GEOM in gdf.columns:
            #         # Extract the column which might contain geometries or None
            #         debris_col = gdf[StudyBuildingSchema.DEBRIS_GEOM]

            #         # Filter out None values before creating GeoSeries
            #         debris_geoms_only = debris_col.dropna()

            #         if not debris_geoms_only.empty:
            #             # Convert only the non-null geometries to a GeoSeries
            #             debris_geoseries = gpd.GeoSeries(debris_geoms_only)

            #             # Create mask for valid and non-empty geometries within the GeoSeries
            #             valid_mask = debris_geoseries.is_valid & ~debris_geoseries.is_empty

            #             # Filter the GeoSeries using the mask
            #             valid_debris_geoseries = debris_geoseries[valid_mask]

            #             # Check if any valid debris geometries remain
            #             if not valid_debris_geoseries.empty:
            #                 try:
            #                         b = valid_debris_geoseries.total_bounds
            #                         if np.all(np.isfinite(b)):
            #                             all_bounds_arrays.append(b)
            #                 except Exception as e:
            #                         print(f"Warning: Could not get total_bounds for debris geometries: {e}")


            if not all_bounds_arrays:
                print("Warning: Could not determine valid bounds for plotting. Using default [0,0,1,1].")
                return 0, 0, 1, 1 # Default fallback bounds

            bounds_array = np.array(all_bounds_arrays)
            x_min = np.min(bounds_array[:, 0])
            y_min = np.min(bounds_array[:, 1])
            x_max = np.max(bounds_array[:, 2])
            y_max = np.max(bounds_array[:, 3])

            # Add a small check for inverted bounds (though unlikely with min/max)
            if x_min > x_max: x_min, x_max = x_max, x_min
            if y_min > y_max: y_min, y_max = y_max, y_min

            return x_min, y_min, x_max, y_max
        # --- END OF CORRECTED bounds function ---

        x_min, y_min, x_max, y_max = bounds([buildings, roads, traffic_roads])

        padding = 0.05
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Avoid division by zero or negative range if bounds are weird
        if x_range <= 0: x_range = 1.0
        if y_range <= 0: y_range = 1.0

        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

        # --- (Keep Traffic Node ID labeling logic as before) ---
        if show_traffic_ids:
            plotted_node_ids = set()
            try:
                # Check if required columns exist before iterating
                if "init_node" not in traffic_roads.columns or "term_node" not in traffic_roads.columns:
                     raise KeyError(f"Missing 'init_node' or 'term_node' in traffic_roads GDF.")

                for idx, row in traffic_roads.iterrows():
                    geom = row.geometry
                    if isinstance(geom, LineString) and not geom.is_empty:
                        start_node_id = row[StudyRoadSchema.FROMNODE]
                        end_node_id = row[StudyRoadSchema.TONODE]
                        start_coord = geom.coords[0]
                        end_coord = geom.coords[-1]
                        nodes_to_plot = [
                            (start_node_id, start_coord[0], start_coord[1]),
                            (end_node_id, end_coord[0], end_coord[1])
                        ]
                        for node_id, node_x, node_y in nodes_to_plot:
                            if node_id not in plotted_node_ids:
                                ax.text(
                                    node_x, node_y, str(node_id), fontsize=9, color='darkred',
                                    ha='center', va='bottom', fontweight='bold',
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkred', boxstyle='round,pad=0.2'),
                                    zorder=30
                                )
                                plotted_node_ids.add(node_id)
                    elif geom and not isinstance(geom, LineString):
                         print(f"Warning: Traffic link {idx} geometry is not a LineString ({type(geom)}). Cannot label nodes.")

            except KeyError as e:
                 print(f"Error during traffic node labeling: {e}")
            except Exception as e:
                 print(f"An unexpected error occurred during traffic node labeling: {e}")


        # --- (Keep Road ID labeling logic as before) ---
        for i, building in enumerate(buildings.iterrows()):
            # print(building)
            center = building[1]["geometry"].centroid
            ax.text(
                center.x, center.y, str(str(building[1][StudyBuildingSchema.OCC_TYPE]) + "_" + str(i)),
                fontsize=10, color='black', ha='center', va='center', fontweight='bold', rotation=0
            )

        if show_road_ids:
            for idx, row in roads.iterrows():
                line = row.geometry
                if isinstance(line, LineString) and not line.is_empty:
                    midpoint = line.interpolate(0.5, normalized=True)
                    label_b = row.get(StudyRoadSchema.HAZUS_BRIDGE_CLASS)
                    label_r = row.get(StudyRoadSchema.HAZUS_ROAD_CLASS)
                    label_parts = []
                    if label_b and str(label_b).strip() and str(label_b).lower() != 'none': label_parts.append(str(label_b))
                    elif label_r and str(label_r).strip() and str(label_r).lower() != 'none': label_parts.append(str(label_r))
                    label_parts.append(str(idx))
                    label = "_".join(label_parts)

                    if label:
                        angle = 0
                        try:
                            if line.length > 1e-6:
                                p1 = line.interpolate(0.49, normalized=True)
                                p2 = line.interpolate(0.51, normalized=True)
                                dx = p2.x - p1.x
                                dy = p2.y - p1.y
                                if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                                    angle = np.degrees(np.arctan2(dy, dx))
                                    if angle > 90: angle -= 180
                                    if angle < -90: angle += 180
                        except Exception: pass
                        ax.text(
                            midpoint.x, midpoint.y, label, fontsize=12, color='darkslateblue', ha='center', va='center',
                            fontweight='normal', rotation=angle, rotation_mode='anchor',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.6, edgecolor='none'), zorder=25
                        )
                elif line and not isinstance(line, LineString):
                    print(f"Warning: Road {idx} geometry is not a LineString ({type(line)}). Cannot label.")

        # --- (Keep Title, Legend, Grid, Axis, Layout, Show logic as before) ---
        ax.set_title(plot_name, fontsize=18, fontweight='bold')

        legend_handles = list(building_handles.values())
        if road_handle: legend_handles.append(road_handle)

        traffic_legend_handle = mpatches.Patch(color='darkred', label='Traffic Routes', linewidth=0.25, linestyle='--')
        lines = ax.get_lines()
        for line in lines:
             if line.get_label() == 'Traffic Routes':
                  traffic_legend_handle = line
                  break
        legend_handles.append(traffic_legend_handle)

        # if any(StudyBuildingSchema.DEBRIS_GEOM in gdf.columns and not gdf[gdf[StudyBuildingSchema.DEBRIS_GEOM].notna()].empty for gdf in [buildings]):
        #      debris_patch = MplPolygon(
        #           [(0,0), (0,1), (1,1), (1,0)], facecolor='orange', alpha=0.2, edgecolor='black',
        #           linewidth=0.5, linestyle=':', label='Building Debris'
        #      )
        #      legend_handles.append(debris_patch)

        ax.legend(handles=legend_handles, loc='lower center', fontsize=10, frameon=True, edgecolor='black', ncol=1)
        ax.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        # ax.set_aspect('equal', adjustable='box') # Uncomment if aspect ratio is important

        plt.tight_layout()
        plt.show()

# --- Example Usage Placeholder (as before) ---
# ... (rest of your example usage code if any) ...
# --- Example Usage (requires dummy data or actual GeoDataFrames) ---
# Make sure your GeoDataFrames have the necessary columns ('geometry', 'occ_type',
# 'fromnode', 'tonode', etc.) and a defined CRS.

# Create dummy data for demonstration:
# roads_gdf = gpd.GeoDataFrame({
#     'geometry': [LineString([(0,0), (1,1)]), LineString([(1,1), (1,0)]), LineString([(0,1),(1,1)])],
#     'hazus_road_class': ['HRD2', 'HRD1', 'HRD2'],
#     'hazus_bridge_class': [None, None, 'HWB1'],
#     'fromnode': [1, 2, 3],
#     'tonode': [2, 4, 2]
# }, crs="EPSG:4326")

# buildings_gdf = gpd.GeoDataFrame({
#     'geometry': [Polygon([(0.2,0.2), (0.4,0.2), (0.4,0.4), (0.2,0.4)]), Polygon([(0.6,0.6), (0.8,0.6), (0.8,0.8), (0.6,0.8)])],
#     'occ_type': ['RES1', 'COM1'],
#     'str_typ2': ['W1', 'RM1'],
#     'debris_geom': [Polygon([(0.1,0.1), (0.5,0.1), (0.5,0.5), (0.1,0.5)]), None] # Example debris
# }, crs="EPSG:4326")

# # traffic_links_gdf should ideally be a subset or derived from roads_gdf, containing node IDs
# traffic_links_gdf = roads_gdf.iloc[[0, 2]].copy() # Example: traffic uses roads 0 and 2

# # Instantiate the class and call the method
# environment_visualizer = YourClass(roads_gdf, buildings_gdf, traffic_links_gdf)
# environment_visualizer.viz_environment(
#     plot_name="Study Area Environment",
#     show_traffic_ids=True, # Set to True to see node labels
#     show_road_ids=True
# )






