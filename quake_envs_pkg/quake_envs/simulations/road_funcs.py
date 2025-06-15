from dataclasses import dataclass
from typing import List, Optional, Final, Tuple, Dict
import random
import uuid
import math
import numpy as np
import overpy
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate
from pathlib import Path
import json
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import networkx as nx
import concurrent.futures
from enum import Enum
from .building_funcs import Building

# from road_config import (
#     OSMRoadSchema,
#     StudyRoadSchema,
#     FHWARoadReplacementCosts,
#     NBISchema,
#     HAZUSBridge_k3d_coefficients,
#     OSMHazusRoadMapper
# )
from .road_config import *


def get_osm_roads(
    bounds: Optional[Tuple[float, float, float, float]] = None,
    buffer: int = 150
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph]:
    """
    Retrieve OpenStreetMap road network data within specified bounds.

    Parameters:
    -----------
    bounds : Tuple[float, float, float, float], optional
        Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
    buffer : int, default 150
        Buffer distance in meters to expand the bounding box

    Returns:
    --------
    Tuple containing:
    - GeoDataFrame of nodes
    - GeoDataFrame of roads
    - NetworkX graph
    """
    # Input validation
    if bounds is None:
        raise ValueError("Bounds must be provided")

    if len(bounds) != 4:
        raise ValueError("Bounds must be a tuple of 4 coordinates (min_lon, min_lat, max_lon, max_lat)")

    # Calculate buffer expansion
    # Use more precise conversion factors
    expansion_deg_lat = buffer / 111320  # meters to degrees latitude
    expansion_deg_long = buffer / (111320 * np.cos(np.radians((bounds[1] + bounds[3]) / 2)))

    # Expand bounds with buffer
    expanded_bounds = (
        bounds[0] - expansion_deg_long,
        bounds[1] - expansion_deg_lat,
        bounds[2] + expansion_deg_long,
        bounds[3] + expansion_deg_lat
    )

    try:
        # Retrieve road network
        G = ox.graph_from_bbox(
            expanded_bounds,
            network_type='drive',
            simplify=True
        )

        # Convert graph to GeoDataFrames
        G_gdfs = ox.graph_to_gdfs(G, nodes=True, edges=True)
        G_gdf_nodes = G_gdfs[0].copy()
        G_gdf_roads = G_gdfs[1].copy()

        # Add local IDs and unique identifiers
        G_gdf_nodes['local_id'] = range(len(G_gdf_nodes))
        G_gdf_nodes['guid'] = [str(uuid.uuid4()) for _ in range(len(G_gdf_nodes))]

        # Create mapping from OSM node IDs to local IDs
        osm_to_local = dict(zip(G_gdf_nodes.index, G_gdf_nodes['local_id']))
        node_guid_dict = dict(zip(G_gdf_nodes.index, G_gdf_nodes['guid']))

        # Create one-way dictionary for roads
        one_way_dict = {
            (u, v): data.get('oneway', None)
            for u, v, data in G.edges(data=True)
        }

        # Add additional road attributes
        G_gdf_roads['local_id'] = range(len(G_gdf_roads))
        G_gdf_roads['local_u'] = G_gdf_roads.index.get_level_values(0).map(osm_to_local)
        G_gdf_roads['local_v'] = G_gdf_roads.index.get_level_values(1).map(osm_to_local)

        # Add node GUIDs and one-way information
        G_gdf_roads['fnodeguid'] = G_gdf_roads.index.get_level_values(0).map(node_guid_dict)
        G_gdf_roads['tnodeguid'] = G_gdf_roads.index.get_level_values(1).map(node_guid_dict)
        G_gdf_roads['one_way'] = G_gdf_roads.apply(
            lambda row: one_way_dict.get((row.name[0], row.name[1]), None),
            axis=1
        )

        return G_gdf_nodes, G_gdf_roads, G

    except Exception as e:
        raise RuntimeError(f"Error retrieving OSM road network: {str(e)}")

def get_road_unit_costs(
        highway_type: str
) -> Dict:
    if isinstance(highway_type, list) and highway_type:
        _highway_type = highway_type[0]
    else:
        _highway_type = highway_type
    return FHWARoadReplacementCosts(
            osm_highway_type=_highway_type
        ).get_costs_per_mile()

def map_study_road_data(
    roads_study_gdf: gpd.GeoDataFrame,
    osm_roads_nodes_gdf: gpd.GeoDataFrame,
    osm_roads_edges_gdf: gpd.GeoDataFrame,
):
        """
        Maps road-related information from OpenStreetMap (OSM) data to a study roads GeoDataFrame.

        This function processes the road data from OSM (in the form of nodes and edges)
        and calculates various road attributes such as unit costs, road lengths,
        and replacement costs, which are then mapped to a given study roads GeoDataFrame.

        Parameters:
        - roads_study_gdf (gpd.GeoDataFrame): A GeoDataFrame where processed road information will be stored.
        - osm_roads_nodes_gdf (gpd.GeoDataFrame): A GeoDataFrame containing road nodes from OSM.
        - osm_roads_edges_gdf (gpd.GeoDataFrame): A GeoDataFrame containing road edges from OSM.

        Returns:
        - gpd.GeoDataFrame: The updated `roads_study_gdf` containing additional calculated attributes
        like road lengths, replacement costs, and other road characteristics.
        """

        unit_costs = [sum(get_road_unit_costs(x).values()) for x in osm_roads_edges_gdf[OSMRoadSchema.HIGHWAY]]
        lens_miles = [x / 1609.34 for x in osm_roads_edges_gdf.to_crs(epsg=3857).geometry.length]
        lens_km = [x * 1609.34 for x in lens_miles]
        repl_costs = [unit_cost * len_mile for unit_cost, len_mile in zip(unit_costs, lens_miles)]

        roads_study_gdf['geometry'] = osm_roads_edges_gdf['geometry']
        roads_study_gdf[StudyRoadSchema.HIGHWAY] = osm_roads_edges_gdf[OSMRoadSchema.HIGHWAY]
        roads_study_gdf[StudyRoadSchema.UNIT_COST] =  unit_costs
        roads_study_gdf[StudyRoadSchema.REPL_COST] = repl_costs
        roads_study_gdf[StudyRoadSchema.LINKNWID] = osm_roads_edges_gdf['local_id']
        roads_study_gdf[StudyRoadSchema.FROMNODE] = osm_roads_edges_gdf['local_u']
        roads_study_gdf[StudyRoadSchema.TONODE] = osm_roads_edges_gdf['local_v']
        roads_study_gdf[StudyRoadSchema.DIRECTION] = [1 if x == True else 0 for x in osm_roads_edges_gdf['one_way']]
        roads_study_gdf[StudyRoadSchema.LEN_MILE] = lens_miles
        roads_study_gdf[StudyRoadSchema.LEN_KM] = lens_km
        roads_study_gdf[StudyRoadSchema.GUID] = uuid.uuid4()
        roads_study_gdf[StudyRoadSchema.FNODE_GUID] = osm_roads_edges_gdf['fnodeguid']
        roads_study_gdf[StudyRoadSchema.TNODE_GUID] = osm_roads_edges_gdf['tnodeguid']

        for idx in roads_study_gdf.index:
            osm_highway = roads_study_gdf.loc[idx, StudyRoadSchema.HIGHWAY]
            hazus_class = OSMHazusRoadMapper.get_hazus_road_type(osm_highway=osm_highway)
            road_width = OSMHazusRoadMapper.get_hazus_road_width(hazus_road_class=hazus_class)
            roads_study_gdf.loc[idx, StudyRoadSchema.WIDTH] = road_width
            roads_study_gdf.loc[idx, StudyRoadSchema.HAZUS_ROAD_CLASS] = hazus_class

        return roads_study_gdf

def map_bridges_to_roads(
    bridges_nbi_gdf: gpd.GeoDataFrame,
    roads_study_gdf: gpd.GeoDataFrame
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Map the nearest bridge to each road, ensuring no bridge is assigned to multiple roads.

    Args:
        bridges_nbi_gdf (gpd.GeoDataFrame): GeoDataFrame containing bridge points
        roads_study_gdf (gpd.GeoDataFrame): GeoDataFrame containing road geometries
            with MultiDiGraph index (u, v, key)

    Returns:
        gpd.GeoDataFrame: Roads GeoDataFrame with added 'bridge_id' column
    """
    def _filter_bridges(
        bridges: gpd.GeoDataFrame,
        roads: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Filter bridges to those within the bounding box of the roads.

        Args:
            bridges (gpd.GeoDataFrame): Input bridges GeoDataFrame
            roads (gpd.GeoDataFrame): Input roads GeoDataFrame

        Returns:
            gpd.GeoDataFrame: Filtered bridges within roads' bounding box
        """
        # Get the bounding box of the lines (minx, miny, maxx, maxy)
        lines_bounds = roads.bounds

        # Calculate the overall bounds
        minx = lines_bounds['minx'].min()
        miny = lines_bounds['miny'].min()
        maxx = lines_bounds['maxx'].max()
        maxy = lines_bounds['maxy'].max()

        # Filter bridges that are within the bounds
        filtered_bridges = bridges[
            (bridges.geometry.x >= minx) &
            (bridges.geometry.x <= maxx) &
            (bridges.geometry.y >= miny) &
            (bridges.geometry.y <= maxy)
        ]

        return filtered_bridges

    # Initialize bridge_id column with -1

    # Initialize bridge_id column with -1
    roads_study_gdf['bridge_id'] = -1

    # Filter bridges to those within the roads' bounding box
    gdf_bridges = _filter_bridges(bridges_nbi_gdf, roads_study_gdf)

    # Handle case of empty bridges GeoDataFrame
    if gdf_bridges.empty:
        return roads_study_gdf, gdf_bridges

    # For each bridge, find the road with minimum distance
    for bridge_idx, bridge_row in gdf_bridges.iterrows():
        bridge_point = bridge_row.geometry

        # Calculate minimum distance to each road
        min_distances = roads_study_gdf.geometry.apply(
            lambda road: bridge_point.distance(road)
        )

        # Find road with smallest minimum distance
        closest_road_idx = min_distances.idxmin()

        # Assign bridge to that road
        roads_study_gdf.at[closest_road_idx, 'bridge_id'] = bridge_idx

    return roads_study_gdf, gdf_bridges

def get_bridge_class(
    year_built: int,
    structure: int,
    max_span: float,
    num_spans: int
) -> str:
    """Determine HAZUS bridge classification based on bridge properties."""

    # Special cases for large spans
    if max_span > 150:
        return 'HWB1' if year_built < 1975 else 'HWB2'

    if num_spans == 1:
        return 'HWB3' if year_built < 1975 else 'HWB4'

    # Structure type classifications
    structure_classes = {
        (101, 106): ('HWB6', 'HWB7'),
        (205, 206): ('HWB8', 'HWB9'),
        (201, 206): ('HWB10', 'HWB11'),
        (301, 306): ('HWB13', 'HWB14'),
        (402, 410): ('HWB15', 'HWB16'),
        (501, 506): ('HWB18', 'HWB19'),
        (605, 606): ('HWB20', 'HWB21'),
        (601, 607): ('HWB22', 'HWB23')
    }



    for (start, end), (old_class, new_class) in structure_classes.items():
        if start <= structure <= end:
            return old_class if year_built < 1975 else new_class

    # Special cases for small spans
    if max_span < 20 and year_built < 1975:
        if 301 <= structure <= 306:
            return 'HWB24'
        if 402 <= structure <= 410:
            return 'HWB25'

    return 'HWB28'

def map_bridges_data(
    roads_study_gdf: gpd.GeoDataFrame,
    bridges_filtered: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Map bridge data to road segments and classify bridges according to HAZUS classes.

    Args:
        roads_study_gdf: GeoDataFrame containing road segments
        bridges_filtered: GeoDataFrame containing bridge data

    Returns:
        GeoDataFrame with added HAZUS bridge classifications
    """
    ## bridge ishape
    bridge_shapes = {
        0: [
            'HWB1', 'HWB2', 'HWB6', 'HWB7',
            'HWB8', 'HWB9', 'HWB12', 'HWB13',
            'HWB14', 'HWB17', 'HWB18', 'HWB19',
            'HWB20', 'HWB21', 'HWB24', 'HWB25'
        ]
        ## 1 is all else
    }


    # Extract bridge properties
    bridges_struct = {
        idx: int(f"{s1}{s2}")
        for idx, (s1, s2) in bridges_filtered[
            [NBISchema.STRUCTURE_1, NBISchema.STRUCTURE_2]
        ].iterrows()
    }

    # Initialize bridge class column
    roads_study_gdf[StudyRoadSchema.HAZUS_BRIDGE_CLASS] = 'None'

    # Map bridge classifications
    mask = roads_study_gdf['bridge_id'] > -1
    for idx in roads_study_gdf[mask].index:
        bridge_idx = roads_study_gdf.loc[idx, 'bridge_id']

        bridge_class = get_bridge_class(
            year_built=bridges_filtered.loc[bridge_idx, NBISchema.YEAR_BUILT],
            structure=bridges_struct[bridge_idx],
            max_span=bridges_filtered.loc[bridge_idx, NBISchema.LENGTH_MAX_SPAN],
            num_spans=bridges_filtered.loc[bridge_idx, NBISchema.NUM_SPANS]
        )
        skew_angle = bridges_filtered.loc[bridge_idx, NBISchema.SKEW_ANGLE]
        num_spans = bridges_filtered.loc[bridge_idx, NBISchema.NUM_SPANS]
        bridge_shape = 0 if bridge_class in bridge_shapes[0] else 1
        k3d_A, k3d_B = HAZUSBridge_k3d_coefficients.get_coefficients(bridge_class)

        roads_study_gdf.loc[idx, StudyRoadSchema.BRIDGE_SHAPE] = bridge_shape
        roads_study_gdf.loc[idx, StudyRoadSchema.SKEW_ANGLE] = skew_angle
        roads_study_gdf.loc[idx, StudyRoadSchema.NUM_SPANS] = num_spans
        roads_study_gdf.loc[idx, StudyRoadSchema.K3D_A] = k3d_A
        roads_study_gdf.loc[idx, StudyRoadSchema.K3D_B] = k3d_B
        roads_study_gdf.loc[idx, StudyRoadSchema.HAZUS_BRIDGE_CLASS] = bridge_class

    return roads_study_gdf

def get_road_repair_cost(
    hazus_road_class: str,
    damage_state: int,
    length_miles: float
):
    road_repair_cost_model = RoadReplacementCosts()
    costs = road_repair_cost_model.get_costs(hazus_road_class)
    lane_miles = length_miles * 4 if hazus_road_class == 'HRD2' else length_miles * 6

    if  0 < damage_state < 3:
        return costs['resurface'] * lane_miles
    elif damage_state != 0:
        return (costs['reconstruct'] + costs['resurface']) * lane_miles
    else:
        return 0

def get_bridge_repair_cost(
    hazus_bridge_class,
    damage_state,
):
    bridge_repair_cost_model = BridgeRepairCost()
    if damage_state == 0:
        return 0
    else:
        return bridge_repair_cost_model.get_cost(hazus_bridge_class)

def get_road_capacity_reduction(damage_state):
    if isinstance(damage_state, str):
        ds_dict = {
            DamageStates.UNDAMAGED: 0,
            DamageStates.SLIGHT: 1,
            DamageStates.MODERATE: 2,
            DamageStates.EXTENSIVE: 3,
            DamageStates.COMPLETE: 4
        }
        damage_state = ds_dict[damage_state]
    if damage_state > 2:
        return 0.9
    elif damage_state == 2:
        return 0.75
    elif damage_state == 1:
        return 0.1
    else:
        return 0.0

def get_bridge_capacity_reduction(damage_state):
    if isinstance(damage_state, str):
        ds_dict = {
            DamageStates.Undamaged: 0,
            DamageStates.SLIGHT: 1,
            DamageStates.MODERATE: 2,
            DamageStates.EXTENSIVE: 3,
            DamageStates.COMPLETE: 4
        }
        damage_state = ds_dict[damage_state]
    if damage_state == 4:
        return 1
    elif damage_state == 3:
        return 0.98
    elif damage_state == 2:
        return 0.7
    elif damage_state == 1:
        return 0.3
    else:
        return 0.0

def get_road_obs_bounds():
    repair_roads_model = RoadRepairDistributions()
    repair_bridges_model = BridgeRepairDistributions()
    min_rt_roads, max_rt_roads = repair_roads_model.compute_repair_time_bins()
    min_rt_bridges, max_rt_bridges = repair_bridges_model.compute_repair_time_bins()
    min_obs = min(min_rt_roads, min_rt_bridges)
    max_obs = max(max_rt_roads, max_rt_bridges)
    return min_obs, max_obs

def get_road_repair_time(damage_state):
    COMPONENTS_PER_ROAD = 6
    repair_time_model =  RoadRepairDistributions()
    med, disp = repair_time_model.get_distribution(damage_state)
    if med == 0:
        return 0
    med *= COMPONENTS_PER_ROAD
    _, max_rt = get_road_obs_bounds()
    sigma = math.sqrt(math.log(1 + disp**2))
    mu = math.log(med)

    # Sample from lognormal
    sample = np.random.lognormal(mean=mu, sigma=sigma)

    # Round up and clip to max
    rt = min(math.ceil(sample), max_rt)
    return rt

def get_bridge_repair_time(damage_state):
    repair_time_model = BridgeRepairDistributions()
    med, disp = repair_time_model.get_distribution(damage_state)
    if med == 0:
        return 0
    _, max_rt = get_road_obs_bounds()
    sigma = math.sqrt(math.log(1 + disp**2))
    mu = math.log(med)

    # Sample from lognormal
    sample = np.random.lognormal(mean=mu, sigma=sigma)

    # Round up and clip to max
    rt = min(math.ceil(sample), max_rt)
    return rt

class StudyRoadsAccessor:
    """
    An accessor class for managing study road data with flexible retrieval and manipulation methods.

    This class provides methods to:
    - Retrieve road networks from OpenStreetMap (OSM)
    - Set local road networks
    - Clear stored road data
    - Access current road data

    Attributes:
        _parent (object): The parent instance containing simulation context
        nodes (gpd.GeoDataFrame): Road network nodes
        links (gpd.GeoDataFrame): Road network links
        graph (nx.Graph): Road network graph
    """
    def __init__(self, parent_instance):
        """
        Initialize the StudyRoadsAccessor.

        Args:
            parent_instance (object): The parent simulation instance
        """
        self._parent = parent_instance
        self._validate_parent_instance()

        # Initialize road network components
        self.nodes: Optional[gpd.GeoDataFrame] = None
        self.links: Optional[gpd.GeoDataFrame] = None
        self.graph: Optional[nx.Graph] = None

    def _validate_parent_instance(self):
        """
        Validate that the parent instance has all required attributes.

        Raises:
            AttributeError: If required attributes are missing
        """
        required_attrs = [
            # 'verbose',
            # '_buildings_nsi_gdf',
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
        curr_roads_study_gdf = self._parent._roads_study_gdf

        return curr_roads_study_gdf

    def get_osm(self
    ) ->None:
        """
        Retrieve road network from OpenStreetMap (OSM) within building centres bounds.

        Returns:
            Tuple containing:
            - gpd.GeoDataFrame: OSM road nodes
            - gpd.GeoDataFrame: OSM road links
            - nx.Graph: Road network graph

        Raises:
            ValueError: If building centres data is not set
        """
        # Validate prerequisites
        if self._parent._buildings_study_gdf is None:
            raise ValueError("Building centres data is not set.")

        # Log initial road retrieval info if verbose mode is on
        self.__log(f"Retrieving roads with bounds of {len(self._parent._buildings_study_gdf)} building centres.")

        # Retrieve OSM road network
        _osm_roads_nodes_gdf, _osm_roads_edges_gdf, _osm_roads_graph_nx = get_osm_roads(self._parent.bounds)

        # Store road network components
        self.nodes = _osm_roads_nodes_gdf
        self.links = _osm_roads_edges_gdf
        self.graph = _osm_roads_graph_nx

        # Store graph in parent instance
        self._parent._roads_osm_graph = _osm_roads_graph_nx

        # Map road information
        self._parent._roads_study_gdf = map_study_road_data(
            gpd.GeoDataFrame(
                columns=['geometry'],
                geometry='geometry',
                crs="EPSG:4326"
            ),
            self.nodes,
            self.links
        )
        # Log retrieval details
        self.__log(f"Fetched {len(_osm_roads_edges_gdf['geometry'])} links and {len(_osm_roads_nodes_gdf['geometry'])} nodes from OSM.")

    def set_local(
        self,
        roads_study_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Set local road network to be used instead of OSM data.

        Args:
            nodes (gpd.GeoDataFrame): Local road nodes GeoDataFrame
            links (gpd.GeoDataFrame, optional): Local road links GeoDataFrame

        Returns:
            gpd.GeoDataFrame: First few rows of mapped local road network

        Raises:
            ValueError: If input nodes are invalid
            TypeError: If input is not a GeoDataFrame
        """
        # Map road information
        self._parent._roads_study_gdf = None
        self._parent._roads_study_gdf = roads_study_gdf

    def _validate_local_input(self,
        nodes: gpd.GeoDataFrame
    ) -> None:
        """
        Validate local road network input.

        Args:
            nodes (gpd.GeoDataFrame): Nodes GeoDataFrame to validate

        Raises:
            ValueError: If input is invalid
            TypeError: If input is not a GeoDataFrame
        """
        if nodes is None:
            raise ValueError('Missing Nodes GeoDataFrame input.')

        if not isinstance(nodes, gpd.GeoDataFrame):
            raise TypeError('Input must be a GeoDataFrame')

        if 'geometry' not in nodes.columns:
            raise ValueError('Input GeoDataFrame must have a geometry column')\

    def map_bridges(
        self,
        bridges_nbi_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Map the nearest bridge to each road, ensuring no bridge is assigned to multiple roads.

        Args:
            bridges_nbi_gdf (gpd.GeoDataFrame): GeoDataFrame containing bridge points

        Returns:
            Tuple containing:
            - gpd.GeoDataFrame: Roads GeoDataFrame with added 'bridge_id' column
            - gpd.GeoDataFrame: Filtered bridges GeoDataFrame
        """
        # Map bridges to roads
        roads_study_gdf, gdf_bridges = map_bridges_to_roads(
            bridges_nbi_gdf,
            self._parent._roads_study_gdf
        )
        roads_study_gdf = map_bridges_data(roads_study_gdf, gdf_bridges)
        self._parent._roads_study_gdf = roads_study_gdf

        self.__log("Bridges were mapped to their associated roads")

    def clear(self) -> None:
        """
        Clear all stored road network data.

        Resets study roads to an empty GeoDataFrame with geometry column
        and clears stored road network components.
        """
        # Reset road network components
        self.nodes = None
        self.links = None
        self.graph = None

        # Reset study roads to empty GeoDataFrame
        self._parent._roads_study_gdf = gpd.GeoDataFrame(
            columns=['geometry'],
            geometry='geometry',
            crs="EPSG:4326"
        )

        self.__log("Cleared all stored road network data.")

    def __log(self, message: str) -> None:
        """
        Log messages if verbose mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._parent.verbose:
            print(message)


class RoadAction(Enum):
    """
    Enum representing possible actions for a road in an RL environment.

    Attributes:
        DO_NOTHING (int): No action is taken on the road. Value = 0.
        REPAIR (int): Perform a repair for one time-step. Value = 1.
        MAJOR_REPAIR (int): Perform a repair until road is fully repaired. Value = 2.
    """
    DO_NOTHING = 0  # No action taken
    REPAIR = 1
    # DO_NOTHING_ = 2

    def __str__(self):
        """Returns a human-readable string representation of the action."""
        return self.name.replace("_", " ").title()  # "REPAIR" -> "Minor Repair"

class Road:
    def __init_empty(self):
        self.initial_damage_state = 0
        self.current_damage_state = 0
        self.initial_repair_time = 0
        self.initial_repair_cost = 0
        self.current_repair_cost = 0
        self.current_repair_time = 0
        self.capacity_red_damage_state = 0.0
        self.capacity_red_debris = 0.0
        self.capacity_reduction = 0.0

    def __init__(
        self,
        id,
        geometry: LineString,
        init_node: int,
        term_node: int,
        flow: float,
        capacity,
        length_miles,
        hazus_road_class,
        hazus_bridge_class,
        is_bridge,
        time_step_duration,
        traffic_idx,
        verbose: bool,
        stoch_ds: bool,
        calc_debris: bool,
        stoch_rt: bool,
        stoch_cost: bool
    ):
        self.stoch_ds = stoch_ds
        self.stoch_rt = stoch_rt
        self.stoch_cost = stoch_cost
        self.calc_debris = calc_debris
        self.cost_decay = "quadratic"
        # self.cost_decay = "linear"
        self.id = id
        self.geometry = geometry
        self.centroid = geometry.centroid
        self.init_node = init_node
        self.term_node = term_node
        self.flow = flow
        self.is_bridge = is_bridge
        self.road_class = hazus_road_class
        self.bridge_class = hazus_bridge_class
        self.capacity = capacity
        self.verbose = verbose
        self.traffic_idx = traffic_idx
        self.length_miles = length_miles
        self.time_step_duration = time_step_duration
        self.value = 0.0
        self.__init_empty()
        if self.is_bridge:
            self.max_rep_cost = get_bridge_repair_cost(
                self.bridge_class,
                4
            )
        else:
            self.max_rep_cost = get_road_repair_cost(
                self.road_class,
                4,
                self.length_miles
            )

    def reset(
        self,
        damage_state: int,
        capacity_red_debris: float
    ):
        ## ---------------------Damage State---------------------
        if self.stoch_ds:
            self.initial_damage_state = damage_state
            self.current_damage_state = damage_state
            if self.is_bridge:
                self.capacity_red_damage_state = get_bridge_capacity_reduction(damage_state=damage_state)
            else:
                self.capacity_red_damage_state = get_road_capacity_reduction(damage_state=damage_state)
        else:
            self.initial_damage_state = 4
            self.current_damage_state = 4
            self.capacity_red_damage_state = 1.0
        ## ---------------------Debris---------------------
        if self.calc_debris:
            self.capacity_red_debris = capacity_red_debris
            # print(f"Log----2)road_{self.id}: Initial debris capacity reduction: {self.capacity_red_debris}")
        else:
            self.capacity_red_debris = 0.0

        self.initial_capacity_reduction_damage_state = self.capacity_red_damage_state
        self.capacity_reduction = max(self.capacity_red_damage_state, self.capacity_red_debris)
        self.is_fully_repaired = self.current_damage_state == 0
        self.is_debris_free = self.capacity_red_debris == 0.0


        ##---------------------Repair---------------------
        if self.is_bridge: ## Bridges
            ## Repair Time
            if self.stoch_rt:
                self.initial_repair_time = get_bridge_repair_time(
                    self.initial_damage_state
                )
            else:
                self.initial_repair_time = self.initial_damage_state * 80
            ## Repair Cost
            if self.stoch_cost:
                self.initial_repair_cost = get_bridge_repair_cost(
                    self.bridge_class,
                    self.initial_damage_state
                )
            else:
                self.initial_repair_cost = self.initial_damage_state * 100000
        else: ## Roads
            ## Repair Time
            if self.stoch_rt:
                self.initial_repair_time = get_road_repair_time(
                    self.initial_damage_state
                )
            else:
                self.initial_repair_time = self.initial_damage_state * 40
            ## Repair Cost
            if self.stoch_cost:
                self.initial_repair_cost = get_road_repair_cost(
                    self.road_class,
                    self.initial_damage_state,
                    self.length_miles
                )
            else:
                self.initial_repair_cost = self.initial_damage_state * 100000

        self.current_repair_time = self.initial_repair_time
        self.current_repair_cost = self.initial_repair_cost
        # print(f"Log----road_{self.id}: Initial repair time: {self.current_repair_time}")

    def step(
        self,
        action: RoadAction,
        dependant_buildings: List[Building]
    ):
        self.__log(f"Stepping road object: {self.id} with action: {action}")
        self.dependant_buildings = dependant_buildings

        if action == RoadAction.DO_NOTHING:
            self.__log(f"Road {self.id} is doing nothing")
            state = self.current_repair_time
            reward = 0.0 ## reward for the transportation network is taken from the traffic model
            done = self.is_fully_repaired
            info = self.__get_info()
            info["road_has_repaired"] = False
            return info

        elif action == RoadAction.REPAIR:
            was_repaired = self.is_fully_repaired
            self.__log(f"Road {self.id} is undergoing minor repair")
            try:
                repair_cost = self.__step_repair()
                reward = repair_cost
            except Exception as e:
                reward = 0.0
                self.__log(f"Road already repaired,: {str(e)}")

            if was_repaired == self.is_fully_repaired:
                road_has_repaired = False
            else:
                road_has_repaired = True
            state = self.current_repair_time
            done = self.is_fully_repaired
            info = self.__get_info()
            info["road_has_repaired"] = road_has_repaired
            return info

        else:
            raise ValueError(f"Invalid action: {action}")

    def __step_repair(self):
        assert not self.is_fully_repaired, "Road is already fully repaired"
        assert self.initial_repair_time > 0, "Road has no repair time"

        # If any dependent buildings have debris, skip the repair step
        if any(b.has_debris for b in self.dependant_buildings):
            return self.current_repair_cost

        time_step_duration = self.time_step_duration

        # Track the repair progress
        elapsed_time = self.initial_repair_time - self.current_repair_time
        fraction_complete = elapsed_time / self.initial_repair_time

        # Update current repair time
        self.current_repair_time = max(0, self.current_repair_time - time_step_duration)

        # Apply cost reduction based on the chosen decay method
        if self.cost_decay == "linear":
            repair_fraction = time_step_duration / self.initial_repair_time
            # Linear decay
            cost_reduction = repair_fraction * self.initial_repair_cost
            self.current_repair_cost = max(0, self.current_repair_cost - cost_reduction)
        else:
            # Quadratic decay
            remaining_cost_fraction = (1 - fraction_complete) ** 2
            self.current_repair_cost = self.initial_repair_cost * remaining_cost_fraction

        # Apply damage state capacity reduction, using the quadratic decay as well
        remaining_capacity_fraction = 1 - (fraction_complete) ** 2
        self.capacity_red_damage_state = round(
            max(0.0, (
                self.initial_capacity_reduction_damage_state * remaining_capacity_fraction
            )), 6)
        # if self.current_damage_state == 0:
        #     self.capacity_red_damage_state = 0.0


        # Update the total capacity reduction
        self.capacity_reduction = max(self.capacity_red_damage_state, self.capacity_red_debris)

        # If repair is complete, finalize the repair process
        if self.current_repair_time == 0:
            self.is_fully_repaired = True
            self.current_repair_cost = 0
            self.current_damage_state = 0
            self.capacity_red_damage_state = 0.0
            self.capacity_reduction = max(self.capacity_red_damage_state, self.capacity_red_debris)
            return 0
        else:
            # If repair isn't complete, continue updating the damage state
            self.__step_damage_state()
            return self.current_repair_cost

    def __step_damage_state(self):
        steps = self.initial_damage_state
        if steps <= 0:
            return self.initial_damage_state

        days_per_step = self.initial_repair_time / steps
        completed_repair_days = self.initial_repair_time - self.current_repair_time

        levels_repaired = int(completed_repair_days // days_per_step)

        self.current_damage_state = max(self.initial_damage_state - levels_repaired, 0)

    def __get_info(self):
        info = {
                'damage_state': self.current_damage_state,
                'repair_time': self.current_repair_time,
                'repair_cost': self.current_repair_cost,
                'is_fully_repaired': self.is_fully_repaired,
                'is_debris_free': self.is_debris_free,
                'capacity_reduction': self.capacity_reduction,
                'capacity_reduction_debris': self.capacity_red_debris,
                'capacity_reduction_damage_state': self.capacity_red_damage_state
        }
        return info

    def __log(
        self,
        msg
    ) -> None:
        if self.verbose:
            print(msg)

    def __str__(self):
        return (
            f"Road ID: {self.id}\n"
            f"Type: {'Bridge' if self.is_bridge else 'Road'}\n"
            f"Damage State: {self.initial_damage_state}\n"
            f"Length (miles): {self.length_miles}\n"
            f"HAZUS Road Class: {self.hazus_road_class}\n"
            f"HAZUS Bridge Class: {self.bridge_class}\n"
            f"Repair Time: {self.repair_time} days\n"
            f"Repair Cost: ${self.repair_cost:,.2f}"
        )

def reset_road_objects(
    buildings: List[Building],
    roads_study_gdf: gpd.GeoDataFrame,
    roads: List[Road]
):
    for idx, row in roads_study_gdf.iterrows():
        ## this is done after all buildings reset()
        id = idx
        capacity_red_debris = -1
        for building in buildings:
            if building.access_road_id == id:
                if building.debris_capacity_reduction > capacity_red_debris:
                    capacity_red_debris = building.debris_capacity_reduction

        if capacity_red_debris == -1:
            capacity_red_debris = 0.0

        roads[idx].reset(
            damage_state=row[StudyRoadSchema.DAMAGE_STATE],
            capacity_red_debris=capacity_red_debris
        )

    return roads



def make_road_objects(
    buildings: List[Building],
    roads_study_gdf: gpd.GeoDataFrame,
    time_step_duration: int,
    traffic_net_df: pd.DataFrame
) -> List[Road]:
    road_objs = []
    for idx, row in roads_study_gdf.iterrows():
        id = idx
        capacity_red_debris = -1
        for building in buildings:
            if building.access_road_id == id:
                if building.debris_capacity_reduction > capacity_red_debris:
                    capacity_red_debris = building.debris_capacity_reduction

        if capacity_red_debris == -1:
            capacity_red_debris = 0.0

        # print(f"Log----1)road_{id}: Initial debris capacity reduction: {capacity_red_debris}")


        _link_index = row[StudyRoadSchema.TRAFFIC_LINK_INDEX]
        if _link_index is None:
            init_node = -1
            term_node = -1
        else:
            _traffic_row = traffic_net_df.loc[_link_index]
            init_node = _traffic_row['init_node']
            term_node = _traffic_row['term_node']
            capacity = _traffic_row['capacity']
        length_miles = row[StudyRoadSchema.LEN_MILE]
        hazus_road_class = row[StudyRoadSchema.HAZUS_ROAD_CLASS]
        hazus_bridge_class = row[StudyRoadSchema.HAZUS_BRIDGE_CLASS]
        is_bridge = True if hazus_bridge_class != 'None' else False
        traffic_idx = row[StudyRoadSchema.TRAFFIC_LINK_INDEX]
        t = time_step_duration
        road_obj = Road(
            id=id,
            geometry=row["geometry"],
            init_node=init_node,
            term_node=term_node,
            flow=0.0,
            capacity=capacity,
            length_miles=length_miles,
            hazus_road_class=hazus_road_class,
            hazus_bridge_class=hazus_bridge_class,
            is_bridge=is_bridge,
            time_step_duration=t,
            traffic_idx=traffic_idx,
            verbose=False,
            stoch_ds=True,
            calc_debris=True,
            stoch_rt=True,
            stoch_cost=True
        )
        road_objs.append(road_obj)

    return road_objs

def map_road_objects(
    roads_study_gdf: gpd.GeoDataFrame,
    roads: List[Road]
) -> gpd.GeoDataFrame:
    for idx, road_obj in enumerate(roads):
        roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_RED_DS] = road_obj.capacity_red_damage_state
        roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_RED_DEBRIS] = road_obj.capacity_red_debris
        roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_REDUCTION] = road_obj.capacity_reduction
    return roads_study_gdf
