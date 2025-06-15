from typing import List, Optional, Tuple
import random
import uuid
import math
import numpy as np
import overpy
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.affinity import scale, rotate, translate
import re
from enum import Enum


from .building_config import *
from .utils import DamageStates

def get_osm_bldg_footprints(
    bldg_centres: gpd.GeoDataFrame,
    search_radius: int = 50,
    osm_call_limit: int = None
) -> gpd.GeoDataFrame:
        """
        Convert a GeoDataFrame of points to corresponding building footprints from OSM.

        Args:
            bldg_centres (gpd.GeoDataFrame): GeoDataFrame with point geometries and NSI Schema ID.
            search_radius (int, optional): Search radius in meters around each point. Defaults to 50.
            osm_call_limit (Optional[int], optional): Limit OSM API calls. Defaults to None.

        Returns:
            gpd.GeoDataFrame: Building footprints with NSI and OSM IDs.

        Raises:
            ValueError: If input GeoDataFrame lacks required identification column.
        """
        # Verify input has required columns
        if NSISchema.FD_ID not in bldg_centres.columns:
            raise ValueError("Input GeoDataFrame must contain identification column")

        # Initialize Overpy API
        api = overpy.Overpass()

        # Lists to store building data
        building_geometries = []
        building_ids = []
        osm_ids = []  # List to store OSM IDs

        # Process each point
        for idx, row in bldg_centres.iterrows():

            if osm_call_limit is not None and  idx >= osm_call_limit:
                break

            point = row.at[NSISchema.GEOM]
            fd_id = row.at[NSISchema.FD_ID]
            lat = point.y
            lon = point.x

            try:
                # Query for buildings
                query = f"""
                [out:json][timeout:25];
                (
                way["building"](around:{search_radius},{lat},{lon});
                relation["building"](around:{search_radius},{lat},{lon});
                );
                out body;
                >;
                out skel qt;
                """

                # Execute the query
                result = api.query(query)

                # Process ways (most buildings)
                ways_polygons = []
                ways_osm_ids = []  # List to store osm_ids for ways
                for way in result.ways:
                    if way.nodes[0] == way.nodes[-1]:
                        coords = [(float(node.lon), float(node.lat)) for node in way.nodes]
                        poly = Polygon(coords)
                        ways_polygons.append(poly)
                        ways_osm_ids.append(way.id)  # Append osm_id of the way

                # Process relations (complex buildings)
                relations_polygons = []
                relations_osm_ids = []  # List to store osm_ids for relations
                for relation in result.relations:
                    if relation.tags.get('type') == 'multipolygon' and 'building' in relation.tags:
                        try:
                            outer_coords = []
                            for member in relation.members:
                                if member.role == 'outer':
                                    way = member.resolve()
                                    outer_coords.extend([(float(node.lon), float(node.lat))
                                                    for node in way.nodes])

                            if outer_coords:
                                poly = Polygon(outer_coords)
                                relations_polygons.append(poly)
                                relations_osm_ids.append(relation.id)  # Append osm_id of the relation
                        except Exception as e:
                            print(f"Error processing relation: {e}")

                # Combine all found polygons and osm_ids
                all_polygons = ways_polygons + relations_polygons
                all_osm_ids = ways_osm_ids + relations_osm_ids

                if all_polygons:
                    target_point = Point(lon, lat)

                    # Find closest building
                    try:
                        distances = [target_point.distance(poly) for poly in all_polygons]
                    except Exception:
                        distances = [math.dist((lon, lat), (poly.centroid.x, poly.centroid.y))
                                for poly in all_polygons]

                    closest_poly = all_polygons[np.argmin(distances)]
                    closest_osm_id = all_osm_ids[np.argmin(distances)]  # Get corresponding OSM ID

                    building_geometries.append(closest_poly)
                    building_ids.append(fd_id)
                    osm_ids.append(closest_osm_id)  # Store the OSM ID
                else:
                    print(f"No buildings found near point with fd_id {fd_id} at ({lat}, {lon})")
                    building_geometries.append(None)
                    building_ids.append(fd_id)
                    osm_ids.append(None)

            except Exception as e:
                print(f"Error processing point with fd_id {fd_id} at ({lat}, {lon}): {e}")
                building_geometries.append(None)
                building_ids.append(fd_id)
                osm_ids.append(None)

        # Create output GeoDataFrame
        footprints_gdf = gpd.GeoDataFrame(
            {'fd_id': building_ids, 'osm_id': osm_ids},  # Include osm_id in the attributes
            geometry=building_geometries,
            crs="EPSG:4326"
        )

        # Remove rows where no building was found (optional - comment out if you want to keep all points)
        footprints_gdf = footprints_gdf.dropna(subset=['geometry'])

        return footprints_gdf

def map_study_bldg_data(
    bldg_centres: gpd.GeoDataFrame,
    study_bldgs_gdf: gpd.GeoDataFrame,
    bldgs_brecs : List[Polygon],
    avg_dwell_size : int
) -> gpd.GeoDataFrame:
    """
    Map building information from NSI to INCORE schema.

    Args:
        bldg_centres (gpd.GeoDataFrame): Building center points from NSI.
        study_bldgs_gdf (gpd.GeoDataFrame): Target GeoDataFrame to populate.
        bldgs_brecs (List[Polygon]): Building footprint polygons.

    Returns:
        gpd.GeoDataFrame: Populated GeoDataFrame with mapped building information.
    """

    def _nsi_to_incore_coords(
        bldg_centres: gpd.GeoDataFrame,
        study_bldgs_gdf : gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame :
        """
        Convert NSI coordinate system to INCORE coordinate system.

        Args:
            bldg_centres (gpd.GeoDataFrame): Source building centers.
            study_bldgs_gdf (gpd.GeoDataFrame): Target GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with converted coordinates.
        """
        longs = bldg_centres[NSISchema.X]
        lats = bldg_centres[NSISchema.Y]

        for i, _ in enumerate(longs):
            long = longs[i]
            lat = lats[i]
            centroid = Point(long,lat)
            study_bldgs_gdf.at[i, StudyBuildingSchema.GEOM] = centroid

        return study_bldgs_gdf

    def _predict_str_types(
        bldg_centres : gpd.GeoDataFrame,
        study_bldgs_gdf : gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Predict HAZUS structure types based on NSI building characteristics.

        Args:
            bldg_centres (gpd.GeoDataFrame): Source building centers.
            study_bldgs_gdf (gpd.GeoDataFrame): Target GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Updated GeoDataFrame with predicted structure types.
        """
        predictor = StructuralTypePredictor()

        _bldg_types = bldg_centres[NSISchema.BLDGTYPE]
        _num_stories = bldg_centres[NSISchema.NUM_STORY]
        _sq_footage = bldg_centres[NSISchema.SQFT]
        _occ_types = bldg_centres[NSISchema.ST_DAMCAT]


        for i, _ in enumerate(_bldg_types):
            num_stories = _num_stories[i]
            sq_footage = _sq_footage[i]
            bldg_type = str(_bldg_types[i].upper())
            occtype = _occ_types[i]

            predicted_str_type_detailed = predictor.predict_str_type(
                study_label=bldg_type,
                num_stories=num_stories,
                sq_footage=sq_footage,
                building_occupancy=occtype
            )
            predicted_str_type_basic = predicted_str_type_detailed[:-1] if predicted_str_type_detailed[-1] in ['L', 'M', 'H'] else predicted_str_type_detailed
            study_bldgs_gdf.at[i, StudyBuildingSchema.STR_TYP2] = predicted_str_type_detailed
            study_bldgs_gdf.at[i, StudyBuildingSchema.STRUCT_TYP] = predicted_str_type_basic

        return study_bldgs_gdf

    study_bldgs_gdf['geometry'] = bldgs_brecs
    study_bldgs_gdf = _nsi_to_incore_coords(bldg_centres, study_bldgs_gdf)
    study_bldgs_gdf[StudyBuildingSchema.PARID] =  bldg_centres[NSISchema.CBFIPS]
    study_bldgs_gdf[StudyBuildingSchema.YEAR_BUILT] = bldg_centres[NSISchema.MED_YR_BLT]
    study_bldgs_gdf[StudyBuildingSchema.NO_STORIES] = [math.ceil(x) for x in bldg_centres[NSISchema.NUM_STORY]]
    study_bldgs_gdf[StudyBuildingSchema.OCC_TYPE] = bldg_centres[NSISchema.OCCTYPE]
    study_bldgs_gdf[StudyBuildingSchema.APPR_BLDG] = bldg_centres[NSISchema.VAL_STRUCT]
    study_bldgs_gdf[StudyBuildingSchema.CONT_VAL] = bldg_centres[NSISchema.VAL_CONT]
    study_bldgs_gdf[StudyBuildingSchema.EFACILITY] = ['TRUE' if x in ESSENTIAL_FACILITY_OCC_TYPES else 'FALSE' for x in bldg_centres[NSISchema.OCCTYPE]]
    for idx, row in bldg_centres.iterrows():
        dwelling_unit_calculator = DwellingUnitCalculator(
            populations=[
                row[NSISchema.POP2AMU65],
                row[NSISchema.POP2AMO65],
                row[NSISchema.POP2PMU65],
                row[NSISchema.POP2PMO65]
            ],
            avg_dwell_size=avg_dwell_size
        )
        study_bldgs_gdf.at[idx, StudyBuildingSchema.DWELL_UNIT] = math.ceil(dwelling_unit_calculator.calculate_dwell_units())

    study_bldgs_gdf = _predict_str_types(bldg_centres, study_bldgs_gdf)
    study_bldgs_gdf[StudyBuildingSchema.SQ_FOOT] = bldg_centres[NSISchema.SQFT]
    study_bldgs_gdf[StudyBuildingSchema.GUID] = str(uuid.uuid4())

    return study_bldgs_gdf

def validate_NSI(
    _buildings_nsi_gdf: gpd.GeoDataFrame,
    required_columns: Optional[List[str]] = None
) -> gpd.GeoDataFrame:
    """
    Validate and deduplicate the National Structures Inventory (NSI) GeoDataFrame.

    Parameters:
    -----------
    _buildings_nsi_gdf : gpd.GeoDataFrame
        Input GeoDataFrame representing structures inventory
    required_columns : Optional[List[str]], optional
        List of columns to validate, by default uses all columns from NSISchema

    Returns:
    --------
    gpd.GeoDataFrame
        Validated and deduplicated GeoDataFrame

    Raises:
    -------
    ValueError
        If required schema columns are missing
    """
    # Use all columns from NSISchema if no specific columns are provided
    if required_columns is None:
        required_columns = [NSISchema.FD_ID]

    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in _buildings_nsi_gdf.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}. "
            "Shapefile should conform to NSI Structures Inventory schema."
        )

    # Remove duplicate geometries, keeping the largest building by square footage
    NSI_gdf_remove_dupes = (
        _buildings_nsi_gdf
        .sort_values(NSISchema.SQFT, ascending=False)  # Sort by square footage in descending order
        .groupby(_buildings_nsi_gdf.geometry.apply(lambda geom: geom.wkt))  # Group by geometry WKT
        .first()  # Select the first (largest) building per unique geometry
        .reset_index(drop=True)  # Reset index to convert from grouped index back to regular DataFrame
    )

    return NSI_gdf_remove_dupes

def get_NSI_bounds(
    _buildings_nsi_gdf: gpd.GeoDataFrame
) -> tuple[
    tuple[float, float, float, float],
    Point
]:
    """
    Calculate the bounds and center point of a GeoDataFrame.

    Args:
        _buildings_nsi_gdf (gpd.GeoDataFrame): Input GeoDataFrame

    Returns:
        tuple: A tuple containing:
            - Bounds as (min_x, min_y, max_x, max_y)
            - Centre point as a Shapely Point

    Raises:
        ValueError: If input GeoDataFrame is empty
    """
    # Check if the GeoDataFrame is empty
    if _buildings_nsi_gdf.empty:
        raise ValueError("Input GeoDataFrame is empty")

    # Calculate total bounds
    bounds = _buildings_nsi_gdf.total_bounds
    min_x, min_y, max_x, max_y = bounds

    # Calculate average coordinates (center point)
    centre = Point((min_x + max_x) / 2, (min_y + max_y) / 2)

    return (min_x, min_y, max_x, max_y), centre

def calculate_debris_geometries(
        buildings_gdf:gpd.GeoDataFrame,
        story_height=3.0,
        volume_factor=0.2
) -> gpd.GeoDataFrame:
    """
    Calculate debris geometries for building footprints and add them to the GeoDataFrame.

    Args:
        buildings_gdf (GeoDataFrame): Input GeoDataFrame containing building geometries
        story_height (float): Height per story in meters (default: 3.0)
        volume_factor (float): Factor for calculating building volume (default: 0.2)

    Returns:
        GeoDataFrame: Copy of input GeoDataFrame with added debris geometry columns
    """
    # Create a copy to avoid modifying the input
    gdf = buildings_gdf.copy()

    def get_masonry_collape_polygon(
        building_geometry: Polygon
    ) -> Polygon:
        """Calculate the debris geometry for a building footprint.

        Args:
            building_geometry (Polygon): The building footprint geometry.

        Returns:
            Polygon: The debris geometry.
        """
        # Extract building dimensions
        coords = np.array(building_geometry.exterior.coords[:-1])
        side1 = np.linalg.norm(coords[1] - coords[0])
        side2 = np.linalg.norm(coords[2] - coords[1])

        # Calculate building parameters
        length = max(side1, side2)
        width = min(side1, side2)
        footprint_area = building_geometry.area
        height = building_stories * story_height
        building_volume = volume_factor * (footprint_area * height)

        # Calculate debris footprint scale factor
        scale_factor = CollapseFunctions.masonry_collapse(
            length, width, footprint_area, height, building_volume
        )


        # Generate scaled debris geometry
        debris_geom = scale(
            building_polygon,
            xfact=scale_factor,
            yfact=scale_factor,
            origin='center'
        )
        return debris_geom

    def get_concrete_collapse_polygon(
            building_geometry: Polygon,
            stories: int
    ) -> Polygon:
        collapse_mode = random.choice(["Aligned", "Skewed"])
        scale_factors =  CollapseFunctions.concrete_collapse(collapse_mode, stories)

        def shift_list(lst):
            # Ensure the length of the list is 4, otherwise return an error
            if len(lst) != 4:
                raise ValueError("The list must have exactly 4 elements.")

            # Find all the valid pairs for a-b and c-d
            valid_pairs = [(0, 2), (1, 3)]  # a-b, c-d pairs that are 2 positions apart

            # Randomly shuffle the valid pairs
            random.shuffle(valid_pairs)

            # Create a new list based on the valid pairings
            new_lst = [None] * 4
            for i, (first, second) in enumerate(valid_pairs):
                new_lst[first] = lst[i * 2]
                new_lst[second] = lst[i * 2 + 1]

            return new_lst

        scale_factors = shift_list(scale_factors)
        coords = np.array(building_geometry.exterior.coords[:-1])
        center = np.mean(coords, axis=0)

        side1 = np.linalg.norm(coords[1] - coords[0])
        side2 = np.linalg.norm(coords[2] - coords[1])
        vector = coords[1] - coords[0] if side1 > side2 else coords[2] - coords[1]
        angle = np.degrees(np.arctan2(vector[1], vector[0]))

        length = max(side1, side2)
        width = min(side1, side2)
        length_scales = [scale_factors[0], scale_factors[2]]
        width_scales = [scale_factors[1], scale_factors[3]]

        new_length = length * (1 + np.mean(length_scales))
        new_width = width * (1 + np.mean(width_scales))

        shift_x = length * (length_scales[0] - length_scales[1]) / 4
        shift_y = width * (width_scales[0] - width_scales[1]) / 4

        rect = Polygon([
            (-new_length/2, -new_width/2),
            (new_length/2, -new_width/2),
            (new_length/2, new_width/2),
            (-new_length/2, new_width/2)
        ])

        rad_angle = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(rad_angle), -np.sin(rad_angle)],
            [np.sin(rad_angle), np.cos(rad_angle)]
        ])
        shift = rot_matrix @ np.array([shift_x, shift_y])

        return translate(rotate(rect, angle),
                        xoff=center[0] + shift[0],
                        yoff=center[1] + shift[1])

    def get_all_other_collapse_polygon(
        building_geometry: Polygon,
        random_seed: Optional[int] = None
    ) -> Polygon:
        """
        Scale one random side of a rectangular polygon while maintaining perpendicular angles.

        Args:
            polygon: Input rectangle as shapely Polygon
            scale_factor: Amount to scale chosen side (e.g. 0.5 = 50% increase)
            random_seed: Optional seed for reproducible random side selection

        Returns:
            Scaled polygon with one side expanded/contracted
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Extract building dimensions
        coords = np.array(building_geometry.exterior.coords[:-1])
        side1 = np.linalg.norm(coords[1] - coords[0])
        side2 = np.linalg.norm(coords[2] - coords[1])

        width = min(side1, side2)
        height = building_stories * story_height

        new_width = CollapseFunctions.all_other_collapse(width, height)
        scale_factor = new_width / width
        # Select random side to scale
        side_to_scale = np.random.randint(0, 4)
        scale_factors = [0.0] * 4
        scale_factors[side_to_scale] = scale_factor

        # Get dimensions and orientation
        coords = np.array(building_geometry.exterior.coords[:-1])
        center = np.mean(coords, axis=0)
        side1 = np.linalg.norm(coords[1] - coords[0])
        side2 = np.linalg.norm(coords[2] - coords[1])
        vector = coords[1] - coords[0] if side1 > side2 else coords[2] - coords[1]
        angle = np.degrees(np.arctan2(vector[1], vector[0]))

        # Calculate new dimensions
        length = max(side1, side2)
        width = min(side1, side2)
        length_scales = [scale_factors[0], scale_factors[2]]
        width_scales = [scale_factors[1], scale_factors[3]]

        new_length = length * (1 + np.mean(length_scales))
        new_width = width * (1 + np.mean(width_scales))

        # Calculate center shift
        shift_x = length * (length_scales[0] - length_scales[1]) / 4
        shift_y = width * (width_scales[0] - width_scales[1]) / 4

        # Create base rectangle
        rect = Polygon([
            (-new_length/2, -new_width/2),
            (new_length/2, -new_width/2),
            (new_length/2, new_width/2),
            (-new_length/2, new_width/2)
        ])

        # Apply rotation and translation
        rad_angle = np.radians(angle)
        rot_matrix = np.array([
            [np.cos(rad_angle), -np.sin(rad_angle)],
            [np.sin(rad_angle), np.cos(rad_angle)]
        ])
        shift = rot_matrix @ np.array([shift_x, shift_y])

        return translate(
            rotate(rect, angle),
            xoff=center[0] + shift[0],
            yoff=center[1] + shift[1]
        )

    # Reproject to EPSG:3857 for accurate measurements
    gdf_3857 = gdf.to_crs(epsg=3857)
    debris_geometries = []

    # # Calculate debris geometry for each building
    # for _, building in gdf_3857.iterrows():
    #     # Extract building geometry and properties
    #     building_polygon = building.geometry
    #     if building_polygon is None:
    #         debris_geometries.append(None)
    #         continue
    #     building_stories = building[StudyBuildingSchema.NO_STORIES]

    #     if building[StudyBuildingSchema.STRUCT_TYP][0:1] in ['U', 'R']:
    #         debris_geometries.append(get_masonry_collape_polygon(building_polygon))
    #     elif building[StudyBuildingSchema.STRUCT_TYP][0:1] in ['C', 'PC']:
    #         debris_geometries.append(get_concrete_collapse_polygon(building_polygon, building_stories))
    #     else:
    #         debris_geometries.append(get_all_other_collapse_polygon(building_polygon))
    # Calculate debris geometry for each building
    for _, building in gdf_3857.iterrows():
        # Extract building geometry and properties
        building_polygon = building.geometry
        if building_polygon is None:
            debris_geometries.append(None)
            continue

        building_stories = building[StudyBuildingSchema.NO_STORIES]
        struct_type = building[StudyBuildingSchema.STRUCT_TYP]

        # Determine collapse polygon
        if struct_type[0:1] in ['U', 'R']:
            debris_poly = get_masonry_collape_polygon(building_polygon)
        elif struct_type[0:1] in ['C', 'PC']:
            debris_poly = get_concrete_collapse_polygon(building_polygon, building_stories)
        else:
            debris_poly = get_all_other_collapse_polygon(building_polygon)

        # Enlarge the debris polygon by 10% of its bounding box
        if debris_poly is not None and not debris_poly.is_empty:
            bounds = debris_poly.bounds  # (minx, miny, maxx, maxy)
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            buffer_dist = 0.2 * max(width, height) / 2  # 20% on all sides
            debris_poly = debris_poly.buffer(buffer_dist, cap_style="flat")

        debris_geometries.append(debris_poly)

    # Add debris geometries to GeoDataFrame
    gdf_3857[StudyBuildingSchema.DEBRIS_GEOM] = debris_geometries

    # Convert debris geometries to EPSG:4326
    gdf_3857[StudyBuildingSchema.DEBRIS_GEOM] = gdf_3857[StudyBuildingSchema.DEBRIS_GEOM].apply(
        lambda geom: gpd.GeoSeries([geom], crs=3857).to_crs(epsg=4326).iloc[0]
    )

    # Return final GeoDataFrame in EPSG:4326
    return gdf_3857.to_crs(epsg=4326)

def get_structural_repair_cost(
    occ_type: str,
    num_stories: int,
    sqft: float,
    damage_state_probs: np.array=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    bldg_repl_cost: float = None
) -> float:
    """
    Equation 11-1 from HAZUS MR4 Technical Manual, Chapter 11: Earthquake Model.
    """
    if bldg_repl_cost is None:
        bldg_replacement_cost_model = BuildingReplacementCosts()
        bldg_replacement_cost = bldg_replacement_cost_model.get_costs(occupancy_type=occ_type, num_stories=num_stories, sqft=sqft)
        return bldg_replacement_cost
    else:
        bldg_replacement_cost = bldg_repl_cost
        structural_repair_cost_ratios_model = StructuralRepairCostRatios()
        structural_repair_cost_ratios = (structural_repair_cost_ratios_model.get_repair_cost_ratios(occ_type) / 100)
        # print(f"damage state probs: {damage_state_probs}")
        # print(f"structural repair cost ratios: {structural_repair_cost_ratios}")
        # print(f"bldg replacement cost: {bldg_replacement_cost}")
        # print(f"sum of products: {np.sum(damage_state_probs[1:] * structural_repair_cost_ratios)}")
        struct_rep_cost = int(bldg_replacement_cost * np.sum(
                damage_state_probs[1:] * structural_repair_cost_ratios
            )
        )
        # print(f"structural repair cost: {struct_rep_cost}")
        # print(f"bldg replacement cost: {bldg_replacement_cost}")
        return struct_rep_cost

def get_income_loss(
    occ_type: str,
    sqft: float,
    repair_time: int=0,
    damage_state_probs: np.array=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    max_income: int = None
) -> int:
    """Calculate yearly income loss and maximum potential income for a building.

    This function implements equation 11-15 from:
    https://www.fema.gov/sites/default/files/documents/fema_hazus-earthquake-model-technical-manual-6-1.pdf
    It computes the yearly loss of service (YLOS) and maximum yearly income (YMAX)
    for a building based on its occupancy type, square footage, and probability of
    different damage states.

    Args:
        occ_type: String representing the occupancy type of the building.
        sqft: Float representing the total square footage of the building.
        damage_state_probs: Numpy array containing probabilities for different damage states.

    Returns:
        Tuple[int, int]: A tuple containing:
            - YLOS: Yearly loss of service (income loss) in dollars
            - YMAX: Maximum potential yearly income in dollars

    Note:
        The calculation takes into account:
        - Recapture factors specific to the occupancy type
        - Daily income rates per occupancy type
        - Loss of function time for different damage states
    """
    # Initialize recovery data object to access building-specific parameters
    recovery_building_data = BuildingRecoveryData()

    # Get recapture factors that represent the portion of income that can be recovered
    recap_factors = recovery_building_data.get_recapture_factors(occ_type)

    # Get daily income rates for the specific occupancy type
    income = recovery_building_data.get_income_per_day(occ_type)

    # Get expected loss of function time for each damage state
    loss_of_functionality = recovery_building_data.get_loss_of_function_time(occ_type)

    # Calculate yearly loss of service (YLOS) considering damage probabilities
    # and income loss factors
    YLOS = sqft * np.sum((1 - recap_factors) * income) * np.sum(
        damage_state_probs * loss_of_functionality
    )

    # Account for repair time in yearly income loss (YLOS) calculation
    # repair time is in days so yearly income loss (YLOS) is converted to daily income loss
    YLOS += (YLOS / 365) * repair_time

    # Calculate maximum potential yearly income (YMAX)
    YMAX = sqft * np.sum(income) * 365
    YLOS = min(YMAX, YLOS)


    return np.ceil(YLOS) if max_income is not None else np.ceil(YMAX)

def get_relocation_cost(
    occtype: str,
    sqft: float,
    damage_state_probs: np.array=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    max_reloc_cost: int = None
) -> int:
    """
    Calculate relocation cost for a building, optionally capped at a maximum value.

    Args:
        occtype: Occupancy type of the building.
        sqft: Total square footage.
        damage_state_probs: Probabilities for different damage states.
        max_reloc_cost: Maximum allowed relocation cost (optional).

    Returns:
        int: Final relocation cost, capped if max_reloc_cost is provided.
    """
    recovery_building_data = BuildingRecoveryData()

    if max_reloc_cost is None:
        relocation_cost = recovery_building_data.get_relocation_cost(
            occupancy_type=occtype,
            sqft=sqft,
            damage_state_probs=np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        )

    else:
        relocation_cost = recovery_building_data.get_relocation_cost(
            occupancy_type=occtype,
            sqft=sqft,
            damage_state_probs=damage_state_probs
        )
        relocation_cost = min(relocation_cost, max_reloc_cost)

    return int(relocation_cost)

def get_loss_of_function_time(
    occ_type: str,
    damage_state_probs: np.array
) -> float:
    """Calculate the expected loss of function time based on damage state probabilities.

    Computes the weighted average of loss of function times for different damage
    states using their respective probabilities of occurrence.

    Args:
        occ_type: String representing the occupancy type of the building.
        damage_state_probs: Numpy array containing probabilities for different damage states.

    Returns:
        float: Expected loss of function time weighted by damage state probabilities.
    """
    # Initialize recovery data object to access building-specific parameters
    recovery_building_data = BuildingRecoveryData()

    # Get expected loss of function time for each damage state
    loss_of_functionality_times = recovery_building_data.get_loss_of_function_time(occ_type)


    # Calculate expected loss of function time as weighted average
    expected_LOF_time = np.sum(loss_of_functionality_times * damage_state_probs)

    return math.ceil(expected_LOF_time)

def get_debris_weight(
    str_type: str
):
    debris_unit_weight = DebrisUnitWeight()
    total_weight = debris_unit_weight.get_unit_weight(str_type)

    return total_weight

def get_debris_cleanup_time(
    debris_weight: int,  # weight in tons
    trucks_per_day: int,  # number of trucks available to building
    loading_time: int = 1,  # loading time of each truck in hours
    truck_capacity: int = 5,  # weight capacity of each truck in tons
    travel_time_to_temp_depot: float = 2,  # travel time to temporary disposal in hours
    working_hours_per_day: int = 8
) -> int:
    """
    Calculate the number of working days needed to clean up debris.

    Args:
        debris_weight: Total weight of debris in tons
        trucks_per_day: Number of trucks available per day
        loading_time: Time to load each truck in hours (default: 1)
        truck_capacity: Weight capacity of each truck in tons (default: 5)
        travel_time_to_temp_depot: One-way travel time to depot in hours (default: 2)
        working_hours_per_day: Number of working hours per day (default: 8)

    Returns:
        int: Number of working days needed, rounded up to the nearest day

    Raises:
        ValueError: If any input parameters are less than or equal to 0
    """
    # Validate input parameters
    if any(param <= 0 for param in [
        debris_weight, trucks_per_day, loading_time,
        truck_capacity, travel_time_to_temp_depot, working_hours_per_day
    ]):
        raise ValueError("All parameters must be positive numbers")

    # Calculate number of truck trips needed
    truck_trips_needed = debris_weight / truck_capacity

    # Calculate total travel time (subtract return trip for last load)
    total_travel_time = truck_trips_needed * (
        2 * travel_time_to_temp_depot
    ) - travel_time_to_temp_depot

    # Calculate total loading time considering parallel operations
    total_clearing_time = (truck_trips_needed * loading_time) / trucks_per_day

    # Sum up total working time needed
    total_working_time = total_travel_time + total_clearing_time

    # Convert to working days
    working_days_needed = total_working_time / working_hours_per_day

    return math.ceil(working_days_needed)

def _get_hosp_beds(sqft: float, occtype: str) -> int:
    """Calculate the number of hospital beds based on building square footage."""
    if occtype == "COM6": # Hospital occupancy type
        building_rec_data = BuildingRecoveryData()
        n_beds = building_rec_data.get_num_hosp_beds(sqft=sqft, occtype=occtype)
        return n_beds
    return 0

def _get_num_doctors(sqft: float, occtype: str) -> int:
    """Calculate the number of doctors based on building square footage."""
    if occtype == "COM6": # Hospital occupancy type
        building_rec_data = BuildingRecoveryData()
        n_doctors = building_rec_data.get_num_doctors(sqft=sqft)
        return n_doctors
    return 0


def get_building_importance(occ_type: str) -> float:
    pass

def get_building_value(
    undisturbed_income: float,
    nominal_income: float,
    sqft: float,
    nominal_sqft: float,
    is_essential: bool,
    damage_state: int
) -> float:
    income_value = undisturbed_income / nominal_income
    ds_value = damage_state / len(DamageStates)
    sqft_value = sqft / nominal_sqft
    essential_value = 1 if is_essential else 0.5

    value = income_value * ds_value * sqft_value * essential_value
    return value

def get_building_obs_bounds():
    repair_model = BuildingRecoveryData()
    min_rt, max_rt = repair_model.compute_repair_time_bins()
    return min_rt, max_rt

def get_building_repair_time(
    occ_type: str,
    damage_state: int,
    random_state=None
) -> int:
    """
    Sample repair time based on damage state with damage state-dependent coefficient of variation.

    Parameters:
    -----------
    damage_state : int
        Damage state (0=None, 1=Slight, 2=Moderate, 3=Extensive, 4=Complete)
    mean_repair_times : np.ndarray
        Array of mean repair times for each damage state
    random_state : int, optional
        Random seed for reproducibility

    Returns:
    --------
    integer
        Sampled repair time
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Validate damage state
    if not (isinstance(damage_state, (int, np.integer)) and 0 <= damage_state <= 4):
        raise ValueError("Damage state must be an integer between 0 and 4")
    if damage_state == 0:
        return 0
    building_recovery_model = BuildingRecoveryData()
    mean_repair_times = building_recovery_model.get_repair_time(occ_type)

    # Get mean repair time for the damage state
    mean_time = mean_repair_times[damage_state]

    # Sample CV based on damage state
    if damage_state in {1, 2}:  # Slight or Moderate
        cv = np.random.uniform(0.15, 0.20)
    elif damage_state in {3, 4}:  # Extensive or Complete
        cv = np.random.uniform(0.35, 0.40)
    else:  # No damage (state 0)
        return 0

    # Convert lognormal mean and COV to normal mu and sigma
    sigma = np.sqrt(np.log(1 + cv**2))
    mu = np.log(mean_time) - 0.5 * sigma**2

    # Sample repair time from lognormal distribution
    repair_time = np.random.lognormal(mean=mu, sigma=sigma)

    # Clamp repair time to [0, max_rt]
    _, max_rt = get_building_obs_bounds()
    repair_time = max(0, repair_time)
    repair_time = min(repair_time, max_rt)

    return int(repair_time)

def get_building_downtime_delay(
    damage_state: int,
    is_essential: bool,
    num_stories: int,
    financing_method: str,
    random_state=None
) -> int:
    if random_state is not None:
        np.random.seed(random_state)

    downtime_delays = BuildingDowntimeDelays(
        essential=is_essential,
        num_stories=num_stories,
        financing_method=financing_method,
        damage_state=damage_state
    )
    downtime_delay = downtime_delays.get_delay_time()

    return int(downtime_delay)

class StudyBuildingsAccessor:
    """
    An accessor class for managing study building data with flexible retrieval and manipulation methods.

    This class provides methods to:
    - Retrieve building footprints from OpenStreetMap (OSM)
    - Set local building footprints
    - Clear stored building data
    - Access current building data

    Attributes:
        _parent (object): The parent instance containing simulation context
    """

    def __init__(self, parent_instance):
        """
        Initialize the StudyBuildingsAccessor.

        Args:
            parent_instance (object): The parent simulation instance
        """
        self._parent = parent_instance


    def __call__(self) -> gpd.GeoDataFrame:
        """
        Return the currently set buildings.

        Returns:
            gpd.GeoDataFrame: Currently stored study buildings
        """
        curr_buildings_study_gdf = self._parent._buildings_study_gdf

        return curr_buildings_study_gdf

    def get_osm(self) -> gpd.GeoDataFrame:
        """
        Retrieve building footprints from OpenStreetMap (OSM).

        Returns:
            gpd.GeoDataFrame: Mapped OSM building footprints

        Raises:
            ValueError: If building centres data is not set
        """
        # Validate prerequisites
        if self._parent._buildings_nsi_gdf is None:
            raise ValueError("Building centres data is not set.")

        # Log initial building centres info if verbose mode is on
        self._log(f"Using {len(self._parent._buildings_nsi_gdf)} building centres.")

        # Download building footprints from OSM
        study_ftprnts_gdf = get_osm_bldg_footprints(
            self._parent._buildings_nsi_gdf,
            self._parent._osm_search_radius,
            osm_call_limit=self._parent.osm_call_limit
        )
        self._log(f"Retrieved {len(study_ftprnts_gdf)} building footprints.")

        # Compute minimum rotated rectangles
        study_brecs = [
            x.minimum_rotated_rectangle for x in study_ftprnts_gdf['geometry']
        ]
        self._log(f"Computed minimum rotated rectangles for {len(study_brecs)} buildings.")

        # Map building information
        self._parent._buildings_study_gdf = map_study_bldg_data(
            self._parent._buildings_nsi_gdf,
            gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs="EPSG:4326"),
            study_brecs,
            self._parent._avg_dwell_size
        )

        # Log mapping results
        not_none_count = self._parent._buildings_study_gdf['geometry'].notna().sum()
        none_count = len(self._parent._buildings_study_gdf['geometry']) - not_none_count
        self._log(f"Study buildings mapping from OSM complete: \n -- Found: {not_none_count}, Not Found: {none_count}")

        return self._parent._buildings_study_gdf

    def get_debris(self) -> gpd.GeoDataFrame:
        self._parent._buildings_study_gdf = calculate_debris_geometries(self._parent._buildings_study_gdf)

    def set_local(self,
        buildings_study_gdf: gpd.GeoDataFrame
    ) -> None:
        """
        Set local building footprints to be used instead of OSM data.

        Args:
            buildings_study_gdf (gpd.GeoDataFrame): Local building footprints GeoDataFrame

        Returns:
            gpd.GeoDataFrame: First few rows of mapped local building footprints

        Raises:
            ValueError: If input GeoDataFrame is invalid
            TypeError: If input is not a GeoDataFrame
        """
        def string_to_shapely_point(input_string):
            # Use regular expression to extract coordinates from the input string
            match = re.match(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)', input_string)
            if match:
                longitude = float(match.group(1))
                latitude = float(match.group(2))
            else:
                # Try extracting coordinates directly if no 'POINT' keyword is present
                coords = re.findall(r'-?\d+\.\d+', input_string)
                if len(coords) == 2:
                    longitude, latitude = map(float, coords)
                else:
                    raise ValueError("Invalid coordinate string format")

            # Create and return a Shapely Point object
            return Point(longitude, latitude)

        # Apply function to the column if necessary
        if isinstance(buildings_study_gdf[StudyBuildingSchema.GEOM][0], str):
            buildings_study_gdf[StudyBuildingSchema.GEOM] = [
                string_to_shapely_point(x) for x in buildings_study_gdf[StudyBuildingSchema.GEOM]
            ]

        buildings_study_gdf = buildings_study_gdf[buildings_study_gdf['geometry'].notna()]
        self._parent._buildings_study_gdf = buildings_study_gdf
        self._parent.bounds, self._parent.center = get_NSI_bounds(buildings_study_gdf)

    def clear(self) -> None:
        """
        Clear all stored building footprints.

        Resets study buildings to an empty GeoDataFrame with geometry column.
        """
        self._parent._buildings_study_gdf = gpd.GeoDataFrame(
            columns=['geometry'],
            geometry='geometry',
            crs="EPSG:4326"
        )

        self._log("Cleared all stored building footprints.")

    def _log(self, message: str) -> None:
        """
        Log messages if verbose mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._parent.verbose:
            print(message)


class BuildingAction(Enum):
    DO_NOTHING = 0
    REPAIR = 1
    # CLEAR_DEBRIS = 2

    def __str__(self):
        """Returns a human-readable string representation of the action."""
        return self.name.replace("_", " ").title()  # "REPAIR" -> "Minor Repair"



class Building:
    def __init_empty(self):
        self.initial_damage_state = 0
        self.current_damage_state = 0
        self.initial_repair_time = 0
        self.initial_structural_repair_cost = 0
        self.current_structural_repair_cost = 0
        self.initial_income_loss = 0
        self.current_income = 0
        self.initial_loss_of_function_time = 0
        self.current_loss_of_function_time = 0
        self.initial_relocation_cost = 0
        self.current_relocation_cost = 0

    def __init__(
        self,
        id: str,
        geometry: Polygon,
        occtype: str,
        str_type: str,
        num_stories: int,
        sqft: float,
        is_essential: bool = False,
        access_road_id: int = -1,
        debris_capacity_reduction: float = 0.0,
        trucks_per_day: float = 1.0,
        time_step_duration: int = 7,
        verbose: bool = True,
        is_under_repair: bool = False,
        stoch_ds: bool = False,
        stoch_rt: bool = False,
        calc_debris: bool = False,
        stoch_cost: bool = False,
        stoch_inc_loss: bool = False,
        stoch_loss_of_function: bool = False,
        stoch_relocation_cost: bool = False,
        cost_decay: str = "quadratic",
        income_loss_decay: str = "quadratic"
    ):
        self.id = id
        self.geometry = geometry
        self.centroid = geometry.centroid
        self.occtype = occtype
        self.str_type = str_type
        self.num_stories = num_stories
        self.sqft = sqft
        self.is_essential = is_essential
        self.access_road_id = access_road_id
        self.debris_capacity_reduction = debris_capacity_reduction
        self.trucks_per_day = trucks_per_day
        self.time_step_duration = time_step_duration
        self.verbose = verbose
        self.is_under_repair = is_under_repair
        self.stoch_ds = stoch_ds
        self.stoch_rt = stoch_rt
        self.calc_debris = calc_debris
        self.stoch_cost = stoch_cost
        self.stoch_inc_loss = stoch_inc_loss
        self.stoch_loss_of_function = stoch_loss_of_function
        self.stoch_relocation_cost = stoch_relocation_cost
        self.cost_decay = cost_decay
        self.income_loss_decay = income_loss_decay

        self.value = 0.0

        self.max_income = get_income_loss(
            occ_type=self.occtype,
            sqft=self.sqft
        )
        self.max_rep_cost = get_structural_repair_cost(
            occ_type=self.occtype,
            num_stories=self.num_stories,
            sqft=self.sqft
        )
        self.max_reloc_cost = get_relocation_cost(
            occtype=self.occtype,
            sqft=self.sqft
        )
        if self.is_essential:
            self.initial_critical_func = 1.0
        else:
            self.initial_critical_func = 0.0
        self.current_critical_func = self.initial_critical_func
        self.initial_beds = _get_hosp_beds(self.sqft, self.occtype)
        self.initial_doctors = _get_num_doctors(self.sqft, self.occtype)
        self.__init_empty()


    def reset(self, damage_state_probs: np.array, debris_capacity_reduction: float):

        ## ---------------------Damage State---------------------
        if self.stoch_ds:
            self.damage_state_probs = damage_state_probs
            self.initial_damage_state = np.random.choice(
                len(damage_state_probs),
                p=damage_state_probs
            )
        else:
            self.damage_state_probs = np.array([0.0,0.0,0.0,0.0,1.0])
            self.initial_damage_state = 4

        self.current_damage_state = self.initial_damage_state

        ## ---------------------Debris---------------------
        if self.calc_debris:
            self.has_debris = self.current_damage_state >= 3
            if self.has_debris:
                self.debris_capacity_reduction = debris_capacity_reduction
            else:
                self.debris_capacity_reduction = 0.0
        else:
            self.has_debris = False
        if self.has_debris:
            self.debris_weight = get_debris_weight(
                self.str_type
            )
            self.debris_cleanup_time = get_debris_cleanup_time(
                self.debris_weight,
                self.trucks_per_day
            )
            self.current_debris_cleanup_time = self.debris_cleanup_time
        else:
            self.debris_weight = 0
            self.debris_cleanup_time = 0
            self.current_debris_cleanup_time = 0
        # print(f"Has debris: {self.has_debris}, debris weight: {self.debris_weight}, debris cleanup time: {self.debris_cleanup_time}")
        # print(f"Debris capacity reduction: {self.debris_capacity_reduction}")

        ## ---------------------Repair---------------------
        if self.stoch_rt:
            self.initial_repair_time = get_building_repair_time(
                occ_type=self.occtype,
                damage_state=self.initial_damage_state
            )
        else:
            self.initial_repair_time = self.initial_damage_state * 40
        self.current_repair_time = self.initial_repair_time
        # print(f"Log----buiding_{self.id}: Initial repair time: {self.initial_repair_time}")
        ## is repaired var
        self.is_fully_repaired = self.initial_damage_state == 0
        self.is_functional = self.initial_damage_state == 0
        self.time_step_after_repair = -1

        # repair cost / $
        if self.stoch_cost:
            self.initial_structural_repair_cost = get_structural_repair_cost(
                self.occtype,
                self.num_stories,
                self.sqft,
                self.damage_state_probs,
                self.max_rep_cost
            )
        else:
            self.initial_structural_repair_cost = self.initial_damage_state * 100000

        if self.current_damage_state == 0: ## TODO what the fuck this took so long to fix
            self.current_structural_repair_cost = 0.0
        else:
            self.current_structural_repair_cost = self.initial_structural_repair_cost

        ## ---------------------Functionality---------------------

        self.current_beds = self.get_hosp_beds()

        self.current_doctors = self.get_doctors()


        self.current_critical_func = self.get_critical_functionality()

        ## post-repair func. downtime
        if self.stoch_loss_of_function:
            self.initial_loss_of_function_time = get_loss_of_function_time(
                occ_type=self.occtype,
                damage_state_probs=self.damage_state_probs
            )
        else:
            self.initial_loss_of_function_time = self.initial_damage_state * 30
        if self.current_damage_state == 0:
            self.current_loss_of_function_time = 0
        else:
            self.current_loss_of_function_time = self.initial_loss_of_function_time

        ## ---------------------Income/Costs---------------------
        ## income
        if self.stoch_inc_loss:
            self.income_loss = get_income_loss(
                occ_type=self.occtype,
                sqft=self.sqft,
                damage_state_probs=self.damage_state_probs,
                repair_time=self.current_repair_time,
                max_income=self.max_income
            )
        else:
            self.max_income =  1.25 * ((self.sqft / 750) * 60000)
            self.income_loss = 0.8 * self.max_income
        if self.current_damage_state == 0:
            self.current_income = self.max_income
            self.initial_income_loss = 0.0
        else:
            self.current_income = self.max_income - self.income_loss
            self.initial_income_loss = self.income_loss
        # print(f"initial post-quake income: {self.current_income}")
        # print(f"Max income: {self.max_income}")
        ## relocation costs
        if self.stoch_relocation_cost:
            self.initial_relocation_cost = get_relocation_cost(
                occtype=self.occtype,
                sqft=self.sqft,
                damage_state_probs=self.damage_state_probs,
                max_reloc_cost=self.max_reloc_cost
            )
        else:
            self.initial_relocation_cost = self.max_income * 0.25
        if self.current_damage_state == 0:
            self.current_relocation_cost = 0.0
        else:
            self.current_relocation_cost = self.initial_relocation_cost

    def step(
        self,
        action: BuildingAction
    ):
        self.__log(f'Stepping building: {self.id} with action: {action.value}...')
        if action == BuildingAction.DO_NOTHING:
            was_functional = self.is_functional
            try:
                self.__step_functionality()
            except AssertionError as e:
                pass
            if was_functional == self.is_functional:
                functionality_restored = False
            else:
                functionality_restored = True
            self.__log(f'Building {self.id} is doing nothing')
            state = self.current_repair_time
            done = self.is_functional
            info = self.__get_info()
            info["repair_has_finished"] = False
            info["debris_has_cleared"] = False
            info["functionality_has_restored"] = functionality_restored
            return info

        ## ---------- Repair Action ----------
        elif action == BuildingAction.REPAIR:
            had_debris = self.has_debris
            was_repaired = self.is_fully_repaired
            was_functional = self.is_functional

            self.__log('...trying minor repair')
            try:
                self.__step_debris()
            except AssertionError as e:
                try:
                    self.__step_repair()
                except AssertionError as e:
                    try:
                        self.__step_functionality()
                    except AssertionError as e:
                        pass
            # track if debris was cleared in this timestep
            if self.has_debris == had_debris:
                debris_cleared = False
            else:
                debris_cleared = True
            ## track if repair finished in this time step
            if was_repaired == self.is_fully_repaired:
                repair_finished = False
            else:
                repair_finished = True
            ## track if functionality changed

            if was_functional == self.is_functional:
                functionality_restored = False
            else:
                functionality_restored = True

            state = self.current_repair_time
            done = self.is_functional
            info = self.__get_info()
            info["debris_has_cleared"] = debris_cleared
            info["repair_has_finished"] = repair_finished
            info["functionality_has_restored"] = functionality_restored
            self.is_under_repair = False
            return info



        else:
            raise ValueError(f'Invalid action: {action}')

    def __get_info(self):
        info = {
            "damage_state": self.current_damage_state,
            'repair_time': self.current_repair_time,
            'has_debris': self.has_debris,
            'is_fully_repaired': self.is_fully_repaired,
            'is_functional': self.is_functional,
            'income': self.current_income,
            'repair_cost': self.current_structural_repair_cost,
            'loss_of_functionality_time': self.current_loss_of_function_time,
            "relocation_cost": self.current_relocation_cost,
            "debris_has_cleared": None,
            "repair_has_finished": None,
            "functionality_has_restored": None
        }
        return info

    def get_downtime_delay(self, financing_method: str):
        return get_building_downtime_delay(
            damage_state=self.current_damage_state,
            is_essential=self.is_essential,
            num_stories=self.num_stories,
            financing_method=financing_method
        )

    def get_relocation_cost(self):
        if self.is_fully_repaired:
            self.current_relocation_cost = 0.0
        else:
            damage_state_probs = np.zeros(len(self.damage_state_probs))
            damage_state_probs[self.current_damage_state] = 1.0

            self.current_relocation_cost = get_relocation_cost(
                occtype=self.occtype,
                sqft=self.sqft,
                damage_state_probs=damage_state_probs,
                max_reloc_cost=self.max_reloc_cost
            )
        return self.current_relocation_cost

    def get_critical_functionality(self):
        # Define functionality values for each damage state
        damage_to_func = {
            0: 1.0,   # Fully functional
            1: 0.75,   # Partial
            2: 0.3,  # Minimal
            3: 0.1,   # Non-functional
            4: 0.0    # Destroyed
        }

        # Get the functionality for current damage state
        current_critical_func = damage_to_func.get(self.current_damage_state, 0.0) * self.initial_critical_func

        return current_critical_func

        # Non-essential facilities have no critical function
        self.initial_critical_func = 0.0
        self.current_critical_func = 0.0
        return 0.0

    def get_hosp_beds(self):
        # Damage-to-bed availability mapping
        damage_factor = {
            0: 1.0,
            1: 0.75,
            2: 0.3,
            3: 0.1,
            4: 0.0
        }.get(self.current_damage_state)

        current_beds = int(self.initial_beds * damage_factor)

        return current_beds

    def get_doctors(self):
        # Damage-to-bed availability mapping
        damage_factor = {
            0: 1.0,
            1: 0.75,
            2: 0.3,
            3: 0.1,
            4: 0.0
        }.get(self.current_damage_state)

        current_doctors = int(self.initial_doctors * damage_factor)

        return current_doctors

    def __step_debris(self):
        assert not self.is_fully_repaired
        assert self.has_debris

        time_step_duration = self.time_step_duration

        self.current_debris_cleanup_time = max( self.current_debris_cleanup_time - time_step_duration, 0)

        if self.current_debris_cleanup_time == 0:
            self.has_debris = False
            self.debris_capacity_reduction = 0.0

    def __step_damage_state(self):
        steps = self.initial_damage_state

        if steps <= 0:
            return self.initial_damage_state
        days_per_step = self.initial_repair_time / steps
        completed_repair_days = self.initial_repair_time - self.current_repair_time

        levels_repaired = int(completed_repair_days // days_per_step)

        self.current_damage_state = max(self.initial_damage_state - levels_repaired, 0)

    def __step_repair(self):
        assert not self.is_functional
        assert not self.is_fully_repaired
        assert not self.has_debris
        assert self.initial_repair_time > 0

        time_step_duration = self.time_step_duration

        # Calculate the fraction of repair done this step
        repair_fraction = time_step_duration / self.initial_repair_time

        # Reduce current repair time
        self.current_repair_time = max(0, self.current_repair_time - time_step_duration)

        if self.cost_decay == "linear":
            # Reduce repair cost by the same fraction
            cost_reduction = repair_fraction * self.initial_structural_repair_cost
            self.current_structural_repair_cost = max(
                0, self.current_structural_repair_cost - cost_reduction
            )
        else:
            # Compute fraction of repair completed
            elapsed_time = self.initial_repair_time - self.current_repair_time
            fraction_complete = elapsed_time / self.initial_repair_time

            # Quadratic decay (fast drop early, slow later)
            remaining_cost_fraction = (1 - fraction_complete) ** 2
            self.current_structural_repair_cost = self.initial_structural_repair_cost * remaining_cost_fraction

        # If repair is now complete
        if self.current_repair_time == 0:
            self.is_fully_repaired = True
            self.current_damage_state = 0
            self.time_step_after_repair = 0
            self.current_relocation_cost = 0.0
            self.current_structural_repair_cost = 0
        else:
            self.__step_damage_state()
            self.get_relocation_cost()

    def __step_functionality(self):
        assert self.is_fully_repaired
        assert self.current_damage_state == 0
        # assert self.time_step_after_repair > -1
        assert not self.is_functional

        if self.max_income == 0:
            self.is_functional = True
            return
        time_step_duration = self.time_step_duration
        self.time_step_after_repair += 1

        # Decrease remaining loss-of-function time
        self.current_loss_of_function_time = max(
            0, self.current_loss_of_function_time - time_step_duration
        )

        if self.current_loss_of_function_time == 0:
            self.is_functional = True
            self.current_income = self.max_income
            return
        else:
            # Compute fraction of progress
            fraction_recovered = (self.initial_loss_of_function_time - self.current_loss_of_function_time) / self.initial_loss_of_function_time
            if self.income_loss_decay == "quadratic":
                income_fraction = fraction_recovered ** 2
            else:  # default to linear
                income_fraction = fraction_recovered

            target_income = (self.max_income - self.initial_income_loss) + self.initial_income_loss * income_fraction

            # Only increase income, never decrease
            self.current_income = max(self.current_income, target_income)

    def __log(
        self,
        msg
    ) -> None:
        if self.verbose:
            print(msg)

    def __str__(self):
        # damage state names
        damage_states = ['None', 'Slight', 'Moderate', 'Extensive', 'Complete']

        # formatted print string
        return f"""
Building: {self.id}, occupancy: {self.occtype}, total area: {self.sqft} sqft
--------------------
1) Sampled Damage State: {damage_states[self.current_damage_state]}
2) Initial Repair Time: {self.initial_repair_time} days
3) Income Loss: $ {self.income_loss:,}
4) Maximum Income: $ {self.max_income:,}
5) Structural Repair Costs: $ {self.initial_structural_repair_cost:,}
7) Relocation Costs: $ {self.current_relocation_cost:,}
6) Debris Weight: Tons {self.debris_weight:,}
7) Debris Cleanup Time: {self.debris_cleanup_time:,}
        """


def reset_building_objects(
    buildings_study_gdf: gpd.GeoDataFrame,
    building_objs: List[Building]
):
    damage_state_probs = np.array([
        buildings_study_gdf[StudyBuildingSchema.PLS0],
        buildings_study_gdf[StudyBuildingSchema.PLS1],
        buildings_study_gdf[StudyBuildingSchema.PLS2],
        buildings_study_gdf[StudyBuildingSchema.PLS3],
        buildings_study_gdf[StudyBuildingSchema.PLS4]
    ]).T
    for idx, row in buildings_study_gdf.iterrows():
        building_objs[idx].reset(
            damage_state_probs=damage_state_probs[idx],
            debris_capacity_reduction=row[StudyBuildingSchema.CAPACITY_REDUCTION]
        )

    return building_objs

def make_building_objects(
    buildings_study_gdf: gpd.GeoDataFrame,
    time_step_duration: int,
    trucks_per_day: float
) -> List[Building]:
    """
    Create Building objects from a GeoDataFrame of study buildings.

    Args:
        study_building_gdf (gpd.GeoDataFrame): DataFrame with columns found in StudyBuildingSchema
        time_step_duration (int): Duration of each time step

    Returns:
        List[Building]: List of Building objects created from the input DataFrame
    """
    building_objs = []


    for idx, row in buildings_study_gdf.iterrows():

        # print(row["geometry"])
        building_obj = Building(
            id=str(idx),
            geometry=row["geometry"],
            occtype=row[StudyBuildingSchema.OCC_TYPE],
            str_type=row[StudyBuildingSchema.STR_TYP2],
            num_stories=row[StudyBuildingSchema.NO_STORIES],
            sqft=row[StudyBuildingSchema.SQ_FOOT],
            is_essential=row[StudyBuildingSchema.EFACILITY],
            access_road_id=row[StudyBuildingSchema.ACCESS_ROAD_IDX],
            time_step_duration=time_step_duration,
            trucks_per_day=trucks_per_day,
            verbose=False,
            stoch_ds=True,
            calc_debris=True,
            stoch_rt=True,
            stoch_cost=True,
            stoch_inc_loss=True,
            stoch_loss_of_function=True,
            stoch_relocation_cost=True
        )

        building_objs.append(building_obj)

    return building_objs

# def map_buildings_objects(
#     buildings_study_gdf: gpd.GeoDataFrame,
#     building_objs: List[Building]
# ) -> gpd.GeoDataFrame:

#     for building_obj in building_objs:
#         idx = building_obj.id
#         debris_capacity_reduction = building_obj.debris_capacity_reduction
#         damage_state = building_obj.current_damage_state
#         repair_time = building_obj.current_repair_time

#         buildings_study_gdf.loc[idx, StudyBuildingSchema.CAPACITY_REDUCTION] = debris_capacity_reduction
#         buildings_study_gdf.loc[idx, StudyBuildingSchema.DAMAGE_STATE] = damage_state
#         buildings_study_gdf.loc[idx, StudyBuildingSchema.CURR_REPAIR_TIME] = repair_time

#     return buildings_study_gdf
