import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely import shortest_line, affinity, wkt
from pyproj import CRS, Transformer, transform
from typing import Dict

from .road_config import *
from .building_config import *
from .road_funcs import *
from .building_funcs import *

def road_to_building(buildings_gdf, roads_gdf):
    """
    Adds a column 'access_road_index' to buildings_gdf that maps the index of the closest road,
    considering the closest point along the entire road geometry.

    Parameters:
    - buildings_gdf: GeoDataFrame containing building centroids in the 'geom' column.
    - roads_gdf: GeoDataFrame containing road geometries.

    Returns:
    - buildings_gdf with the added 'access_road_index' column.
    """
    # Ensure the CRS is the same
    if buildings_gdf.crs != roads_gdf.crs:
        raise ValueError("The CRS of the two GeoDataFrames must be the same.")

    # Initialize a list to store the closest road indices
    closest_road_indices = []

    # Iterate over each building
    for building in buildings_gdf[StudyBuildingSchema.GEOM]:
        min_distance = float('inf')  # Initialize with a large value
        closest_road_index = None

        # Iterate over each road to find the closest one
        for idx, road_row in roads_gdf.iterrows():
            # Calculate the shortest line between the building and the road (using GeoPandas distance method)
            road = road_row['geometry']
            distance = building.distance(road)  # GeoPandas handles the distance calculation

            if distance < min_distance:
                min_distance = distance
                closest_road_index = roads_gdf.loc[idx, StudyRoadSchema.LINKNWID]

        # Append the closest road index to the list
        closest_road_indices.append(closest_road_index)

    # Add the closest road indices as a new column to buildings_gdf
    buildings_gdf[StudyBuildingSchema.ACCESS_ROAD_IDX] = closest_road_indices

    return buildings_gdf


def calculate_min_distances_to_roads(
    road_lines: Dict[int, LineString],
    rectangles: Dict[int, Polygon],
    road_width_meters: float
) -> Dict[int, float]:
    """
    Calculate minimum distances from rectangles to road offset lines.

    Args:
        road_lines: Dict of road centerlines {id: LineString} in EPSG:4326
        rectangles: Dict of building footprints {id: Polygon} in EPSG:4326
        road_width_meters: Road width in meters

    Returns:
        Dict mapping rectangle IDs to their minimum distances to road offsets
    """

    def _calculate_min_distance_to_nonoverlap_offset(rectangle, center_line, total_width):
        """
        Calculate minimum distance to non-overlapping offset line using center line and road width.

        Args:
            rectangle: Shapely Polygon representing the building footprint
            center_line: Shapely LineString representing road center line
            total_width: Float total width of the road

        Returns:
            dict: Dictionary containing:
                - 'overlapped_line': String indicating which line is overlapped ('north', 'south', or 'both')
                - 'min_distance': Float representing minimum distance to non-overlapping line (if applicable)
                - 'target_line': String indicating which line the distance was calculated to
            None: If rectangle doesn't overlap any lines
        """
        # Create offset lines (half width on each side)
        half_width = total_width / 2
        try:
            north_offset = center_line.parallel_offset(half_width, 'left')
            south_offset = center_line.parallel_offset(half_width, 'right')
        except ValueError as e:
            print(f"Warning: Error creating offset lines: {e}")
            return None

        # Check which lines the rectangle overlaps
        overlaps_north = rectangle.intersects(north_offset)
        overlaps_south = rectangle.intersects(south_offset)

        # If no overlap with any line, return None
        if not overlaps_north and not overlaps_south:
            return None

        # Initialize result dictionary
        result = {
            'overlapped_line': None,
            'min_distance': None,
            'target_line': None
        }

        # Handle overlap scenarios
        if overlaps_north and overlaps_south:
            result['overlapped_line'] = 'both'
        elif overlaps_north:
            result['overlapped_line'] = 'north'
            result['min_distance'] = _min_distance_to_line(rectangle, south_offset)
            result['target_line'] = 'south'
        else:  # overlaps_south
            result['overlapped_line'] = 'south'
            result['min_distance'] = _min_distance_to_line(rectangle, north_offset)
            result['target_line'] = 'north'

        return result

    def _min_distance_to_line(rectangle, line):
        """
        Helper function to calculate minimum distance between a rectangle and a line.

        Args:
            rectangle: Shapely Polygon
            line: Shapely LineString

        Returns:
            float: Minimum distance
        """
        min_distance = float('inf')

        # Check distance from rectangle vertices to line
        for vertex in list(rectangle.exterior.coords):
            point = Point(vertex)
            distance = point.distance(line)
            min_distance = min(min_distance, distance)

        # Check distance from line vertices to rectangle
        for coord in line.coords:
            point = Point(coord)
            distance = point.distance(rectangle)
            min_distance = min(min_distance, distance)

        return min_distance

    # Setup projections
    wgs84 = CRS('EPSG:4326')
    # Get UTM zone for center of area
    center_lat = next(iter(rectangles.values())).centroid.y
    center_lon = next(iter(rectangles.values())).centroid.x
    utm_zone = int((center_lon + 180) / 6) + 1
    utm_crs = CRS(f'EPSG:326{utm_zone if center_lat >= 0 else utm_zone+30}')

    # Create transformer
    transformer = Transformer.from_crs(wgs84, utm_crs, always_xy=True)

    # Project geometries
    road_lines_utm = {
        id: transform(transformer.transform, line)
        for id, line in road_lines.items()
    }
    rectangles_utm = {
        id: transform(transformer.transform, rect)
        for id, rect in rectangles.items()
    }

    # Calculate minimum distances
    min_distances = {}
    capacity_ratios =  {}
    for rect_id, rect in rectangles_utm.items():
        min_dist = float('inf')
        for road in road_lines_utm.values():
            result = _calculate_min_distance_to_nonoverlap_offset(
                rect, road, road_width_meters
            )
            if result and result['min_distance'] is not None:
                min_dist = min(min_dist, result['min_distance'])
        min_distances[rect_id] = min_dist if min_dist != float('inf') else None
        capacity_ratios[rect_id] = (min_dist/road_width_meters) if min_dist != float('inf') else None

    return min_distances, capacity_ratios

def update_capacities(
    roads_study_gdf: gpd.GeoDataFrame,
    buildings_study_gdf: gpd.GeoDataFrame,
    recalculate: bool=True
) -> gpd.GeoDataFrame:

    def get_debris_capacity_reduction(
        study_buildings_gdf,
        study_roads_gdf,
        study_road_idx,
        recalculate = False,
        plot = False,
        verbose = False
    ):
        road_idx = study_road_idx
        # If we're not recalculating and the data already exists in the GeoDataFrame, use it
        if not recalculate:
            # Filter buildings connected to the given road
            focus_buildings = study_buildings_gdf[study_buildings_gdf[StudyBuildingSchema.ACCESS_ROAD_IDX] == road_idx]

            # If capacity_reduction exists for these buildings, use it directly
            if StudyBuildingSchema.CAPACITY_REDUCTION in focus_buildings.columns and not focus_buildings[StudyBuildingSchema.CAPACITY_REDUCTION].isna().all():
                # Create dictionary of building ID to capacity reduction
                all_capacity_reductions_dict = {
                    idx: reduction for idx, reduction in
                    zip(focus_buildings.index, focus_buildings[StudyBuildingSchema.CAPACITY_REDUCTION])
                }

                # Get the maximum capacity reduction for this road
                max_capacity_reduction = focus_buildings[StudyBuildingSchema.CAPACITY_REDUCTION].max()

                return round(max_capacity_reduction, 3), all_capacity_reductions_dict

        def _calculate_min_distance_to_nonoverlap_offset(rectangle, center_line, total_width):
            """
            Calculate minimum distance to non-overlapping offset line using center line and road width.

            Args:
                rectangle: Shapely Polygon representing the building footprint
                center_line: Shapely LineString representing road center line
                total_width: Float total width of the road

            Returns:
                dict: Dictionary containing:
                    - 'overlapped_line': String indicating which line is overlapped ('north', 'south', or 'both')
                    - 'min_distance': Float representing minimum distance to non-overlapping line (if applicable)
                    - 'target_line': String indicating which line the distance was calculated to
                None: If rectangle doesn't overlap any lines
            """
            def _min_distance_to_line(polygon, linestring):
                """
                Calculate the smallest distance between a Shapely Polygon and a Shapely LineString.

                Parameters:
                polygon (shapely.geometry.Polygon): The polygon.
                linestring (shapely.geometry.LineString): The linestring.

                Returns:
                float: The smallest distance between the polygon and the linestring.
                """
                # Ensure the inputs are of the correct types
                if not isinstance(polygon, Polygon):
                    raise TypeError("The first argument must be a Shapely Polygon.")
                if not isinstance(linestring, LineString):
                    raise TypeError("The second argument must be a Shapely LineString.")

                # Calculate the smallest distance between the polygon and the linestring
                distance = polygon.distance(linestring)

                return distance

            # Create offset lines (half width on each side)
            half_width = total_width / 2
            try:
                north_offset = center_line.parallel_offset(half_width, 'left')
                south_offset = center_line.parallel_offset(half_width, 'right')
            except ValueError as e:
                print(f"Warning: Error creating offset lines: {e}")
                return None

            # Check which lines the rectangle overlaps
            overlaps_north = rectangle.intersects(north_offset)
            overlaps_south = rectangle.intersects(south_offset)

            # If no overlap with any line, return None
            if not overlaps_north and not overlaps_south:
                return None

            # Initialize result dictionary
            result = {
                'overlapped_line': None,
                'min_distance': None,
                'target_line': None
            }

            # Handle overlap scenarios
            if overlaps_north and overlaps_south:
                result['overlapped_line'] = 'both'
            elif overlaps_north:
                result['overlapped_line'] = 'north'
                result['min_distance'] = _min_distance_to_line(rectangle, south_offset)
                result['target_line'] = 'south'
            else:  # overlaps_south
                result['overlapped_line'] = 'south'
                result['min_distance'] = _min_distance_to_line(rectangle, north_offset)
                result['target_line'] = 'north'

            return result

        def _extract_transformed_geometries(focus_buildings_gdf, focus_debris_gdf, focus_road_gdf):
            """
            Transforms and extracts geometries from given GeoDataFrames, setting the origin at (minX, minY).

            Args:
                focus_buildings_gdf (GeoDataFrame): GeoDataFrame of building geometries.
                focus_debris_gdf (GeoDataFrame): GeoDataFrame of debris geometries.
                focus_road_gdf (GeoDataFrame): GeoDataFrame of road geometries.

            Returns:
                Tuple (list, list, list): Transformed geometries for buildings, debris, and roads.
            """
            # Define EPSG:4326 (original CRS)
            wgs84 = CRS("EPSG:4326")

            # Extract all geometries from all GDFs
            all_geometries = (
                list(focus_buildings_gdf.geometry) +
                list(focus_debris_gdf.geometry) +
                list(focus_road_gdf.geometry)
            )

            all_indices = (
                list(focus_buildings_gdf.index) +
                list(focus_debris_gdf.index) +
                list(focus_road_gdf.index)
            )

            # Get bounding box (min X, min Y)
            min_x = min(geom.bounds[0] for geom in all_geometries)
            min_y = min(geom.bounds[1] for geom in all_geometries)

            # Define a transformer (convert EPSG:4326 → local meters with min_x, min_y as origin)
            transformer = Transformer.from_crs(wgs84, CRS("EPSG:3857"), always_xy=True)

            def transform_geometry(geom):
                """Transform geometry to meters relative to (min_x, min_y)."""
                projected_geom = transform(transformer.transform, geom)  # Convert to meters
                return affinity.translate(projected_geom, -min_x, -min_y)  # Shift origin to (0,0)

            # Transform geometries and create dictionaries with indices as keys
            transformed_buildings = {
                idx: transform_geometry(geom)
                for idx, geom in zip(focus_buildings_gdf.index, focus_buildings_gdf.geometry)
            }

            transformed_debris = {
                idx: transform_geometry(geom)
                for idx, geom in zip(focus_debris_gdf.index, focus_debris_gdf.geometry)
            }

            transformed_roads = {
                idx: transform_geometry(geom)
                for idx, geom in zip(focus_road_gdf.index, focus_road_gdf.geometry)
            }

            return transformed_buildings, transformed_debris, transformed_roads

        # Filter buildings connected to the given road
        focus_buildings = study_buildings_gdf[study_buildings_gdf[StudyBuildingSchema.ACCESS_ROAD_IDX] == road_idx]

        # Create GeoDataFrames
        focus_buildings_gdf = gpd.GeoDataFrame(geometry=focus_buildings.geometry, crs=study_buildings_gdf.crs)
        # focus_buildings.loc[:, StudyBuildingSchema.DEBRIS_GEOM] = focus_buildings[StudyBuildingSchema.DEBRIS_GEOM].apply(wkt.loads)
        # Filter focus_buildings to include only 'Extensive' or 'Complete' damage states
        focus_buildings = focus_buildings[focus_buildings[StudyBuildingSchema.DAMAGE_STATE].isin(['Extensive', 'Complete'])]
        focus_buildings.loc[:, StudyBuildingSchema.DEBRIS_GEOM] = focus_buildings[StudyBuildingSchema.DEBRIS_GEOM].apply(
            lambda x: wkt.loads(x) if isinstance(x, str) else x if isinstance(x, Polygon) else None
        )
        focus_debris_gdf = gpd.GeoDataFrame(geometry=focus_buildings[StudyBuildingSchema.DEBRIS_GEOM], crs=study_buildings_gdf.crs)

        focus_road_gdf = gpd.GeoDataFrame([study_roads_gdf.loc[road_idx]], geometry='geometry', crs=study_roads_gdf.crs)

        # rectangles, debris_rectangles, center_line = _extract_transformed_geometries(focus_buildings_gdf, focus_debris_gdf, focus_road_gdf)
        rectangles_dict, debris_rectangles_dict, center_line_dict = _extract_transformed_geometries(focus_buildings_gdf, focus_debris_gdf, focus_road_gdf)

        rectangles = list(rectangles_dict.values())
        debris_rectangles = list(debris_rectangles_dict.values())
        center_line = list(center_line_dict.values())[0]

        rectangles_ids = list(rectangles_dict.keys())
        debris_rectangles_ids = list(debris_rectangles_dict.keys())
        centre_line_ids = list(center_line_dict.keys())

        # Example usage with previously created rectangles
        # Assuming center_line is already defined
        road_width = focus_road_gdf[StudyRoadSchema.WIDTH].tolist()[0]
        max_capacity_reduction = 0.0

        # Initialize capacity reductions dictionary with all debris rectangles IDs set to 0.0
        all_capacity_reductions_dict = {rect_id: 0.0 for rect_id in debris_rectangles_ids}

        if verbose:
            print("Analyzing distances for overlapping rectangles only:")

        for _, (rect_id, rect) in enumerate(debris_rectangles_dict.items(), 1):
            result = _calculate_min_distance_to_nonoverlap_offset(rect, center_line, road_width)

            if result is not None:  # Only process rectangles that overlap at least one line
                if verbose:
                    print(f"\nRectangle {rect_id}:")
                    print(f"Overlaps: {result['overlapped_line']} offset")

                if result['min_distance'] is not None:
                    if verbose:
                        print(f"Minimum distance to {result['target_line']} offset: {result['min_distance']:.2f} units")
                    distance_to_non_overlap_edge = result['min_distance']
                    width_reduction = road_width - distance_to_non_overlap_edge
                    capacity_reduction = width_reduction / road_width
                    all_capacity_reductions_dict[rect_id] = round(capacity_reduction, 3)
                    max_capacity_reduction = max(max_capacity_reduction, capacity_reduction)
                else:
                    if verbose:
                        print("Distance calculation not applicable (overlaps both lines)")
                    all_capacity_reductions_dict[rect_id] = 1.0
                    max_capacity_reduction = 1.0

        def visualize_road_lines(
                debris_rectangles,
                road_width,
                center_line
        ):
            half_width = road_width / 2
            north_offset = center_line.parallel_offset(half_width, 'left')
            south_offset = center_line.parallel_offset(half_width, 'right')

            fig, ax = plt.subplots(figsize=(10,10))

            # Plot lines
            ax.plot(*center_line.xy, 'r-', linewidth=2, label="Center Line")
            ax.plot(*north_offset.xy, 'k--', linewidth=1, label="North Offset")
            ax.plot(*south_offset.xy, 'k--', linewidth=1, label="South Offset")

            # Plot rectangles with indices
            for i, rect in enumerate(debris_rectangles, 1):
                x, y = rect.exterior.xy
                xa, ya = rectangles[i-1].exterior.xy
                ax.fill(x, y, alpha=0.5, edgecolor='black')
                ax.fill(xa,ya)

                # Calculate centroid for text placement
                centroid = rect.centroid
                ax.text(centroid.x, centroid.y, str(i),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12,
                        fontweight='bold',
                        color='red')

            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            plt.title("Road Lines and Rectangle Placement")

            # Adjust plot limits to ensure all rectangles and their indices are visible
            margin = 0.5  # Add some margin around the plot
            minx,miny,maxx,maxy = center_line.bounds
            x_coords = []
            y_coords = []
            x_coords.extend([minx, maxx])
            y_coords.extend([miny, maxy])

            for rect in debris_rectangles:
                x, y = rect.exterior.xy
                x_coords.extend(x)
                y_coords.extend(y)

            ax.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
            ax.set_ylim(min(y_coords) - margin, max(y_coords) + margin)

            plt.show()

        if plot:
            visualize_road_lines(debris_rectangles=debris_rectangles, road_width=road_width, center_line=center_line)

        return round(max_capacity_reduction,3), all_capacity_reductions_dict

    if StudyBuildingSchema.CAPACITY_REDUCTION not in buildings_study_gdf.columns:
        buildings_study_gdf[StudyBuildingSchema.CAPACITY_REDUCTION] = 0.000
    elif StudyRoadSchema.CAPACITY_RED_DS not in roads_study_gdf.columns:
        roads_study_gdf[StudyRoadSchema.CAPACITY_RED_DS] = 0.0
    elif StudyRoadSchema.CAPACITY_RED_DEBRIS not in roads_study_gdf.columns:
        roads_study_gdf[StudyRoadSchema.CAPACITY_RED_DEBRIS] = 0.0

    # # Iterate through each road in roads_study_gdf
    for idx, row in roads_study_gdf.iterrows():
        damage_state = row[StudyRoadSchema.DAMAGE_STATE]
        if row[StudyRoadSchema.HAZUS_BRIDGE_CLASS] != 'None':
            capacity_reduction_ds = get_bridge_capacity_reduction(damage_state=damage_state)
        else:
            capacity_reduction_ds = get_road_capacity_reduction(damage_state=damage_state)

        max_capacity_reduction_debris, capacity_reduction_debris_dict = get_debris_capacity_reduction(
            study_buildings_gdf=buildings_study_gdf,
            study_roads_gdf=roads_study_gdf,
            study_road_idx=idx,
            recalculate=recalculate
        )
        # print(capacity_reduction_ds)
        # print(capacity_reduction_debris)
        roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_RED_DS] = capacity_reduction_ds
        roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_RED_DEBRIS] = max_capacity_reduction_debris

        # Update buildings with capacity reduction values from dictionary
        for building_id, capacity_reduction in capacity_reduction_debris_dict.items():
            buildings_study_gdf.loc[building_id, StudyBuildingSchema.CAPACITY_REDUCTION] = capacity_reduction

    return buildings_study_gdf

def map_capacity_reduction_debris(
    buildings: List[Building],
    roads: List[Road],
):
    """
    Map the max capacity reduction of each road from all the building debris on it.

    Args:
        buildings (List[Building]): List of Building objects.
        roads (List[Road]): List of Road objects.

    Returns:
        Tuple[List[Building], List[Road]]: Updated lists of Building and Road objects.
    """
    # Iterate through each road in roads
    for road in roads:
        accessing_buildings = [building for building in buildings if building.access_road_id == road.id]
        debris_capacity_reductions = [building.debris_capacity_reduction for building in accessing_buildings]
        road.capacity_red_debris = max(debris_capacity_reductions, default=0.0)
        road.capacity_reduction = max(road.capacity_red_damage_state, road.capacity_red_debris)

    return buildings, roads


