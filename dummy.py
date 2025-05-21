import geopandas as gpd
from shapely.geometry import LineString

roads_gdf = gpd.GeoDataFrame({
    'geometry': [LineString([(0,0), (1,1)]), LineString([(1,1), (1,0)]), LineString([(0,1),(1,1)])],
    'hazus_road_class': ['HRD2', 'HRD1', 'HRD2'],
    'hazus_bridge_class': [None, None, 'HWB1'],
}, crs="EPSG:4326")

print(roads_gdf)
