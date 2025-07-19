import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

import math
from typing import List, Tuple

# --- OpenQuake Imports ---
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.imt import PGA, SA, PGD
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.contexts import ContextMaker
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.geo.point import Point as OQPoint
from openquake.hazardlib.geo.line import Line as OQLine
from openquake.hazardlib.geo.surface import SimpleFaultSurface
from openquake.hazardlib.geo.mesh import RectangularMesh
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.sourceconverter import SourceGroup
from openquake.hazardlib.calc.hazard_curve import calc_hazard_curves

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import norm
from mpl_toolkits.basemap import Basemap

## [ATTENTION] This code is not currently in use as it does not work as intended. The intended use is for probabilistic seismic hazard assessment.
## this was made as a last effort during the latter part of the thesis but certain aspects of it are not yet resolved. If you want to use it, it needs
## to be adapted to work properly. Please ensure you test and validate the code for PSHA upon changing it.

def shp_to_oq_line(linestring: LineString) -> OQLine:
    return OQLine([OQPoint(x, y) for x, y in linestring.coords])

def get_grid_and_map_proj(surf, buf=0.3, delta=0.001):
    """
    Return grid of nodes and map projection specific to surface
    from: https://github.com/GEMScienceTools/notebooks/blob/workspace_1/hazardlib/RuptureDistances.ipynb
    """
    min_lon, max_lon, max_lat, min_lat = surf.get_bounding_box()

    min_lon -= buf
    max_lon += buf
    min_lat -= buf
    max_lat += buf

    lons = np.arange(min_lon, max_lon + delta, delta)
    lats = np.arange(min_lat, max_lat + delta, delta)
    lons, lats = np.meshgrid(lons, lats)
    mesh = RectangularMesh(lons=lons, lats=lats, depths=None)

    m = Basemap(projection='merc', llcrnrlat=np.min(lats), urcrnrlat=np.max(lats),
                llcrnrlon=np.min(lons), urcrnrlon=np.max(lons), resolution='l')

    return mesh, m



class SeismicSourceModel:
    def __init__(
                self,
                source_id: str,
                name: str,
                mfd: EvenlyDiscretizedMFD,
                rupture_mesh_spacing: float,
                rupture_aspect_ratio: float,
                upper_seismogenic_depth: float,
                lower_seismogenic_depth: float,
                fault_trace: LineString,
                dip: float,
                rake: float=0.0,
                tectonic_region_type: str="active shallow crust",
                magnitude_scaling_relationship=WC1994(),
                gmpe: GMPE = BooreEtAl2014()

    ):
        self.source_id = source_id
        self.name = name
        self.tectonic_region_type = tectonic_region_type
        self.mfd = mfd
        self.rupture_mesh_spacing = rupture_mesh_spacing
        self.rupture_aspect_ratio = rupture_aspect_ratio
        self.upper_seismogenic_depth = upper_seismogenic_depth
        self.lower_seismogenic_depth = lower_seismogenic_depth
        self.dip = dip
        self.rake = rake
        self.fault_trace = OQLine([OQPoint(x, y) for x, y in fault_trace.coords])
        self.magnitude_scaling_relationship = magnitude_scaling_relationship
        self.gmpe = gmpe
        self.min_mag = mfd.min_mag

    def build_source(self,min_magnitude: float, return_period: np.int32=50):
        try:
            self.mfd.modify_mag_range(new_min_mag=min_magnitude, new_max_mag=min_magnitude + 0.1)
            self.min_mag = min_magnitude
        except ValueError as e:
            pass
        self.src = SimpleFaultSource(
                                    source_id=self.source_id,
                                    name=f"Simple fault source ID {self.source_id}",
                                    tectonic_region_type=self.tectonic_region_type,
                                    mfd=self.mfd,
                                    rupture_mesh_spacing=2.0,
                                    magnitude_scaling_relationship=self.magnitude_scaling_relationship,
                                    rupture_aspect_ratio=1.0,
                                    upper_seismogenic_depth=self.upper_seismogenic_depth,
                                    lower_seismogenic_depth=self.lower_seismogenic_depth,
                                    fault_trace=self.fault_trace,
                                    dip=self.dip,
                                    rake=self.rake,
                                    temporal_occurrence_model=PoissonTOM(return_period)
        )
        self.surface = SimpleFaultSurface.from_fault_data(self.fault_trace,
                                                        self.upper_seismogenic_depth,
                                                        self.lower_seismogenic_depth,
                                                        self.dip,
                                                        self.rupture_mesh_spacing)


class SeismicSourceZone:
    def __init__(self,
                ucerf3: gpd.GeoDataFrame,
                infrastructure_study_sites: List[Point],
                event_rate_cols_prefix: str = 'M',
                gmpe: GMPE = BooreEtAl2014()
    ):
        self.ucerf3 = ucerf3
        self.infrastructure_pts = infrastructure_study_sites
        self.__event_rate_cols_prefix = event_rate_cols_prefix
        self.__gmpe = gmpe
        self.__preprocess_inputs()
        self.source_models = self.__build_sources()
        self.sites = self.__build_sites()
        self.imtls = {"PGA": np.logspace(np.log10(0.005), np.log10(3.0), 50),
                      "SA(0.3)": np.logspace(np.log10(0.005), np.log10(3.0), 50),
                      "SA(1.0)": np.logspace(np.log10(0.005), np.log10(3.0), 50)
        }

    def get_max_return_period(self, min_return_period=475, step=1000, max_search=1_000_000):
        """
        Find the maximum return period starting from min_return_period up to max_search
        where at least one source has non-zero occurrence rates at or above the corresponding magnitude.

        Parameters:
        - min_return_period: starting return period (default 475 years)
        - step: increment step in years (default 1000)
        - max_search: upper limit for return period search (default 1,000,000 years)

        Returns:
        - max_return_period: maximum valid return period found
        """

        rp = min_return_period
        last_valid_rp = None

        while rp <= max_search:
            annual_exceedance_prob = 1.0 / rp
            eligible_ids = []

            for idx, source in enumerate(self.source_models):
                # Check if the source has any magnitude bin with occurrence rate > annual_exceedance_prob
                annual_rates = source.mfd.get_annual_occurrence_rates()
                if any(rate > 0 and rate >= annual_exceedance_prob for mag, rate in annual_rates):
                    eligible_ids.append(idx)

            if eligible_ids:
                last_valid_rp = rp
                rp += step
            else:
                # No eligible sources at this return period or beyond, stop searching
                break

        if last_valid_rp is None:
            raise ValueError("No valid return period found where sources have non-zero occurrence rates.")

        return last_valid_rp

    def __preprocess_inputs(self):
        gdf = self.ucerf3
        self.mag_cols = sorted([col for col in gdf.columns if col.startswith(self.__event_rate_cols_prefix)])

        if not self.mag_cols:
            raise ValueError(f"No event rate columns found with prefix '{self.__event_rate_cols_prefix}'.")

        self.mags_bins = sorted([float(col.replace(self.__event_rate_cols_prefix, '')) for col in self.mag_cols])
        self.mag_bin_width = round(self.mags_bins[1] - self.mags_bins[0], 5)

        # --- NEW: Spatial filtering of sources ---
        # Compute average point from infrastructure
        infra_coords = np.array([[pt.x, pt.y] for pt in self.infrastructure_pts])
        avg_x, avg_y = infra_coords.mean(axis=0)
        centroid = Point(avg_x, avg_y)

        # Create GeoSeries of the average location
        centroid_gdf = gpd.GeoSeries([centroid], crs=gdf.crs)

        # Project to a metric CRS (e.g., UTM) for distance calculation
        if gdf.crs is None or gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=3857)  # Web Mercator (meters)
            centroid_gdf = centroid_gdf.to_crs(epsg=3857)

        # Compute distance from centroid to each fault geometry
        distances = gdf.geometry.distance(centroid_gdf.iloc[0])  # in meters

        # Filter sources within 200 km (200,000 meters)
        self.ucerf3 = gdf[(distances >= 1_000) & (distances <= 199_000)].copy().to_crs(epsg=4326)

    def __build_sources(self)-> List[SeismicSourceModel]:
        sources = []
        for _, row in self.ucerf3.iterrows():
            occurrence_rates = row[self.mag_cols].values.astype(float).tolist()
            mfd = EvenlyDiscretizedMFD(min_mag=5.0, bin_width=self.mag_bin_width, occurrence_rates=occurrence_rates)
            source = SeismicSourceModel(
                source_id=str(row["ID"]),
                name=f"Simple fault source ID {row['ID']}",
                mfd=mfd,
                rupture_mesh_spacing=2.0,
                rupture_aspect_ratio=1.0,
                upper_seismogenic_depth=float(row['Upper Seis Depth']),
                lower_seismogenic_depth=float(row['Lower Seis Depth']),
                fault_trace=row['geometry'],
                dip=float(row['Ave Dip'])
            )
            sources.append(source)

        return sources

    def select_seismic_source(self, return_period: np.int32) -> SeismicSourceModel:
        eligible_sources = self._get_valid_srcs(return_period)

        if not eligible_sources:
            raise ValueError(f"No sources found meeting criteria: return_period={return_period}")

        selected_index = np.random.choice(len(eligible_sources))
        selected_id, min_mag = eligible_sources[selected_index]
        source_model = self.source_models[selected_id]
        source_model.build_source(return_period=return_period, min_magnitude=min_mag)
        return source_model

    def _get_valid_srcs(self, rt: float) -> List[Tuple[int, float]]:
        annual_exceedance_prob = 1.0 / rt
        eligible = []
        # print(rt)
        for source_idx, source_model in enumerate(self.source_models):
            rates = [
                (magnitude, rate) for magnitude, rate in source_model.mfd.get_annual_occurrence_rates()
                if 0 < rate <= annual_exceedance_prob
            ]
            # print(f"Source {source_idx} rates: {rates}")

            if rates:
                # Pick the magnitude whose rate is closest to AEP
                best_magnitude, _ = min(rates, key=lambda x: abs(x[1] - annual_exceedance_prob))
                eligible.append((source_idx, best_magnitude))
        # print(eligible)

        return eligible

    def _is_source_valid(
        self,
        source_model: SeismicSourceModel,
        annual_exceedance_prob: float
    ) -> bool:
        """
        Check if a source model has a magnitude with annual rate close to AEP.
        """
        rates = [
            (magnitude, rate) for magnitude, rate in source_model.mfd.get_annual_occurrence_rates()
            if 0 < rate <= annual_exceedance_prob
        ]

        if not rates:
            return False

        self.min_mag, _ = min(rates, key=lambda x: abs(x[1] - annual_exceedance_prob))
        return True

    def __build_sites(self, vs30_values: List[float]=None):
        if vs30_values is None:
            vs30_values = np.random.uniform(180, 270, len(self.infrastructure_pts)) ## this is called in __init__() so vs30 is constant across experiments

        sites_oq = [Site(location=OQPoint(pt.x, pt.y), vs30=vs30)
                    for pt, vs30 in zip(self.infrastructure_pts, vs30_values)]
        sites_collection = SiteCollection(sites_oq)

        return sites_collection


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


    def attenuate(self, src:  SeismicSourceModel):
        means = []
        mags = []
        imts = [PGA(), SA(0.3), SA(1.0)]
        src_grp = SourceGroup("active shallow crust", [src.src])
        param = dict(
                    imtls=self.imtls,
                    truncation_level=99,
                    cluster=src_grp.cluster
        )

        ctx_maker = ContextMaker(src.tectonic_region_type, [src.gmpe], param)
        # print("Checking source:", src.name)
        # print("Fault trace coords:", [(p.longitude, p.latitude) for p in src.fault_trace.points])
        # print("Surface bounding box:", src.surface.get_bounding_box())
        # print("Number of sites:", len(self.sites))

        ctxs = ctx_maker.from_srcs([src.src], self.sites)
        # print(ctxs)

        gms = ctx_maker.get_mean_stds(ctxs)

        mean = gms[0, 0]  # G=1 assumed
        std_total = gms[1, 0]

        # Sample 1 realization per IMT/site
        sample = np.exp(mean)

        # Split per IMT
        pga_samples = sample[0]
        sa03_samples = sample[1]
        sa10_samples = sample[2]

        # Total number of infrastructure points
        S = len(self.infrastructure_pts)

        # Reshape each IMT sample array into shape (S, K)
        K = pga_samples.shape[0] // S
        pga_split = pga_samples.reshape(S, K)
        sa03_split = sa03_samples.reshape(S, K)
        sa10_split = sa10_samples.reshape(S, K)

        # Stack per site
        sitewise_samples = [
            [pga_split[i], sa03_split[i], sa10_split[i]]
            for i in range(S)
        ]

        # Optional: convert to np.array if shape is uniform
        sitewise_array = np.array(sitewise_samples)  # shape: (S, 3, K)
        avg_gms = sitewise_array.mean(axis=-1)  # shape: (S  , 3)
        s = avg_gms.shape[0]

        extended_avg_gms = np.zeros((avg_gms.shape[0], 4))
        extended_avg_gms[:, :3] = avg_gms
        M = src.min_mag

        for i in range(extended_avg_gms.shape[0]):
            pga_threshold = self.threshold_PGA_liquefaction(extended_avg_gms[i, 0], M)
            k_delta = (0.0086 * M**3) - (0.0914 * (M**2)) + (0.4698 * M) - 0.9835
            pga_ratio = extended_avg_gms[i, 0] / pga_threshold

            extended_avg_gms[i, 3] = max(0.1, self.lateral_spreading(pga_ratio) * k_delta)

        # print(f"Average GMS: {extended_avg_gms}")
        return extended_avg_gms
