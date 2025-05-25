import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

import math
from typing import List

# --- OpenQuake Imports ---
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.source import PointSource, SimpleFaultSource
from openquake.hazardlib.scalerel import WC1994
from openquake.hazardlib.imt import PGA, SA
from openquake.hazardlib.site import Site, SiteCollection
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
from openquake.hazardlib.geo.nodalplane import NodalPlane
from openquake.hazardlib.geo.point import Point as OQPoint
from openquake.hazardlib.geo.line import Line as OQLine
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.calc.hazard_curve import calc_hazard_curves

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import norm


def shp_to_oq_line(linestring: LineString) -> OQLine:
    return OQLine([OQPoint(x, y) for x, y in linestring.coords])


class PSHA:
    class _DistanceContainer:
        """A simple container class to hold distance attributes for the GMPE."""
        pass

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

        self.__seismic_source_models = self.__make_source_model_objs()

        self.default_imts = [PGA(), SA(0.3), SA(1.0)]
        self.default_imts_str = ["PGA", "SA(0.3)", "SA(1.0)"]
        self.default_im_levels = {imt: np.logspace(np.log10(0.005), np.log10(3.0), 50) for imt in self.default_imts}
        self.imtls = {imt: np.logspace(np.log10(0.005), np.log10(3.0), 50) for imt in self.default_imts_str}
        self.imls = [self.imtls[imt] for imt in self.default_imts_str]

    def __make_source_model_objs(self) -> List[SimpleFaultSource]:
        gdf = self.ucerf3
        mag_cols = sorted([col for col in gdf.columns if col.startswith(self.__event_rate_cols_prefix)])

        if not mag_cols:
            raise ValueError(f"No event rate columns found with prefix '{self.__event_rate_cols_prefix}'.")

        mags = sorted([float(col.replace(self.__event_rate_cols_prefix, '')) for col in mag_cols])
        bin_width = round(mags[1] - mags[0], 5) if len(mags) > 1 else 0.1

        seismic_source_models = []
        has_rake_col = 'Rake' in gdf.columns or 'Ave Rake' in gdf.columns
        rake_col_name = 'Rake' if 'Rake' in gdf.columns else 'Ave Rake'

        for _, row in gdf.iterrows():
            occurrence_rates = row[mag_cols].values.astype(float).tolist()
            if sum(occurrence_rates) == 0:
                continue

            mfd = EvenlyDiscretizedMFD(min_mag=mags[0], bin_width=bin_width, occurrence_rates=occurrence_rates)
            upper_depth, lower_depth, dip = float(row['Upper Seis Depth']), float(row['Lower Seis Depth']), float(row['Ave Dip'])
            strike = self.get_strike_from_linestring(row['geometry'])
            rake = float(row[rake_col_name]) if has_rake_col else 0.0


            source = SimpleFaultSource(
                source_id=str(row["ID"]),
                name=f"Simple fault source ID {row['ID']}",
                tectonic_region_type="active shallow crust",
                mfd=mfd,
                rupture_mesh_spacing=2.0,
                magnitude_scaling_relationship=WC1994(), rupture_aspect_ratio=1.0,
                upper_seismogenic_depth=upper_depth, lower_seismogenic_depth=lower_depth,
                fault_trace=shp_to_oq_line(row['geometry']),
                dip=dip,
                rake=rake,
                temporal_occurrence_model=PoissonTOM(50.)
            )

            seismic_source_models.append(source)

        return seismic_source_models

    @property
    def source_models(self) -> List[SimpleFaultSource]:
        return self.__seismic_source_models

    def run_psha_curves(self,
                        max_dist_km: float = 250.0,
                        imts: List[GMPE] = None,
                        im_levels: dict = None,
                        site_vs30_mapping: dict = None
                        ) -> dict:
        if imts is None: imts = self.default_imts
        if im_levels is None: im_levels = self.default_im_levels

        for imt in imts:
            if imt not in im_levels:
                raise ValueError(f"IMT {imt} is in 'imts' but not defined in 'im_levels'.")

        if site_vs30_mapping:
            vs30_values = [site_vs30_mapping.get(i) for i in range(len(self.infrastructure_pts))]
            if None in vs30_values:
                    raise ValueError("`site_vs30_mapping` is missing values for some sites.")
        else:
            vs30_values = np.random.uniform(180, 760, len(self.infrastructure_pts))

        sites_oq = [Site(location=OQPoint(pt.x, pt.y), vs30=vs30)
                    for pt, vs30 in zip(self.infrastructure_pts, vs30_values)]
        sites_collection = SiteCollection(sites_oq)
        hazard_curves_output = calc_hazard_curves(
            groups=self.source_models,
            srcfilter=sites_collection,
            gsim_by_trt={"active shallow crust": self.__gmpe},
            imtls=self.imtls
        )
        return hazard_curves_output

    @staticmethod
    def get_strike_from_linestring(linestring: LineString) -> float:
        coords = list(linestring.coords)
        if len(coords) < 2: return 0.0

        lon1, lat1 = map(math.radians, coords[0])
        lon2, lat2 = map(math.radians, coords[-1])
        dlon = lon2 - lon1

        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        azimuth = math.degrees(math.atan2(x, y))
        return (azimuth + 360) % 360

