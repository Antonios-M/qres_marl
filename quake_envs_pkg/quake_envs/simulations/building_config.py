from dataclasses import dataclass, field, fields
from typing import Final, List, Dict, Callable, Tuple, Optional
import random
from enum import Enum
import math
import numpy as np
import sys
import os
from . import utils

# Constants for essential facility occupation types
ESSENTIAL_FACILITY_OCC_TYPES: Final[List[str]] = ['GOV2', 'COM6']

@dataclass(frozen=True)
class DwellingUnitCalculator:
    populations: List[int]  # List of population values (e.g., [pop2amu65, pop2amo65, pop2pmu65, pop2pmo65])
    avg_dwell_size: int     # Average dwelling size

    def calculate_dwell_units(self) -> int:
        """
        Calculate the number of dwelling units based on the provided population data and average dwelling size.

        Returns:
            int: The calculated number of dwelling units.
        """
        total_population = sum(self.populations)
        dwelling_units = total_population // self.avg_dwell_size
        return dwelling_units

@dataclass (frozen=True)
class NSISchema:  ## National Structure Inventory Schema
    GEOM: Final[str] = 'geometry' ## Longitude, Latitude
    FD_ID: Final[str] = 'fd_id'
    BID: Final[str] = 'bid'
    X: Final[str] = 'x'
    Y: Final[str] = 'y'
    CBFIPS: Final[str] = 'cbfips'
    ST_DAMCAT: Final[str] = 'st_damcat'
    OCCTYPE: Final[str] = 'occtype'
    BLDGTYPE: Final[str] = 'bldgtype'
    SOURCE: Final[str] = 'source'
    SQFT: Final[str] = 'sqft'
    FTPRNTID: Final[str] = 'ftprntid'
    FTPRNTSRC: Final[str] = 'ftprntsrc'
    FOUND_TYPE: Final[str] = 'found_type'
    FOUND_HT: Final[str] = 'found_ht'
    NUM_STORY: Final[str] = 'num_story'
    VAL_STRUCT: Final[str] = 'val_struct'
    VAL_CONT: Final[str] = 'val_cont'
    VAL_VEHIC: Final[str] = 'val_vehic'
    MED_YR_BLT: Final[str] = 'med_yr_blt'
    POP2AMU65: Final[str] = 'pop2amu65'
    POP2AMO65: Final[str] = 'pop2amo65'
    POP2PMU65: Final[str] = 'pop2pmu65'
    POP2PMO65: Final[str] = 'pop2pmo65'
    STUDENTS: Final[str] = 'students'
    O65DISABLE: Final[str] = 'o65disable'
    U65SIABLE: Final[str] = 'u65siable'
    FIRMZONE: Final[str] = 'firmzone'
    GRND_ELV_M: Final[str] = 'grnd_elv_m'
    GROUND_ELV: Final[str] = 'ground_elv'

@dataclass(frozen=True)
class INCOREBuildingSchema: ## Chosen schema: ergo:buildingInventoryVer4
    GEOM: Final[str] = 'geom' ## shapely points Point(long,lat)
    PARID: Final[str] = 'parid'
    STRUCT_TYP: Final[str] = 'struct_typ'
    STR_TYP2: Final[str] = 'str_typ2'
    YEAR_BUILT: Final[str] = 'year_built'
    NO_STORIES: Final[str] = 'no_stories'
    OCC_TYPE: Final[str] = 'occ_type'
    APPR_BLDG: Final[str] = 'appr_bldg'
    CONT_VAL: Final[str] = 'cont_val'
    EFACILITY: Final[str] = 'efacility'
    DWELL_UNIT: Final[str] = 'dwell_unit'
    SQ_FOOT: Final[str] = 'sq_foot'
    CAPACITY_REDUCTION: Final[str] = 'capacity_red'
    GUID: Final[str] = 'guid'

@dataclass (frozen=True)
class StudyBuildingSchema(INCOREBuildingSchema):
    PGA: Final[str] = 'pga'
    PLS0: Final[str] = 'pls0'
    PLS1: Final[str] = 'pls1'
    PLS2: Final[str] = 'pls2'
    PLS3: Final[str] = 'pls3'
    PLS4: Final[str] = 'pls4'
    DAMAGE_STATE: Final[str] = 'damage_state'
    INIT_REPAIR_TIME: Final[str] = 'init_repair_time'
    CURR_REPAIR_TIME: Final[str] = 'curr_repair_time'
    ACCESSIBLE: Final[str] = 'accessible_repair'
    ACCESS_ROAD_IDX: Final[str] = 'access_road_index'
    DEBRIS_GEOM: Final[str] = 'debris_geo'
    IS_FIRE_STATION: Final[str] = 'is_fire_station'
    IS_HOSPITAL: Final[str] = 'is_hospital'

@dataclass (frozen=True)
class MaterialLabel:
    MASONRY: Final[str] = 'M'
    CONCRETE: Final[str] = 'C'
    STEEL: Final[str] = 'S'
    WOOD: Final[str]= 'W'
    MANUFACTURED: Final[str] = 'H'

class CollapseFunctions:
    """
    Contains the different collapse functions for various materials.
    """
    @staticmethod
    def masonry_collapse(L: float, W: float, Af: float, h: float, Vb: float) -> float:
        """
        Masonry collapse function:
        returns ε = 1.228 + 0.0787 * L / W + 0.0563 * Afh * h^2 / (b * Vb * L)
        - L and W are Length and Width of building footprint respectively
        - Af is the area of the building footprint
        - h is the height of the building
        - Vb is the volume of masonry of the building (normally some fraction of the total volume of the building)
        where ε is an amplification factor which is applied on the area of the building to get the area of the debris field.
        """
        return 1.228 + 0.0787 * (L / W) + 0.0563 * (Af * h**2) / (Vb * L)

    @staticmethod
    def concrete_collapse(collapse_mode: str, stories: int, random_state: int = None) -> Tuple[float, float, float, float]:
        """
        Concrete collapse function for RC moment frame buildings:
        Uses a lognormal distribution to calculate debris extent in four directions (a, b, c, d).
        """
        parameter_table = {
            ("Aligned", 4): (0.84, 0.09, 0.14, 0.68, 0.36, 0.29, 0.39, 0.23),
            ("Aligned", 8): (0.74, 0.17, 0.12, 0.38, 0.21, 0.37, 0.22, 0.46),
            ("Aligned", 12): (0.72, 0.16, 0.09, 0.43, 0.14, 0.52, 0.16, 0.48),
            ("Skewed", 4): (0.76, 0.081, 0.29, 0.38, 0.54, 0.15, 0.29, 0.23),
            ("Skewed", 8): (0.66, 0.21, 0.13, 0.23, 0.46, 0.24, 0.14, 0.44),
            ("Skewed", 12): (0.64, 0.20, 0.08, 0.37, 0.39, 0.24, 0.10, 0.22),
        }
        stories = min([4,8,12], key=lambda x: abs(x - stories))
        params = parameter_table.get((collapse_mode, stories))
        if not params:
            raise ValueError(f"Invalid combination of collapse_mode={collapse_mode} and stories={stories}")

        med_a, disp_a, med_b, disp_b, med_c, disp_c, med_d, disp_d = params

        # Sample areas
        dim_a = utils.sample_lognormal(med_a, disp_a)
        dim_b = utils.sample_lognormal(med_b, disp_b)
        dim_c = utils.sample_lognormal(med_c, disp_c)
        dim_d = utils.sample_lognormal(med_d, disp_d)

        return float(dim_a), float(dim_b), float(dim_c), float(dim_d)

    # @staticmethod
    # def all_other_collapse(W: float, H: float, kv: float = utils.sample_lognormal(0.5, 0.15), theta: float = utils.sample_lognormal(45, 13.5)) -> float:
    #     """
    #     Generic collapse function for materials like Steel, Wood, Manufactured
    #     """
    #     return math.sqrt(W**2 + ((2 * kv * W * H) / math.tan(theta)) - W)
    @staticmethod
    def all_other_collapse(W: float, H: float, kv: float = None, theta: float = None) -> float:
        # Function to extract first item if input is a list or NumPy array
        def get_scalar(value):
            if isinstance(value, np.ndarray):
                return value.item()  # Explicitly extract scalar
            elif isinstance(value, list):
                return value[0]
            return value

        # Apply scalar extraction to inputs

        if kv is None:
            kv = utils.sample_lognormal(0.5, 0.15)
        if theta is None:
            theta = max(5, min(utils.sample_lognormal(45, 13.5), 80))\

        W = get_scalar(W)
        H = get_scalar(H)
        kv = get_scalar(kv)
        theta = get_scalar(theta)

        sqrt_input = max(0, W**2 + ((2 * kv * W * H) / math.tan(math.radians(theta))) - W)
        sqrt_input = get_scalar(sqrt_input)
        return math.sqrt(sqrt_input)

@dataclass (frozen=True)
class BuildingOccupancy(str, Enum):
    COMMERCIAL = 'COM'
    INDUSTRIAL = 'IND'
    PUBLIC = 'PUB'
    RESIDENTIAL = 'RES'

@dataclass(frozen=True)
class SquareFootageThresholds:
    MASONRY_COMMERCIAL: Final[int] = 5000
    MASONRY_RESIDENTIAL: Final[int] = 3000
    WOOD: Final[int] = 5000

@dataclass(frozen=True)
class StoryThresholds:
    LOW: Final[int] = 3
    MEDIUM: Final[int] = 7

@dataclass(frozen=True)
class BuildingMaterials:
    CONCRETE_SMALL: Final[List[str]] = ('C1', 'C3')
    CONCRETE_LARGE: Final[List[str]] = ('C2', 'PC1', 'PC2')
    STEEL_COMMERCIAL: Final[List[str]] = ('S4', 'S5')
    STEEL_RESIDENTIAL: Final[List[str]] = ('S1', 'S2', 'S3')

@dataclass(frozen=True)
class HeightCodes:
    LOW: Final[str] = 'L'
    MEDIUM: Final[str] = 'M'
    HIGH: Final[str] = 'H'

@dataclass (frozen=True)
class StructuralTypePredictor:
    def get_height_code(self, num_stories: int) -> str:
        """Determine height code based on number of stories"""
        if num_stories <= StoryThresholds.LOW:
            return HeightCodes.LOW
        elif num_stories <= StoryThresholds.MEDIUM:
            return HeightCodes.MEDIUM
        return HeightCodes.HIGH

    def predict_masonry_type(
        self,
        building_occupancy: BuildingOccupancy,
        sq_footage: int,
        num_stories: int
    ) -> str:
        """Predict structural type for masonry buildings"""
        if building_occupancy in [BuildingOccupancy.COMMERCIAL,
                                BuildingOccupancy.INDUSTRIAL,
                                BuildingOccupancy.PUBLIC]:
            if sq_footage <= SquareFootageThresholds.MASONRY_COMMERCIAL:
                return 'RM1L' if num_stories <= StoryThresholds.LOW else 'RM1M'
            else:
                if num_stories <= StoryThresholds.LOW:
                    return 'RM2L'
                elif num_stories <= StoryThresholds.MEDIUM:
                    return 'RM2M'
                return 'RM2L'
        else:  # Residential
            if sq_footage <= SquareFootageThresholds.MASONRY_RESIDENTIAL:
                return 'URML' if num_stories <= 2 else 'URMM'
            else:
                return 'RM1L' if num_stories <= StoryThresholds.LOW else 'RM1M'

    def predict_concrete_type(
        self,
        building_occupancy: BuildingOccupancy,
        sq_footage: int,
        num_stories: int
    ) -> str:
        """Predict structural type for concrete buildings"""
        if building_occupancy in [BuildingOccupancy.COMMERCIAL,
                                BuildingOccupancy.INDUSTRIAL,
                                BuildingOccupancy.PUBLIC]:
            if sq_footage <= SquareFootageThresholds.MASONRY_COMMERCIAL:
                material = random.choice(BuildingMaterials.CONCRETE_SMALL)
                return material + self.get_height_code(num_stories)
            else:
                material = random.choice(BuildingMaterials.CONCRETE_LARGE)
                if material == 'PC1':
                    return material
                return material + self.get_height_code(num_stories)
        else:  # Residential
            return 'C3' + self.get_height_code(num_stories)

    def predict_steel_type(
        self,
        building_occupancy: BuildingOccupancy,
        num_stories: int
    ) -> str:
        """Predict structural type for steel buildings"""
        if building_occupancy in [BuildingOccupancy.COMMERCIAL,
                                    BuildingOccupancy.INDUSTRIAL,
                                    BuildingOccupancy.PUBLIC]:

            material = random.choice(BuildingMaterials.STEEL_COMMERCIAL)
            return material + self.get_height_code(num_stories)

        else:  # Residential
            material = random.choice(BuildingMaterials.STEEL_RESIDENTIAL)
            if material == 'S3':
                return material
            return material + self.get_height_code(num_stories)

    def predict_wood_type(self, sq_footage: int) -> str:
        """Predict structural type for wood buildings"""
        return 'W2' if sq_footage <= SquareFootageThresholds.WOOD else 'W1'

    def predict_str_type(
        self,
        study_label: str,
        num_stories: int,
        sq_footage: int,
        building_occupancy: str
    ) -> str:
        """
        Predict structural type based on building characteristics

        Args:
            study_label: Material type (M, C, S, W, H)
            num_stories: Number of stories in building
            sq_footage: Square footage of building
            building_occupancy: Building use (COM, IND, PUB, RES)

        Returns:
            str: Predicted structural type code
        """

        if study_label == MaterialLabel.MASONRY:
            return self.predict_masonry_type(building_occupancy, sq_footage, num_stories)

        elif study_label == MaterialLabel.CONCRETE:

            return self.predict_concrete_type(building_occupancy, sq_footage, num_stories)

        elif study_label == MaterialLabel.STEEL:
            return self.predict_steel_type(building_occupancy, num_stories)

        elif study_label == MaterialLabel.WOOD:
            return self.predict_wood_type(sq_footage)

        elif study_label == MaterialLabel.MANUFACTURED:
            return 'MH'

        else:
            raise ValueError("Invalid study label")

class BuildingDowntimeDelays:
    """
    Table 8 from ARUPS's REDi framework: https://static1.squarespace.com/static/61d5bdb2d77d2d6ccd13b976/t/61e85a429039460930278463/1642617413050/REDi_Final+Version_October+2013+Arup+Website+%288%29.pdf
    """
    def __init__(
        self,
        essential: bool,
        num_stories: int,
        financing_method: str,
        damage_state: int
    ):
        self.essential = essential
        self.num_stories = num_stories
        self.financing_method = financing_method ## "insurance" or "private_loans"
        self.ds = damage_state



    def permitting(self, ds:int):
        _permitting = np.array([
            [7, 0.86],
            [7, 0.86],
            [56, 0.32],
            [56, 0.32],
            [56, 0.32]
        ])
        mean, cov = _permitting[ds]
        sigma = np.sqrt(np.log(1 + cov**2))
        mu = np.log(mean) - 0.5 * sigma**2
        delay = np.random.lognormal(mean=mu, sigma=sigma)
        return delay

    def contractor_mobilisation(self, essential: bool, num_stories: int, ds: int):
        essential_facilities_lt_20_stories = np.array([
            [49, 0.60],
            [49, 0.60],
            [133, 0.38],
            [133, 0.38],
            [133, 0.38]
        ])
        non_essential_facilities_leq_20_stories = np.array([
            [77, 0.43],
            [77, 0.43],
            [161, 0.41],
            [161, 0.41],
            [161, 0.41]
        ])
        essential_facilities_gteq_20_stories = np.array([
            [196, 0.30],
            [196, 0.30],
            [280, 0.31],
            [280, 0.31],
            [280, 0.31]
        ])

        if essential and num_stories < 20:
            params = essential_facilities_lt_20_stories[ds]
        elif essential and num_stories >= 20:
            params = essential_facilities_gteq_20_stories[ds]
        else:
            params = non_essential_facilities_leq_20_stories[ds]
        # Select parameters
        if essential and num_stories < 20:
            mean, cov = essential_facilities_lt_20_stories[ds]
        elif essential and num_stories >= 20:
            mean, cov = essential_facilities_gteq_20_stories[ds]
        else:
            mean, cov = non_essential_facilities_leq_20_stories[ds]

        # Convert mean and COV (dispersion) to lognormal parameters
        sigma = np.sqrt(np.log(1 + cov**2))
        mu = np.log(mean) - 0.5 * sigma**2

        # Sample from lognormal
        delay = np.random.lognormal(mean=mu, sigma=sigma)

        return delay

    def financing(self, ds: int, method: str):
        insurance = np.array([
            [42, 1.11],
            [42, 1.11],
            [42, 1.11],
            [42, 1.11],
            [42, 1.11]
        ])
        private_loans = np.array([
            [105, 0.68],
            [105, 0.68],
            [105, 0.68],
            [105, 0.68],
            [105, 0.68]
        ])
        if method == 'insurance':
            params = insurance[ds]
        else:
            params = private_loans[ds]
        mean, cov = params
        sigma = np.sqrt(np.log(1 + cov**2))
        mu = np.log(mean) - 0.5 * sigma**2
        delay = np.random.lognormal(mean=mu, sigma=sigma)
        return delay

    def engineering_mobilisation(self, ds: int):
        _engineering_mobilisation = np.array([
            [42, 0.40],
            [42, 0.40],
            [105, 0.68],
            [105, 0.68],
            [105, 0.68]
        ])
        mean, cov = _engineering_mobilisation[ds]
        sigma = np.sqrt(np.log(1 + cov**2))
        mu = np.log(mean) - 0.5 * sigma**2
        delay = np.random.lognormal(mean=mu, sigma=sigma)
        return delay

    def inspection(self, essential: bool):
        _inspection = np.array([
            [2, 0.54],
            [5, 0.54]
        ])
        if essential:
            mean, cov = _inspection[0]
        else:
            mean, cov = _inspection[1]
        sigma = np.sqrt(np.log(1 + cov**2))
        mu = np.log(mean) - 0.5 * sigma**2
        delay = np.random.lognormal(mean=mu, sigma=sigma)
        return delay

    def get_delay_time(self):
        delay_time = np.sum([
            self.permitting(self.ds),
            self.contractor_mobilisation(self.essential, self.num_stories, self.ds),
            self.financing(self.ds, self.financing_method),
            self.engineering_mobilisation(self.ds),
            self.inspection(self.essential)
        ])
        return delay_time

class BuildingRecoveryData:
    """
    Table 11-7 and 11-8 and 11-9 from (HAZUS, 2024)
    distribution is assumed to be normal with standard deviation = 0.25 * mean

    Structure: {
        occtype: {
            np.array(Tds1, Tds2, Tds3, Tds4, Tds5)
        }
    }
    """

    REPAIR_TIMES = {
        ## Tables 11-7
        "RES1": np.array([0, 2, 30, 90, 180]),
        "RES2": np.array([0, 2, 10, 30, 60]),
        "RES3": np.array([0, 5, 30, 120, 240]),
        "RES4": np.array([0, 5, 30, 120, 240]),
        "RES5": np.array([0, 5, 30, 120, 240]),
        "RES6": np.array([0, 5, 30, 120, 240]),
        "COM1": np.array([0, 5, 30, 90, 180]),
        "COM2": np.array([0, 5, 30, 90, 180]),
        "COM3": np.array([0, 5, 30, 90, 180]),
        "COM4": np.array([0, 5, 30, 120, 240]),
        "COM5": np.array([0, 5, 30, 90, 180]),
        "COM6": np.array([0, 10, 45, 180, 360]),
        "COM7": np.array([0, 10, 45, 180, 240]),
        "COM8": np.array([0, 5, 30, 90, 180]),
        "COM9": np.array([0, 5, 30, 120, 240]),
        "COM10": np.array([0, 2, 20, 80, 160]),
        "IND1": np.array([0, 10, 30, 120, 240]),
        "IND2": np.array([0, 10, 30, 120, 240]),
        "IND3": np.array([0, 10, 30, 120, 240]),
        "IND4": np.array([0, 10, 30, 120, 240]),
        "IND5": np.array([0, 20, 45, 180, 360]),
        "IND6": np.array([0, 5, 20, 80, 160]),
        "AGR1": np.array([0, 2, 10, 30, 60]),
        "REL1": np.array([0, 10, 30, 120, 240]),
        "GOV1": np.array([0, 10, 30, 120, 240]),
        "GOV2": np.array([0, 5, 20, 90, 180]),
        "EDU1": np.array([0, 10, 30, 120, 240]),
        "EDU2": np.array([0, 10, 45, 180, 360])
    }

    RECOVERY_TIMES = {
        ## Tables 11-8
        "RES1": np.array([0, 5, 120, 360, 720]),
        "RES2": np.array([0, 5, 20, 120, 240]),
        "RES3": np.array([0, 10, 120, 480, 960]),
        "RES4": np.array([0, 10, 90, 360, 480]),
        "RES5": np.array([0, 10, 90, 360, 480]),
        "RES6": np.array([0, 10, 120, 480, 960]),
        "COM1": np.array([0, 10, 90, 270, 360]),
        "COM2": np.array([0, 10, 90, 270, 360]),
        "COM3": np.array([0, 10, 90, 270, 360]),
        "COM4": np.array([0, 20, 90, 360, 480]),
        "COM5": np.array([0, 20, 90, 180, 360]),
        "COM6": np.array([0, 20, 135, 540, 720]),
        "COM7": np.array([0, 20, 135, 270, 540]),
        "COM8": np.array([0, 20, 90, 180, 360]),
        "COM9": np.array([0, 20, 90, 180, 360]),
        "COM10": np.array([0, 5, 60, 180, 360]),
        "IND1": np.array([0, 10, 90, 240, 360]),
        "IND2": np.array([0, 10, 90, 240, 360]),
        "IND3": np.array([0, 10, 90, 240, 360]),
        "IND4": np.array([0, 10, 90, 240, 360]),
        "IND5": np.array([0, 20, 135, 360, 540]),
        "IND6": np.array([0, 10, 60, 160, 320]),
        "AGR1": np.array([0, 2, 20, 60, 120]),
        "REL1": np.array([0, 5, 120, 480, 960]),
        "GOV1": np.array([0, 10, 90, 360, 480]),
        "GOV2": np.array([0, 10, 60, 270, 360]),
        "EDU1": np.array([0, 10, 90, 360, 480]),
        "EDU2": np.array([0, 10, 120, 480, 960])
    }

    SERVICE_INTERRUPTION_MULTIPLIERS = {
        # Table 11-9
        "RES1": np.array([0, 0, 0.5, 1.0, 1.0]),
        "RES2": np.array([0, 0, 0.5, 1.0, 1.0]),
        "RES3": np.array([0, 0, 0.5, 1.0, 1.0]),
        "RES4": np.array([0, 0, 0.5, 1.0, 1.0]),
        "RES5": np.array([0, 0, 0.5, 1.0, 1.0]),
        "RES6": np.array([0, 0, 0.5, 1.0, 1.0]),
        "COM1": np.array([0.5, 0.1, 0.1, 0.3, 0.4]),
        "COM2": np.array([0.5, 0.1, 0.2, 0.3, 0.4]),
        "COM3": np.array([0.5, 0.1, 0.2, 0.3, 0.4]),
        "COM4": np.array([0.5, 0.1, 0.1, 0.2, 0.3]),
        "COM5": np.array([0.5, 0.1, 0.05, 0.03, 0.03]),
        "COM6": np.array([0.5, 0.1, 0.5, 0.5, 0.5]),
        "COM7": np.array([0.5, 0.1, 0.5, 0.5, 0.5]),
        "COM8": np.array([0.5, 0.1, 1.0, 1.0, 1.0]),
        "COM9": np.array([0.5, 0.1, 1.0, 1.0, 1.0]),
        "COM10": np.array([0.1, 0.1, 1.0, 1.0, 1.0]),
        "IND1": np.array([0.5, 0.5, 1.0, 1.0, 1.0]),
        "IND2": np.array([0.5, 0.1, 0.2, 0.3, 0.4]),
        "IND3": np.array([0.5, 0.2, 0.2, 0.3, 0.4]),
        "IND4": np.array([0.5, 0.2, 0.2, 0.3, 0.4]),
        "IND5": np.array([0.5, 0.2, 0.2, 0.3, 0.4]),
        "IND6": np.array([0.5, 0.1, 0.2, 0.3, 0.4]),
        "AGR1": np.array([0, 0, 0.05, 0.1, 0.2]),
        "REL1": np.array([1.0, 0.2, 0.05, 0.03, 0.03]),
        "GOV1": np.array([0.5, 0.1, 0.02, 0.03, 0.03]),
        "GOV2": np.array([0.5, 0.1, 0.02, 0.03, 0.03]),
        "EDU1": np.array([0.5, 0.1, 0.02, 0.05, 0.05]),
        "EDU2": np.array([0.5, 0.1, 0.02, 0.03, 0.03])
    }

    INCOME_RECAPTURE_FACTORS = {
        "RES1": np.array([0, 0, 0, 0]),
        "RES2": np.array([0, 0, 0, 0]),
        "RES3": np.array([0, 0, 0, 0]),
        "RES4": np.array([0.60, 0.60, 0.60, 0.60]),
        "RES5": np.array([0.60, 0.60, 0.60, 0.60]),
        "RES6": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM1": np.array([0.87, 0.87, 0.87, 0.87]),
        "COM2": np.array([0.87, 0.87, 0.87, 0.87]),
        "COM3": np.array([0.51, 0.51, 0.51, 0.51]),
        "COM4": np.array([0.90, 0.90, 0.90, 0.90]),
        "COM5": np.array([0.90, 0.90, 0.90, 0.90]),
        "COM6": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM7": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM8": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM9": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM10": np.array([0.60, 0.60, 0.60, 0.60]),
        "IND1": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND2": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND3": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND4": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND5": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND6": np.array([0.95, 0.95, 0.95, 0.95]),
        "AGR1": np.array([0.75, 0.75, 0.75, 0.75]),
        "REL1": np.array([0.60, 0.60, 0.60, 0.60]),
        "GOV1": np.array([0.80, 0.80, 0.80, 0.80]),
        "GOV2": np.array([0, 0, 0, 0]),
        "EDU1": np.array([0.60, 0.60, 0.60, 0.60]),
        "EDU2": np.array([0.60, 0.60, 0.60, 0.60])
    }

    PROPRIETORS_INCOME = {
        ## Income per sqft per day,
        # wages per sqft per day,
        # employees per sqft,
        # output per sqft per day
        'RES1': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES2': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES3A': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES3B': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES3C': np.array([0.000, 0.000, 0.000, 0.0000]),
        'RES3D': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES3E': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES3F': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES4': np.array([0.132, 0.311, 0.003, 0.693]),
        'RES5': np.array([0.000, 0.000, 0.000, 0.000]),
        'RES6': np.array([0.221, 0.519, 0.005, 1.156]),
        'COM1': np.array([0.082, 0.285, 0.004, 0.603]),
        "COM2": np.array([0.134, 0.351, 0.002, 0.784]),
        "COM3": np.array([0.176, 0.415, 0.004, 0.925]),
        "COM4": np.array([1.390, 0.494, 0.004, 1.351]),
        "COM5": np.array([1.587, 0.805, 0.006, 4.387]),
        "COM6": np.array([0.221, 0.519, 1.005, 1.156]),
        "COM7": np.array([0.441, 1.039, 0.010, 2.311]),
        "COM8": np.array([0.809, 0.644, 0.007, 1.457]),
        "COM9": np.array([0.265, 0.624, 0.006, 1.388]),
        "COM10": np.array([0.000, 0.000, 0.000, 0.000]),
        "IND1": np.array([0.335, 0.554, 0.003, 2.342]),
        "IND2": np.array([0.335, 0.554, 0.003, 2.342]),
        "IND3": np.array([0.446, 0.741, 0.004, 3.123]),
        "IND4": np.array([1.014, 0.572, 0.003, 2.478]),
        "IND5": np.array([0.669, 1.110, 0.006, 4.683]),
        "IND6": np.array([0.326, 0.600, 0.005, 2.321]),
        "AGR1": np.array([0.310, 0.123, 0.004, 1.156]),
        "REL1": np.array([0.176, 0.415, 0.004, 2.311]),
        "GOV1": np.array([0.145, 3.986, 0.025, 0.925]),
        "GOV2": np.array([0.000, 6.060, 0.038, 1.062]),
        "EDU1": np.array([0.221, 0.519, 0.05, 4.478]),
        "EDU2": np.array([0.441, 1.039, 0.010, 6.806])
    }

    NON_STRUCT_REP_COSTS_DRIFT_SENS = { ## Table 11-4, unit: % of building replacement cost
        'RES1': np.array([1.0, 5.0, 25.0, 50.0]),
        'RES2': np.array([0.8, 3.8, 18.9, 37.8]),
        'RES3A': np.array([0.9, 4.3, 21.3, 42.5]),
        'RES4': np.array([0.9, 4.3, 21.6, 43.2]),
        'RES5': np.array([0.8, 4.0, 20.0, 40.0]),
        'RES6': np.array([0.8, 4.1, 20.4, 40.8]),
        'COM1': np.array([0.6, 2.7, 13.8, 27.5]),
        'COM2': np.array([0.6, 2.6, 13.2, 26.5]),
        'COM3': np.array([0.7, 3.4, 16.9, 33.8]),
        'COM4': np.array([0.7, 3.3, 16.4, 32.9]),
        'COM5': np.array([0.7, 3.4, 17.2, 34.5]),
        'COM6': np.array([0.8, 3.5, 17.4, 34.7]),
        'COM7': np.array([0.7, 3.4, 17.2, 34.4]),
        'COM8': np.array([0.7, 3.6, 17.8, 35.6]),
        'COM9': np.array([0.7, 3.5, 17.6, 35.1]),
        'COM10': np.array([0.4, 1.7, 8.7, 17.4]),
        'IND1': np.array([0.2, 1.2, 5.9, 11.8]),
        'IND2': np.array([0.2, 1.2, 5.9, 11.8]),
        'IND3': np.array([0.2, 1.2, 5.9, 11.8]),
        'IND4': np.array([0.2, 1.2, 5.9, 11.8]),
        'IND5': np.array([0.2, 1.2, 5.9, 11.8]),
        'IND6': np.array([0.2, 1.2, 5.9, 11.8]),
        'AGR1': np.array([0.0, 0.8, 3.8, 7.7]),
        'REL1': np.array([0.8, 3.3, 16.3, 32.6]),
        'GOV1': np.array([0.7, 3.3, 16.4, 32.8]),
        'GOV2': np.array([0.7, 3.4, 17.1, 34.2]),
        'EDU1': np.array([0.9, 4.9, 24.3, 48.7]),
        'EDU2': np.array([1.2, 6.0, 30.0, 60.0]),
    }

    PRCNT_OWNER_OCCUPIED = {
        'RES1': 0.75,
        'RES2': 0.85,
        'RES3A': 0.35,
        'RES3B': 0.35,
        'RES3C': 0.35,
        'RES3D': 0.35,
        'RES3E': 0.35,
        'RES3F': 0.35,
        'RES4': 0.00,
        'RES5': 0.00,
        'RES6': 0.00,
        'COM1': 0.55,
        'COM2': 0.55,
        'COM3': 0.55,
        'COM4': 0.55,
        'COM5': 0.75,
        'COM6': 0.95,
        'COM7': 0.65,
        'COM8': 0.55,
        'COM9': 0.45,
        'COM10': 0.25,
        'IND1': 0.75,
        'IND2': 0.75,
        'IND3': 0.75,
        'IND4': 0.75,
        'IND5': 0.55,
        'IND6': 0.85,
        'AGR1': 0.95,
        'REL1': 0.90,
        'GOV1': 0.70,
        'GOV2': 0.95,
        'EDU1': 0.95,
    }

    DISRUPTION_COSTS = {
        # """
        # Table 6-13 from hazus inventory technical manual
        # Columns:
        # - Rental Cost ( $ / sqft / month)
        # - Rental Cost ($ / sqft / day)
        # - Disruption Cost: ($ / sqft)
        # """
        'RES1': np.array([0.91, 0.030, 1.10]),
        'RES2': np.array([0.64, 0.021, 1.10]),
        'RES3A': np.array([0.82, 0.027, 1.10]),
        'RES3B': np.array([0.82, 0.027, 1.10]),
        'RES3C': np.array([0.82, 0.027, 1.10]),
        'RES3D': np.array([0.82, 0.027, 1.10]),
        'RES3E': np.array([0.82, 0.027, 1.10]),
        'RES3F': np.array([0.82, 0.027, 1.10]),
        'RES4': np.array([2.74, 0.091, 1.10]),
        'RES5': np.array([0.55, 0.018, 1.10]),
        'RES6': np.array([1.01, 0.034, 1.10]),
        'COM1': np.array([1.55, 0.052, 1.46]),
        'COM2': np.array([0.64, 0.021, 1.28]),
        'COM3': np.array([1.83, 0.061, 1.28]),
        'COM4': np.array([1.83, 0.061, 1.28]),
        'COM5': np.array([2.29, 0.076, 1.28]),
        'COM6': np.array([1.83, 0.061, 1.83]),
        'COM7': np.array([1.83, 0.061, 1.83]),
        'COM8': np.array([2.29, 0.076, 0.00]),
        'COM9': np.array([2.29, 0.076, 0.00]),
        'COM10': np.array([0.46, 0.015, 0.00]),
        'IND1': np.array([0.27, 0.009, 0.00]),
        'IND2': np.array([0.37, 0.012, 1.28]),
        'IND3': np.array([0.37, 0.012, 1.28]),
        'IND4': np.array([0.27, 0.009, 1.28]),
        'IND5': np.array([0.46, 0.015, 1.28]),
        'IND6': np.array([0.18, 0.006, 1.28]),
        'AGR1': np.array([0.91, 0.030, 0.91]),
        'REL1': np.array([1.37, 0.046, 1.28]),
        'GOV1': np.array([1.83, 0.061, 1.28]),
        'GOV2': np.array([1.83, 0.061, 1.28]),
        'EDU1': np.array([1.37, 0.046, 1.28]),
        'EDU2': np.array([1.83, 0.061, 1.28]),
    }

    @staticmethod
    def compute_repair_time_bins() -> np.ndarray:
        """Compute the min and max repair time based on ±3 standard deviations."""
        all_repair_times = np.concatenate(list(BuildingRecoveryData.REPAIR_TIMES.values()))

        mean_time = np.mean(all_repair_times)
        std_time = np.std(all_repair_times)  # Compute actual standard deviation

        min_time = mean_time - 3 * std_time
        max_time = mean_time + 3 * std_time



        min_rt = math.ceil(max(0, min_time))
        max_rt = math.ceil(max_time)

        bins = list(range(0, max_rt + 7, 7))

        return 0, max_rt

    @staticmethod
    def get_repair_time_bin(repair_time, bins):
        """
        Assigns a repair time to its respective bin using a modified approach:
        If the repair_time exceeds the maximum bin, it assigns it to the last bin.
        Otherwise, it uses searchsorted to find the appropriate index.
        """
        if repair_time == 0:
            return 0  # Separate bin for 0

        # Get the index where the repair_time would fit
        index = np.searchsorted(bins, repair_time, side="right")

        # If the index is beyond the last bin, return the last index
        if index >= len(bins):
            return len(bins) - 1  # Last bin index

        return index

    def get_recovery_time(self, occupancy_type: str):
        if occupancy_type [:4] == "RES1":
            BRT =  self.RECOVERY_TIMES["RES1"]
        elif occupancy_type[:4] == "RES3":
            BRT = self.RECOVERY_TIMES["RES3"]
        elif occupancy_type not in self.REPAIR_TIMES:
            raise KeyError(f"{occupancy_type} not found in DISTRIBUTIONS")
        else:
            BRT =  self.RECOVERY_TIMES[occupancy_type]

        return BRT

    def get_loss_of_function_time(self, occupancy_type: str):
        if occupancy_type [:4] == "RES1":
            BRT =  self.RECOVERY_TIMES["RES1"]
            MOD = self.SERVICE_INTERRUPTION_MULTIPLIERS["RES1"]
            REP = self.REPAIR_TIMES["RES1"]
        elif occupancy_type[:4] == "RES3":
            BRT = self.RECOVERY_TIMES["RES3"]
            MOD = self.SERVICE_INTERRUPTION_MULTIPLIERS["RES3"]
            REP = self.REPAIR_TIMES["RES3"]
        elif occupancy_type not in self.REPAIR_TIMES:
            raise KeyError(f"{occupancy_type} not found in DISTRIBUTIONS")
        else:
            BRT =  self.RECOVERY_TIMES[occupancy_type]
            MOD = self.SERVICE_INTERRUPTION_MULTIPLIERS[occupancy_type]
            REP = self.REPAIR_TIMES[occupancy_type]
        LOF = BRT * MOD + REP

        return LOF

    def get_recapture_factors(self, occupancy_type: str):
        if occupancy_type[:4] == "RES1":
            FA =  self.INCOME_RECAPTURE_FACTORS["RES1"]
        elif occupancy_type[:4] == "RES3":
            FA = self.INCOME_RECAPTURE_FACTORS["RES3"]
        elif occupancy_type not in self.INCOME_RECAPTURE_FACTORS:
            raise KeyError(f"{occupancy_type} not found in DISTRIBUTIONS")
        else:
            FA = self.INCOME_RECAPTURE_FACTORS[occupancy_type]
        return FA

    def get_income_per_day(self, occupancy_type:str):
        if occupancy_type[:4] == "RES1":
            INC =  self.PROPRIETORS_INCOME["RES1"]
        elif occupancy_type not in self.PROPRIETORS_INCOME:
            raise KeyError(f"{occupancy_type} not found in DISTRIBUTIONS")
        else:
            INC = self.PROPRIETORS_INCOME[occupancy_type]
        return INC

    def get_repair_time(self, occupancy_type: str) -> Dict[str, float]:
        """
        Retrieve the repair times for a specific occupancy type.

        :param occupancy_type: The occupancy type code (e.g., 'RES1', 'COM6')
        :return: Dictionary of damage states and their mean PGA and beta values
        :raises KeyError: If the occupancy type is not found in the distributions
        """
        if occupancy_type [:4] == "RES1":
            return self.REPAIR_TIMES["RES1"]
        elif occupancy_type[:4] == "RES3":
            return self.REPAIR_TIMES["RES3"]
        elif occupancy_type not in self.REPAIR_TIMES:
            raise KeyError(f"{occupancy_type} not found in DISTRIBUTIONS")
        return self.REPAIR_TIMES[occupancy_type]

    def get_relocation_cost(self,
        occupancy_type: str,
        sqft: int,
        damage_state_probs: np.array
    ):
        """
        Compute relocation expenses for a given occupancy class based on floor area, rental costs, disruption costs,
        probability of structural damage states, and expected recovery time.

        Parameters:
        occupancy_class (str): The occupancy class identifier (e.g., 'COM1', 'EDU2').
        floor_area (float): The floor area of the building in square feet.
        structural_damage_probs (np.array): Probability distribution of the building being in each structural damage state.
        recovery_times (np.array): Expected recovery times (in days) for each structural damage state.

        Returns:
        float: Estimated relocation expenses in dollars.

        Formula:
        REL_i = FA_i * (1 - %OO_i) * sum(POSTR_ds,i * (DC_i + RENT_i * RT_ds))

        Where:
        - REL_i: Relocation costs for occupancy class i.
        - FA_i: Floor area of occupancy class i (in sqft).
        - %OO_i: Percent owner occupied for occupancy class i.
        - POSTR_ds,i: Probability of the occupancy class being in structural damage state ds.
        - DC_i: Disruption costs for occupancy class i (in $/sqft).
        - RENT_i: Rental cost for occupancy class i (in $/sqft/day).
        - RT_ds: Recovery time for damage state ds (in days).
        """

        area = sqft
        ds_probs = damage_state_probs

        if occupancy_type[:4] == "RES1":
            OO =  self.PRCNT_OWNER_OCCUPIED["RES1"]
            DC = self.DISRUPTION_COSTS["RES1"][2]
            RENT = self.DISRUPTION_COSTS["RES1"][0] ## value in  $ / sqft / month
        elif occupancy_type not in self.PROPRIETORS_INCOME:
            raise KeyError(f"{occupancy_type} not found in DISTRIBUTIONS")
        else:
            OO = self.PRCNT_OWNER_OCCUPIED[occupancy_type]
            DC = self.DISRUPTION_COSTS[occupancy_type][2]
            RENT = self.DISRUPTION_COSTS[occupancy_type][0] ## value in $ / sqft / month

        RT = self.get_recovery_time(occupancy_type=occupancy_type)
        # print(f"area: {area}")
        # print(f"onwer occ: {OO}")
        # print(f"ds probs: {ds_probs}")
        # print(f"disrupt costs: {DC}")
        # print(f"Rental Costs: {RENT}")
        # print(f"Recovery Time: {RT}")
        REL = area * math.ceil(
            (1 - OO) * np.sum(ds_probs * DC) + OO * np.sum(ds_probs * (DC + RENT + RT))
        )

        return REL

    def get_num_hosp_beds(self,
        sqft: float,
        occtype: str
    ):
        """Compute number of hospital beds based on ``HAZUS_INVENTORY_TECH_MANUAL`` Table 7-3"""
        if sqft >= 300000:
            return 200
        elif sqft <= 100000:
            return 50
        return 100

    def get_num_doctors(self,
        sqft: float
    )->int:
        """Compute number of doctors """
        ## assume 75% of employees in a medical office are doctors or nurses
        return math.ceil(sqft * (0.75 * self.PROPRIETORS_INCOME["COM6"][2]))

class BuildingReplacementCosts:
    """
    Table 6-2 and 6-3 from (HAZUS, 2024)
    """
    OCCUPANCIES: Dict[Tuple[str, int], List[float]] = {
        # (Occupancy Type, Number of Stories): [Building Replacement Cost per sqft, Vehicle Replacement Cost]
        ("RES1", 1): 140.95,
        ("RES1", 2): 140.98,
        ("RES1", 3): 130.96,
        ("RES2", -1): 73.92,
        ("RES3A", -1): 134.57,
        ("RES3B", -1): 118.30,
        ("RES3C", -1): 250.19,
        ("RES3D", -1): 232.69,
        ("RES3E", -1): 217.05,
        ("RES3F", -1): 198.83,
        ("RES4", -1): 204.58,
        ("RES5", -1): 225.12,
        ("RES6", -1): 261.45,
        ("COM1", -1): 133.61,
        ("COM2", -1): 135.47,
        ("COM3", -1): 162.58,
        ("COM4", -1): 204.43,
        ("COM5", -1): 303.88,
        ("COM6", -1): 392.76,
        ("COM7", -1): 279.38,
        ("COM8", -1): 259.48,
        ("COM9", -1): 213.33,
        ("COM10", -1): 93.88,
        ("IND1", -1): 162.65,
        ("IND2", -1): 135.47,
        ("IND3", -1): 230.43,
        ("IND4", -1): 230.43,
        ("IND5", -1): 230.43,
        ("IND6", -1): 135.47,
        ("AGR1", -1): 135.47,
        ("REL1", -1): 222.10,
        ("GOV1", -1): 174.35,
        ("GOV2", -1): 284.46,
        ("EDU1", -1): 237.73,
        ("EDU2", -1): 197.10
    }
    def get_costs(self, occupancy_type: str, num_stories: int = -1) -> Dict[str, float]:
        """
        Retrieve the repair times for a specific occupancy type.

        :param occupancy_type: The occupancy type code (e.g., 'RES1', 'COM6')
        :return: Dictionary of damage states and their mean PGA and beta values
        :raises KeyError: If the occupancy type is not found in the distributions
        """
        if occupancy_type[:4] == "RES1":
            return self.OCCUPANCIES[("RES1", num_stories)]
        elif (occupancy_type, -1) not in self.OCCUPANCIES:
            raise KeyError(f"{occupancy_type} not found in OCCUPANCIES")
        return self.OCCUPANCIES[(occupancy_type, -1)]

class StructuralRepairCostRatios:
    """
    Table 11-2 Structural Repair Costs
    """
    COSTS: Dict[str, np.ndarray] = {
        # (Occupancy Type): (Slight, Moderate, Extensive, Complete) as % of building replacement cost
        "RES1": np.array([0.5, 2.3, 11.7, 23.4]),
        "RES2": np.array([0.4, 2.4, 7.3, 24.4]),
        "RES3": np.array([0.3, 1.4, 6.9, 13.8]),
        "RES4": np.array([0.2, 1.4, 6.8, 13.6]),
        "RES5": np.array([0.4, 1.9, 9.4, 18.8]),
        "RES6": np.array([0.4, 1.8, 9.2, 18.4]),
        "COM1": np.array([0.6, 2.9, 14.7, 29.4]),
        "COM2": np.array([0.6, 3.2, 16.2, 32.4]),
        "COM3": np.array([0.3, 1.6, 8.1, 16.2]),
        "COM4": np.array([0.4, 1.9, 9.6, 19.2]),
        "COM5": np.array([0.3, 1.4, 6.9, 13.8]),
        "COM6": np.array([0.2, 1.4, 7.0, 14.0]),
        "COM7": np.array([0.3, 1.4, 7.2, 14.4]),
        "COM8": np.array([0.2, 1.0, 5.0, 10.0]),
        "COM9": np.array([0.3, 1.2, 6.1, 12.2]),
        "COM10": np.array([1.3, 6.1, 30.4, 60.9]),
        "IND1": np.array([0.4, 1.6, 7.8, 15.7]),
        "IND2": np.array([0.4, 1.6, 7.8, 15.7]),
        "IND3": np.array([0.4, 1.6, 7.8, 15.7]),
        "IND4": np.array([0.4, 1.6, 7.8, 15.7]),
        "IND5": np.array([0.4, 1.6, 7.8, 15.7]),
        "IND6": np.array([0.4, 1.6, 7.8, 15.7]),
        "AGR1": np.array([0.8, 4.6, 23.1, 46.2]),
        "REL1": np.array([0.3, 2.0, 9.9, 19.8]),
        "GOV1": np.array([0.3, 1.8, 9.0, 17.9]),
        "GOV2": np.array([0.3, 1.5, 7.7, 15.3]),
        "EDU1": np.array([0.4, 1.9, 9.5, 18.9]),
        "EDU2": np.array([0.2, 1.1, 5.5, 11.0])
    }

    def get_repair_costs(self, occupancy_type: str) -> Tuple[float, float, float, float]:
        """
        Retrieve the structural repair costs for a specific occupancy type.

        :param occupancy_type: The occupancy type code (e.g., 'COM1', 'IND3')
        :return: Tuple containing slight, moderate, extensive, and complete repair costs
        :raises KeyError: If the occupancy type is not found in the repair cost data
        """
        if occupancy_type[:4] == "RES1":
            return self.COSTS["RES1"]
        elif occupancy_type[:4] == "RES3":
            return self.COSTS["RES3"]
        elif occupancy_type not in self.COSTS:
            raise KeyError(f"{occupancy_type} not found in COSTS")
        return self.COSTS[occupancy_type]

class RecapturFactors:
    """
        Table 6-17 from Hazus inventory Technical Manual
    """

    RECAPTURE_FACTORS = {
        ## 4 values are given: Wage Recapture (%), Employment Recapture (%), Income Recapture (%), Output Recapture (%)
        "RES1": np.array([0, 0, 0, 0]),
        "RES2": np.array([0, 0, 0, 0]),
        "RES3": np.array([0, 0, 0, 0]),
        "RES4": np.array([0.60, 0.60, 0.60, 0.60]),
        "RES5": np.array([0.60, 0.60, 0.60, 0.60]),
        "RES6": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM1": np.array([0.87, 0.87, 0.87, 0.87]),
        "COM2": np.array([0.87, 0.87, 0.87, 0.87]),
        "COM3": np.array([0.51, 0.51, 0.51, 0.51]),
        "COM4": np.array([0.90, 0.90, 0.90, 0.90]),
        "COM5": np.array([0.90, 0.90, 0.90, 0.90]),
        "COM6": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM7": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM8": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM9": np.array([0.60, 0.60, 0.60, 0.60]),
        "COM10": np.array([0.60, 0.60, 0.60, 0.60]),
        "IND1": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND2": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND3": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND4": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND5": np.array([0.98, 0.98, 0.98, 0.98]),
        "IND6": np.array([0.95, 0.95, 0.95, 0.95]),
        "AGR1": np.array([0.75, 0.75, 0.75, 0.75]),
        "REL1": np.array([0.60, 0.60, 0.60, 0.60]),
        "GOV1": np.array([0.80, 0.80, 0.80, 0.80]),
        "GOV2": np.array([0, 0, 0, 0]),
        "EDU1": np.array([0.60, 0.60, 0.60, 0.60]),
        "EDU2": np.array([0.60, 0.60, 0.60, 0.60])
    }
    def get_factors(self, occupancy_type: str) -> Tuple[float, float, float, float]:
        """
        Retrieve the structural repair costs for a specific occupancy type.

        :param occupancy_type: The occupancy type code (e.g., 'COM1', 'IND3')
        :return: Tuple containing slight, moderate, extensive, and complete repair costs
        :raises KeyError: If the occupancy type is not found in the repair cost data
        """
        if occupancy_type[:4] == "RES1":
            return self.RECAPTURE_FACTORS["RES1"]
        elif occupancy_type[:4] == "RES3":
            return self.RECAPTURE_FACTORS["RES3"]
        elif occupancy_type not in self.RECAPTURE_FACTORS:
            raise KeyError(f"{occupancy_type} not found in RECAPTURE_FACTORS")
        return self.RECAPTURE_FACTORS[occupancy_type]

class AnnualGrossSalary:
    """
    Table 6-11 Annual Gross Salary
    """
    SALARIES: Dict[str, Tuple[float, int, float]] = {
        # (Occupancy Type): (Total Income, Sqft Floor Space, Annual Sales)
        "COM1": (603863, 825, 732),
        "COM2": (367681, 900, 409),
        "IND1": (390894, 550, 711),
        "IND2": (286005, 590, 485),
        "IND3": (752420, 540, 1393),
        "IND4": (707640, 730, 969),
        "IND5": (357170, 300, 1191),
        "IND6": (240994, 250, 964),
        "AGR1": (327606, 250, 1310),
    }

class FragilityBuildingPGA_low_code:
    """
    Compact storage and lookup of fragility data with mean and standard deviations
    for different structure types and damage states.

    Structure:
    {
        occtype: {
            damage_state: (median_pga, std_dev_pga)
        }
    }
    """

    DISTRIBUTIONS: Dict[str, Dict[int, Tuple[float, float]]] = {
        "W1": {1: (0.20, 0.64), 2: (0.34, 0.64), 3: (0.61, 0.64), 4: (0.95, 0.64)},
        "W2": {1: (0.14, 0.64), 2: (0.23, 0.64), 3: (0.48, 0.64), 4: (0.75, 0.64)},
        "S1L": {1: (0.12, 0.64), 2: (0.17, 0.64), 3: (0.30, 0.64), 4: (0.48, 0.64)},
        "S1M": {1: (0.12, 0.64), 2: (0.18, 0.64), 3: (0.29, 0.64), 4: (0.49, 0.64)},
        "S1H": {1: (0.10, 0.64), 2: (0.15, 0.64), 3: (0.28, 0.64), 4: (0.48, 0.64)},
        "S2L": {1: (0.13, 0.64), 2: (0.17, 0.64), 3: (0.30, 0.64), 4: (0.50, 0.64)},
        "S2M": {1: (0.12, 0.64), 2: (0.18, 0.64), 3: (0.35, 0.64), 4: (0.58, 0.64)},
        "S2H": {1: (0.11, 0.64), 2: (0.17, 0.64), 3: (0.36, 0.64), 4: (0.63, 0.64)},
        "S3": {1: (0.10, 0.64), 2: (0.13, 0.64), 3: (0.20, 0.64), 4: (0.38, 0.64)},
        "S4L": {1: (0.13, 0.64), 2: (0.16, 0.64), 3: (0.26, 0.64), 4: (0.46, 0.64)},
        "S4M": {1: (0.12, 0.64), 2: (0.17, 0.64), 3: (0.31, 0.64), 4: (0.54, 0.64)},
        "S4H": {1: (0.12, 0.64), 2: (0.17, 0.64), 3: (0.33, 0.64), 4: (0.59, 0.64)},
        "S5L": {1: (0.13, 0.64), 2: (0.17, 0.64), 3: (0.28, 0.64), 4: (0.45, 0.64)},
        "S5M": {1: (0.11, 0.64), 2: (0.18, 0.64), 3: (0.34, 0.64), 4: (0.53, 0.64)},
        "S5H": {1: (0.10, 0.64), 2: (0.18, 0.64), 3: (0.35, 0.64), 4: (0.58, 0.64)},
        "C1L": {1: (0.12, 0.64), 2: (0.15, 0.64), 3: (0.27, 0.64), 4: (0.45, 0.64)},
        "C1M": {1: (0.12, 0.64), 2: (0.17, 0.64), 3: (0.32, 0.64), 4: (0.54, 0.64)},
        "C1H": {1: (0.10, 0.64), 2: (0.15, 0.64), 3: (0.27, 0.64), 4: (0.44, 0.64)},
        "C2L": {1: (0.14, 0.64), 2: (0.19, 0.64), 3: (0.30, 0.64), 4: (0.52, 0.64)},
        "C2M": {1: (0.12, 0.64), 2: (0.19, 0.64), 3: (0.38, 0.64), 4: (0.63, 0.64)},
        "C2H": {1: (0.11, 0.64), 2: (0.19, 0.64), 3: (0.38, 0.64), 4: (0.65, 0.64)},
        "C3L": {1: (0.12, 0.64), 2: (0.17, 0.64), 3: (0.26, 0.64), 4: (0.44, 0.64)},
        "C3M": {1: (0.11, 0.64), 2: (0.17, 0.64), 3: (0.32, 0.64), 4: (0.51, 0.64)},
        "C3H": {1: (0.09, 0.64), 2: (0.16, 0.64), 3: (0.33, 0.64), 4: (0.53, 0.64)},
        "PC1": {1: (0.13, 0.64), 2: (0.17, 0.64), 3: (0.25, 0.64), 4: (0.45, 0.64)},
        "PC2L": {1: (0.13, 0.64), 2: (0.15, 0.64), 3: (0.24, 0.64), 4: (0.44, 0.64)},
        "PC2M": {1: (0.11, 0.64), 2: (0.16, 0.64), 3: (0.31, 0.64), 4: (0.52, 0.64)},
        "PC2H": {1: (0.11, 0.64), 2: (0.16, 0.64), 3: (0.31, 0.64), 4: (0.55, 0.64)},
        "RM1L": {1: (0.16, 0.64), 2: (0.20, 0.64), 3: (0.29, 0.64), 4: (0.54, 0.64)},
        "RM1M": {1: (0.14, 0.64), 2: (0.19, 0.64), 3: (0.35, 0.64), 4: (0.63, 0.64)},
        "RM2L": {1: (0.14, 0.64), 2: (0.18, 0.64), 3: (0.28, 0.64), 4: (0.51, 0.64)},
        "RM2M": {1: (0.12, 0.64), 2: (0.17, 0.64), 3: (0.34, 0.64), 4: (0.60, 0.64)},
        "RM2H": {1: (0.11, 0.64), 2: (0.17, 0.64), 3: (0.35, 0.64), 4: (0.62, 0.64)},
        "URML": {1: (0.14, 0.64), 2: (0.20, 0.64), 3: (0.32, 0.64), 4: (0.46, 0.64)},
        "URMM": {1: (0.10, 0.64), 2: (0.16, 0.64), 3: (0.27, 0.64), 4: (0.46, 0.64)},
        "MH": {1: (0.11, 0.64), 2: (0.18, 0.64), 3: (0.31, 0.64), 4: (0.60, 0.64)}
    }

    def get_distribution(self, str_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Retrieve the repair times for a specific occupancy type.

        :param str_type: The occupancy type code (e.g., 'W1', 'S4L')
        :return: Dictionary of damage states and their mean PGA and beta values
        :raises KeyError: If the str_type type is not found in the distributions
        """
        if str_type not in self.DISTRIBUTIONS:
            raise KeyError(f"{str_type} not found in DISTRIBUTIONS")
        return self.DISTRIBUTIONS[str_type]

class DebrisUnitWeight:
    """
        Table 10-1
    """
    UNIT_WEIGHTS = {
        ## unit weight in tons per 1000 sqft
        'W1': np.array([6.5, 12.1, 15.0, 0.0]),
        'W2': np.array([4.0, 8.1, 15.0, 1.0]),
        'S1L': np.array([0.0, 5.3, 44.0, 5.0]),
        'S1M': np.array([0.0, 5.3, 44.0, 5.0]),
        'S1H': np.array([0.0, 5.3, 44.0, 5.0]),
        'S2L': np.array([0.0, 5.3, 44.0, 5.0]),
        'S2M': np.array([0.0, 5.3, 44.0, 5.0]),
        'S2H': np.array([0.0, 5.3, 44.0, 5.0]),
        'S3': np.array([0.0, 0.0, 67.0, 1.5]),
        'S4L': np.array([0.0, 5.3, 65.0, 4.0]),
        'S4M': np.array([0.0, 5.3, 65.0, 4.0]),
        'S4H': np.array([0.0, 5.3, 65.0, 4.0]),
        'S5L': np.array([20.0, 5.3, 45.0, 4.0]),
        'S5M': np.array([20.0, 5.3, 45.0, 4.0]),
        'S5H': np.array([20.0, 5.3, 45.0, 4.0]),
        'C1L': np.array([0.0, 5.3, 98.0, 4.0]),
        'C1M': np.array([0.0, 5.3, 98.0, 4.0]),
        'C1H': np.array([0.0, 5.3, 98.0, 4.0]),
        'C2L': np.array([0.0, 5.3, 112.0, 4.0]),
        'C2M': np.array([0.0, 5.3, 112.0, 4.0]),
        'C2H': np.array([0.0, 5.3, 112.0, 4.0]),
        'C3L': np.array([20.0, 5.3, 90.0, 4.0]),
        'C3M': np.array([20.0, 5.3, 90.0, 4.0]),
        'C3H': np.array([20.0, 5.3, 90.0, 4.0]),
        'PC1': np.array([5.5, 5.3, 40.0, 1.5]),
        'PC2L': np.array([0.0, 5.3, 100.0, 4.0]),
        'PC2M': np.array([0.0, 5.3, 100.0, 4.0]),
        'PC2H': np.array([0.0, 5.3, 100.0, 4.0]),
        'RM1L': np.array([17.5, 5.3, 28.0, 4.0]),
        'RM1M': np.array([17.5, 5.3, 28.0, 4.0]),
        'RM2L': np.array([17.5, 5.3, 78.0, 4.0]),
        'RM2M': np.array([24.5, 5.3, 78.0, 4.0]),
        'RM2H': np.array([24.5, 5.3, 78.0, 4.0]),
        'URML': np.array([35.0, 10.5, 41.0, 4.0]),
        'URMM': np.array([35.0, 10.5, 41.0, 4.0]),
        'MH': np.array([10.0, 18.0, 22.0, 0.0])
    }

    def get_unit_weight(
        self,
        str_type: str
    ) -> float:
        return np.sum(self.UNIT_WEIGHTS[str_type])

