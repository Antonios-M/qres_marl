from dataclasses import dataclass, field
from typing import Final, Tuple, Dict
import math
from .utils import DamageStates
import numpy as np

@dataclass (frozen=True)
class OSMRoadSchema:  ## New schema for OSM road-related data
    OSM_ID: Final[str] = 'osm_id'
    HIGHWAY: Final[str] = 'highway'
    NAME: Final[str] = 'name'
    GEOMETRY: Final[str] = 'geometry'
    LANES: Final[str] = 'lanes'
    MAXSPEED: Final[str] = 'maxspeed'
    ONEWAY: Final[str] = 'oneway'
    SURFACE: Final[str] = 'surface'

@dataclass (frozen=True)
class INCORERoadSchema:
    HIGHWAY: Final[str] = 'highway'  # Type of highway or road segment
    UNIT_COST: Final[str] = 'unit_cost'  # Cost associated with the unit of measure for the road segment
    REPL_COST: Final[str] = 'repl_cost'  # Replacement cost of the road segment
    LINKNWID: Final[str] = 'linknwid'  # Unique identifier for the road segment
    FROMNODE: Final[str] = 'fromnode'  # Identifier for the starting node of the road segment
    TONODE: Final[str] = 'tonode'  # Identifier for the ending node of the road segment
    DIRECTION: Final[str] = 'direction'  # Directed or not (Integer)
    LEN_MILE: Final[str] = 'len_mile'  # Length of the road segment (Miles)
    LEN_KM: Final[str] = 'len_km'  # Length of the road segment (Kilometers)
    GUID: Final[str] = 'guid'  # Unique identifier
    FNODE_GUID: Final[str] = 'fnode_guid'  # GUID of the starting node of the road segment
    TNODE_GUID: Final[str] = 'tnode_guid'  # GUID of the ending node of the road segment

@dataclass (frozen=True)
class StudyRoadSchema(INCORERoadSchema):
    HAZUS_ROAD_CLASS: Final[str] = 'hazus_r'  # Hazus road class
    HAZUS_BRIDGE_CLASS: Final[str] = 'hazus_b'
    PGD: Final[str] = 'PGD'  # Peak ground displacement
    PGA: Final[str] = 'PGA'  # Peak ground acceleration
    SA03SEC: Final[str] = '0.3 SA'
    SA1SEC: Final[str] = '1.0 SA'
    BRIDGE_SHAPE: Final[str] = 'i_shape'
    SKEW_ANGLE: Final[str] = 'skew_angle'
    NUM_SPANS: Final[str] = 'num_spans'
    K3D_A: Final[str] = 'K3D_A'
    K3D_B: Final[str] = 'K3D_B'
    BRIDGE_ID: Final[str] = 'bridge_id' # id from NBI csv  if road is bridge, else -1
    DAMAGE_STATE: Final[str] = 'damage_state'
    INIT_REPAIR_TIME: Final[str] = 'init_repair_time'
    CURR_REPAIR_TIME: Final[str] = 'curr_repair_time'
    ACCESSIBLE: Final[str] = 'accessible_repair'
    WIDTH: Final[str] = 'width'
    CAPACITY_RED_DEBRIS: Final[str] = 'cap_red_deb'
    CAPACITY_RED_DS: Final[str] = 'cap_red_ds'
    CAPACITY_REDUCTION: Final[str] = 'capacity_red'
    TRAFFIC_LINK_INDEX: Final[str] = 'traffic_idx'



class OSMHazusRoadMapper:
    HAZUS_ROAD_TYPES = {
        'motorway': 'HRD1',
        'trunk': 'HRD1',
        'primary': 'HRD1',
        'motorway_link': 'HRD1',
        'primary_link': 'HRD1',
        'trunk_link': 'HRD1'
    }

    HAZUS_ROAD_WIDTHS = {
        'HRD1': 12,
        'HRD2': 6
    }

    @classmethod
    def get_hazus_road_type(cls, osm_highway):
        """
        Convert OSM highway type to HAZUS road type.

        Args:
            osm_highway (str): OSM highway type

        Returns:
            str: Corresponding HAZUS road type
        """
        return cls.HAZUS_ROAD_TYPES.get(osm_highway, 'HRD2')
    ## usage:
    @classmethod
    def get_hazus_road_width(cls, hazus_road_class):
        """
        Get the width of a road based on its HAZUS class.

        Args:
            hazus_road_class (str): HAZUS road class

        Returns:
            float: Road width
        """
        return cls.HAZUS_ROAD_WIDTHS.get(hazus_road_class)


@dataclass (frozen=True)
class FHWARoadReplacementCosts:
    """
    Appendix A-1 from https://www.fhwa.dot.gov/policy/23cpr/appendixa.cfm
    """
    osm_highway_type: str

    def get_costs_per_mile(self) -> dict:
        cost_map = {
            'motorway': {'reconstruct': 7675, 'resurface': 1483},           ## Urban -> Major Urbanized
            'motorway_link': {'reconstruct': 7675, 'resurface': 1483},      ## Urban -> Major Urbanized
            'primary': {'reconstruct': 5857, 'resurface': 1135},            ## Other Principal Arterial -> Major Urbanized
            'primary_link': {'reconstruct': 5857, 'resurface': 1135},
            'secondary': {'reconstruct': 2929, 'resurface': 703},
            'secondary_link': {'reconstruct': 7675, 'resurface': 703},
            'tertiaty': {'reconstruct': 1998, 'resurface': 559},
            'residential': {'reconstruct': 1998, 'resurface': 559},
            'tertiary_link': {'reconstruct': 1998, 'resurface': 559},
            'unclassified': {'reconstruct': 1491, 'resurface': 346},
            'service': {'reconstruct': 1491, 'resurface': 346}
        }
        default_costs = {'reconstruct': 2465, 'resurface': 631}
        return cost_map.get(self.osm_highway_type, default_costs)


class RoadReplacementCosts:
    COSTS: Dict = {
        'HRD1': {'reconstruct': 7675, 'resurface': 1483},
        'HRD2': {'reconstruct': 2929, 'resurface': 703},
    }
    def get_costs(self, hazus_road_class: str):
        return self.COSTS[hazus_road_class]



class RoadRepairDistributions:
    """
    First Part of Table 7-3 from https://www.fema.gov/media-library-data/20130726-1913-25045-4394/hazus_eq_model_technical_manual_2_1.pdf
    Distribution type is normal
    """

    DISTRIBUTIONS: Dict[str, Tuple[float, float]] = {
        0: (0.0, 0.0),
        1: (0.9, 0.05),
        2: (2.2, 1.8),
        3: (21, 2),
        4: (21, 5),
    }
    def get_distribution(self, damage_state: str) -> Tuple[float, float]:
        """
        Retrieve the repair times for a specific damage state.

        Args:
            damage_state (str): The damage state

        Returns:
            Tuple[float, float]: The mean and standard deviation for the repair time distribution
        """
        if damage_state not in self.DISTRIBUTIONS:
            raise KeyError(f"{damage_state} not found in DISTRIBUTIONS")
        return self.DISTRIBUTIONS[damage_state]

    @staticmethod
    def compute_repair_time_bins():
        """Compute min/max repair time bins based on ±3 standard deviations."""
        means, stds = zip(*RoadRepairDistributions.DISTRIBUTIONS.values())

        mean_time = np.mean(means)
        std_time = max(stds)  # Use the highest standard deviation

        # Compute min/max repair time using ±3 standard deviations
        min_time = max(0, mean_time - 3 * std_time)
        max_time = mean_time + 3 * std_time

        min_rt = math.ceil(min_time)
        max_rt = math.ceil(max_time)


        return 0, max_rt

    @staticmethod
    def get_repair_time_bin(repair_time, bins):
        """
        Assigns a repair time to its respective bin using the ceiling method.
        """
        if repair_time == 0:
            return 0  # Separate bin for 0
        return np.searchsorted(bins, repair_time, side="left")

class BridgeRepairDistributions:
    """
    Second Part of Table 7-3 from https://www.fema.gov/media-library-data/20130726-1913-25045-4394/hazus_eq_model_technical_manual_2_1.pdf
    Distribution type is normal
    """
    DISTRIBUTIONS: Dict[str, Tuple[float, float]] = {
        0: (0, 0),
        1: (0.6, 0.6),
        2: (2.5, 2.7),
        3: (75, 20),
        4: (230, 30),
        ## modified dispersion as they were too high and resulted in weird repair times
    }
    def get_distribution(self, damage_state: str) -> Tuple[float, float]:
        """
        Retrieve the repair times for a specific damage state.

        Args:
            damage_state (str): The damage state

        Returns:
            Tuple[float, float]: The mean and standard deviation for the repair time distribution
        """
        if damage_state not in self.DISTRIBUTIONS:
            raise KeyError(f"{damage_state} not found in DISTRIBUTIONS")
        return self.DISTRIBUTIONS[damage_state]

    @staticmethod
    def compute_repair_time_bins():
        """Compute min/max repair time bins based on ±3 standard deviations."""
        means, stds = zip(*BridgeRepairDistributions.DISTRIBUTIONS.values())

        mean_time = np.mean(means)
        std_time = max(stds)  # Use the highest standard deviation

        # Compute min/max repair time using ±3 standard deviations
        min_time = max(0, mean_time - 3 * std_time)
        max_time = mean_time + 3 * std_time

        min_rt = math.ceil(min_time)
        max_rt = math.ceil(max_time)

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

class BridgeRepairCost:
    COSTS = {
        "HWB1": 636,
        "HWB2": 583,
        "HWB3": 424,
        "HWB4": 504,
        "HWB5": 398,
        "HWB6": 398,
        "HWB7": 504,
        "HWB8": 318,
        "HWB9": 424,
        "HWB10": 292,
        "HWB11": 318,
        "HWB12": 583,
        "HWB13": 583,
        "HWB14": 742,
        "HWB15": 583,
        "HWB16": 742,
        "HWB17": 398,
        "HWB18": 398,
        "HWB19": 504,
        "HWB20": 398,
        "HWB21": 504,
        "HWB22": 371,
        "HWB23": 424,
        "HWB24": 583,
        "HWB25": 636,
        "HWB26": 795,
        "HWB27": 795,
        "HWB28": 318
    }

    def get_cost(
        self,
        hazus_bridge_class
    ):
        return self.COSTS[hazus_bridge_class]




class FragilityRoadPGD:
    """
    Table 7-5 from https://www.fema.gov/media-library-data/20130726-1913-25045-4394/hazus_eq_model_technical_manual_2_1.pdf
    """
    DISTRIBUTIONS: Dict[
        str, Dict[
            int, Tuple[float, float]
        ]] =  {
        "HRD1": {
            1: (12, 0.7),
            2: (24, 0.7),
            3: (60, 0.7),
            4: (60, 0.7)
        },
        'HRD2': {
            1: (6, 0.7),
            2: (12, 0.7),
            3: (24, 0.7),
            4: (24, 0.7)
        }

    }

    def get_distribution(self, road_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Retrieve the repair times for a specific occupancy type.

        :param str_type: The occupancy type code (e.g., 'W1', 'S4L')
        :return: Dictionary of damage states and their mean PGA and beta values
        :raises KeyError: If the str_type type is not found in the distributions
        """
        if road_type not in self.DISTRIBUTIONS:
            raise KeyError(f"{road_type} not found in DISTRIBUTIONS")
        return self.DISTRIBUTIONS[road_type]

@dataclass (frozen=True)
class NBISchema:
    """
    National Bridge Database (NBI) Coding Guide from https://www.fhwa.dot.gov/bridge/mtguide.pdf
    """

    # Hazus bridge fragility classes are a mapping from NBI attributes:
    # Refer to Hazus 6.0 Inventory Manual Table 9-6 :
    # https://www.fema.gov/sites/default/files/documents/fema_hazus-6-inventory-technical-manual.pdf


    # year built
    YEAR_BUILT: Final[str] = 'YEAR_BUILT'
    # Structural Attributes:
    # from https://www.fhwa.dot.gov/bridge/mtguide.pdf
    """
    Item 43A and 43B - Structure Type, Main                               3 digits

    Record the description on the inspection form and indicate the type of
    structure for the main span(s) with a 3-digit code composed of 2
    segments.

    Segment          Description                             Length

        43A            Kind of material and/or design          1 digit
        43B            Type of design and/or construction      2 digits
    """
    STRUCTURE_1: Final[str] = "STRUCTUR_1"
    STRUCTURE_2: Final[str] = "STRUCTUR_2"

    # Geometric Attributes
    # from https://www.fhwa.dot.gov/bridge/mtguide.pdf
    """
        Item 48 - Length of Maximum Span (xxxx.x meters)             5 digits

    The length of the maximum span shall be recorded.  It shall be noted
    whether the measurement is center to center of bearing points or clear
    open distance between piers, bents, or abutments.  The measurement shall
    be along the centerline of the bridge.  For this item, code a 5-digit
    number to represent the measurement to the nearest tenth of a meter
    (with an assumed decimal point).

    EXAMPLES:                               |   Code
    ________________________________________|_________
    Length of Maximum Span: 35.5 meters     |  00355
                            117.0 meters    |  01170
                            1219.2 meters   |  12192
    """
    LENGTH_MAX_SPAN: Final[str] = "MAX_SPAN_L"
    NUM_SPANS: Final[str] = 'MAIN_UNIT_'
    """
    Item 52 - Deck Width, Out-to-Out (xxx.x meters)              4 digits

    Record and code a 4-digit number to show the out-to-out width to the
    nearest tenth of a meter (with an assumed decimal point).  If the
    structure is a through structure, the number to be coded will represent
    the lateral clearance between superstructure members.  The measurement
    should be exclusive of flared areas for ramps.  See examples on pages 30
    and 31.

    Where traffic runs directly on the top slab (or wearing surface) of the
    culvert (e.g., an R/C box without fill) code the actual width
    (out-to-out).  This will also apply where the fill is minimal and the
    culvert headwalls affect the flow of traffic.  However, for sidehill
    viaduct structures code the actual out-to-out structure width.  See
    figure in the Commentary Appendix D.

    Where the roadway is on a fill carried across a pipe or box culvert and
    the culvert headwalls do not affect the flow of traffic, code 0000.
    This is considered proper inasmuch as a filled section over a culvert
    simply maintains the roadway cross-section.

    """
    WIDTH_END_TO_END: Final[str] = "APPR_WIDTH"

    """
    Item 34 - Skew (XX degrees) 2 digits

    The skew angle is the angle between the centerline of a pier and a line
    normal to the roadway centerline. When plans are available, the skew
    angle can be taken directly from the plans. If no plans are available,
    the angle is to be field measured if possible. Record the skew angle to
    the nearest degree. If the skew angle is 0E, it should be so coded.
    When the structure is on a curve or if the skew varies for some other
    reason, the average skew should be recorded, if reasonable. Otherwise,
    record 99 to indicate a major variation in skews of substructure units.
    A 2-digit number should be coded.
    """
    SKEW_ANGLE: Final[str] = "DEGREES_SK"

@dataclass (frozen=True)
class FragilityBridgeSA_PGD:
    """
    Compact storage and lookup of fragility data with mean values for Sa (1.0 sec in g's)
    and PGD (inches) for different bridge types and damage states.

    Structure:
    {
        bridge_type: {
            damage_state: {
                "Sa": median_sa 1.0 sec,
                "PGD": median_pgd
            }
        }
    }
    """
    SA_DISP: float = 0.6
    MEDIANS: Dict[str, Dict[int, tuple]] = field(default_factory=lambda: {
        "HWB1": {1: (0.40, 3.9), 2: (0.50, 3.9), 3: (0.70, 3.9), 4: (0.90, 13.8)},
        "HWB2": {1: (0.60, 3.9), 2: (0.90, 3.9), 3: (1.10, 3.9), 4: (1.70, 13.8)},
        "HWB3": {1: (0.80, 3.9), 2: (1.00, 3.9), 3: (1.20, 3.9), 4: (1.70, 13.8)},
        "HWB4": {1: (0.80, 3.9), 2: (1.00, 3.9), 3: (1.20, 3.9), 4: (1.70, 13.8)},
        "HWB5": {1: (0.25, 3.9), 2: (0.35, 3.9), 3: (0.45, 3.9), 4: (0.70, 13.8)},
        "HWB6": {1: (0.30, 3.9), 2: (0.50, 3.9), 3: (0.60, 3.9), 4: (0.90, 13.8)},
        "HWB7": {1: (0.50, 3.9), 2: (0.80, 3.9), 3: (1.10, 3.9), 4: (1.70, 13.8)},
        "HWB8": {1: (0.35, 3.9), 2: (0.45, 3.9), 3: (0.55, 3.9), 4: (0.80, 13.8)},
        "HWB9": {1: (0.60, 3.9), 2: (0.90, 3.9), 3: (1.30, 3.9), 4: (1.60, 13.8)},
        "HWB10": {1: (0.60, 3.9), 2: (0.90, 3.9), 3: (1.10, 3.9), 4: (1.50, 13.8)},
        "HWB11": {1: (0.90, 3.9), 2: (0.90, 3.9), 3: (1.10, 3.9), 4: (1.50, 13.8)},
        "HWB12": {1: (0.25, 3.9), 2: (0.35, 3.9), 3: (0.45, 3.9), 4: (0.70, 13.8)},
        "HWB13": {1: (0.30, 3.9), 2: (0.50, 3.9), 3: (0.60, 3.9), 4: (0.90, 13.8)},
        "HWB14": {1: (0.50, 3.9), 2: (0.80, 3.9), 3: (1.10, 3.9), 4: (1.70, 13.8)},
        "HWB15": {1: (0.75, 3.9), 2: (0.75, 3.9), 3: (0.75, 3.9), 4: (1.10, 13.8)},
        "HWB16": {1: (0.90, 3.9), 2: (0.90, 3.9), 3: (1.10, 3.9), 4: (1.50, 13.8)},
        "HWB17": {1: (0.25, 3.9), 2: (0.35, 3.9), 3: (0.45, 3.9), 4: (0.70, 13.8)},
        "HWB18": {1: (0.30, 3.9), 2: (0.50, 3.9), 3: (0.60, 3.9), 4: (0.90, 13.8)},
        "HWB19": {1: (0.50, 3.9), 2: (0.80, 3.9), 3: (1.10, 3.9), 4: (1.70, 13.8)},
        "HWB20": {1: (0.35, 3.9), 2: (0.45, 3.9), 3: (0.55, 3.9), 4: (0.80, 13.8)},
        "HWB21": {1: (0.60, 3.9), 2: (0.90, 3.9), 3: (1.30, 3.9), 4: (1.60, 13.8)},
        "HWB22": {1: (0.60, 3.9), 2: (0.90, 3.9), 3: (1.10, 3.9), 4: (1.50, 13.8)},
        "HWB23": {1: (0.90, 3.9), 2: (0.90, 3.9), 3: (1.10, 3.9), 4: (1.50, 13.8)},
        "HWB24": {1: (0.25, 3.9), 2: (0.35, 3.9), 3: (0.45, 3.9), 4: (0.70, 13.8)},
        "HWB25": {1: (0.30, 3.9), 2: (0.50, 3.9), 3: (0.60, 3.9), 4: (0.90, 13.8)},
        "HWB26": {1: (0.75, 3.9), 2: (0.75, 3.9), 3: (0.75, 3.9), 4: (1.10, 13.8)},
        "HWB27": {1: (0.75, 3.9), 2: (0.75, 3.9), 3: (0.75, 3.9), 4: (1.10, 13.8)},
        "HWB28": {1: (0.80, 3.9), 2: (1.00, 3.9), 3: (1.20, 3.9), 4: (1.70, 13.8)},
    })

    def get_medians(self, bridge_type: str) -> Dict[str, Tuple[float, float]]:
        """
        Retrieve the repair times for a specific occupancy type.

        :param str_type: The occupancy type code (e.g., 'W1', 'S4L')
        :return: Dictionary of damage states and their mean PGA and beta values
        :raises KeyError: If the str_type type is not found in the distributions
        """
        if bridge_type not in self.MEDIANS:
            raise KeyError(f"{bridge_type} not found in DISTRIBUTIONS")
        return self.MEDIANS[bridge_type]


class HAZUSBridge_k3d_coefficients:
    """
    Class to retrieve coefficients for K3d calculation based on HWB input.
    """
    coefficients_map = {
        "EQ1": (0.25, 1),
        "EQ2": (0.33, 0),
        "EQ3": (0.33, 1),
        "EQ4": (0.09, 1),
        "EQ5": (0.05, 0),
        "EQ6": (0.20, 1),
        "EQ7": (0.10, 0),
    }

    @staticmethod
    def get_coefficients(hwb_label: str):
        """
        Retrieves the coefficients corresponding to an HWB label.

        Args:
            hwb_label (str): The HWB label, e.g., 'HWB1'.

        Returns:
            Tuple[float, int]: The corresponding equation coefficients.
        """
        try:
            # Extract the numerical part of the HWB label
            hwb_number = int(hwb_label[3:])  # Assumes HWB label starts with 'HWB'
        except (IndexError, ValueError):
            raise ValueError(f"Invalid HWB label format: {hwb_label}")

        if 6 <= hwb_number <= 10:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ2"]
        elif 11 <= hwb_number <= 15:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ3"]
        elif 16 <= hwb_number <= 20:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ4"]
        elif 21 <= hwb_number <= 25:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ5"]
        elif 26 <= hwb_number <= 30:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ6"]
        elif 31 <= hwb_number <= 35:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ7"]
        else:
            return HAZUSBridge_k3d_coefficients.coefficients_map["EQ1"]


class BridgeModificationFactors:
    """
    Utility class for calculating bridge modification factors based on
    HAZUS bridge fragility methodology.

    Reference: FEMA HAZUS Earthquake Model Technical Manual
    """

    @staticmethod
    def calculate_kskew(skew_angle: int) -> float:
        """
        Calculate skew modification factor.

        Args:
            skew_angle (int): Skew angle of the bridge.

        Returns:
            float: Skew modification factor.
        """
        if isinstance(skew_angle, str):
            skew_angle = int(skew_angle)
        return math.sqrt(math.sin(math.radians(90 - skew_angle)))

    @staticmethod
    def calculate_kshape(sa_0_3: float, sa_1_0: float) -> float:
        """
        Calculate shape modification factor.

        Args:
            sa_0_3 (float): Spectral acceleration at 0.3 seconds.
            sa_1_0 (float): Spectral acceleration at 1.0 seconds.

        Returns:
            float: Shape modification factor.
        """
        return min(1, (2.5 * sa_1_0) / sa_0_3)

    @staticmethod
    def calculate_k3d(coefficients: Tuple[float, int], num_spans) -> float:
        """
        Calculate 3D modification factor.

        Args:
            coefficients (Tuple[float, int]): Coefficients for K3d calculation.

        Returns:
            float: 3D modification factor.
        """
        a, b = coefficients

        if isinstance(a, str):
            a = float(a)
        if isinstance(b, str):
            b = float(b)
        if isinstance(num_spans, str):
            num_spans = int(num_spans)

        return (1 + a) / (num_spans - b)


class FragilityBridgeSAModifier:
    """
    Class to modify spectral accelerations for bridge fragility assessment.
    """

    def __init__(
        self,
        bridge_fragility: Dict[int, Tuple[float,float]],
        sa_0_3: float,
        sa_1_0: float,
        bridge_shape: int,
        skew_angle: int,
        num_spans: int,
        k3d_coefficients: Tuple[float, int]
    ) -> None:
        """
        Initialize bridge spectral acceleration modifier.

        Args:
            bridge_fragility (Dict[DamageStates, Dict[str, float]]):
                Fragility data for different damage states.
            sa_0_3 (float): Spectral acceleration at 0.3 seconds.
            sa_1_0 (float): Spectral acceleration at 1.0 seconds.
            bridge_shape (int): Bridge shape identifier.
            skew_angle (int): Bridge skew angle.
            k3d_coefficients (Tuple[float, int]): Coefficients for K3d calculation.
        """
        self.bridge_fragility = bridge_fragility.copy()
        self.sa_0_3 = sa_0_3
        self.sa_1_0 = sa_1_0
        self.bridge_shape = bridge_shape
        self.skew_angle = skew_angle
        self.num_spans = num_spans
        self.k3d_coefficients = k3d_coefficients

    def calculate_slight_modification_factor(self) -> float:
        """
        Calculate modification factor for slight damage state.

        Returns:
            float: Modification factor for slight damage.
        """
        if self.bridge_shape == 0:
            return 1.0

        return BridgeModificationFactors.calculate_kshape(
            self.sa_0_3,
            self.sa_1_0
        )

    def modify_spectral_accelerations(self) -> Dict[int, Tuple[float, float]]:
        """
        Modify spectral accelerations for different damage states.

        Returns:
            Dict[DamageStates, Dict[str, float]]: Modified fragility data.
        """
        # Calculate modification factors
        kskew = BridgeModificationFactors.calculate_kskew(self.skew_angle)
        k3d = BridgeModificationFactors.calculate_k3d(self.k3d_coefficients, self.num_spans)
        slight_factor = self.calculate_slight_modification_factor()

        # Modify spectral accelerations for each damage state
        modifications = {
            DamageStates.SLIGHT.value: slight_factor,
            DamageStates.MODERATE.value: kskew * k3d,
            DamageStates.EXTENSIVE.value: kskew * k3d,
            DamageStates.COMPLETE.value: kskew * k3d
        }

        # Apply modifications to bridge fragility
        # print(self.bridge_fragility)
        for ds in [
            DamageStates.SLIGHT.value,
            DamageStates.MODERATE.value,
            DamageStates.EXTENSIVE.value,
            DamageStates.COMPLETE.value
        ]:
            sa, pgd = self.bridge_fragility[ds]
            sa *= modifications[ds]
            self.bridge_fragility[ds] = (sa, pgd)

        return {
            damage_state: (sa, FragilityBridgeSA_PGD.SA_DISP)
            for damage_state, (sa, pgd) in self.bridge_fragility.items()
        }



# Example usage
# def main():
#     """
#     Example demonstration of bridge spectral acceleration modification.
#     """
#     # Sample bridge fragility data
#     sample_fragility = {
#         DamageStates.Slight: {"Sa": 0.40, "PGD": 3.9},
#         DamageStates.MODERATE: {"Sa": 0.50, "PGD": 3.9},
#         DamageStates.Extensive: {"Sa": 0.70, "PGD": 3.9},
#         DamageStates.Complete: {"Sa": 0.90, "PGD": 13.8}
#     }
#     k3d_coeffs = HAZUSBridge_k3d_coefficients.get_coefficients("HWB1")
#     # Example modification
#     modifier = FragilityBridgeSAModifier(
#         bridge_fragility=sample_fragility,
#         sa_0_3=0.5,
#         sa_1_0=0.3,
#         bridge_shape=1,
#         skew_angle=30,
#         num_spans=3,
#         k3d_coefficients=k3d_coeffs
#     )

#     # Modify and print results
#     modified_fragility = modifier.modify_spectral_accelerations()
#     print(modified_fragility.values())
#     print(sum([x[0] for x in list(modified_fragility.values())]))
#     print("Modified Fragility:", modified_fragility)


# if __name__ == "__main__":
#     main()


