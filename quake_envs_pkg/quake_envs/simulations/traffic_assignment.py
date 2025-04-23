import heapq
import math
import time
from typing import Tuple

import networkx as nx
import scipy
import geopandas as gpd
from shapely import affinity, wkt
from shapely.ops import transform
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString, Point
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import seaborn as sns

from .traffic_network_import import *
from .utils import PathUtils

from .road_config import *
from .building_config import *
from .road_funcs import *
from .building_funcs import *

"""
Citation:
Everything up to TrafficAccessor is authored by Matteo Bettini (see below) and TrafficAccessor acts as a wrapper

@misc{Bettini2021Traffic,
    author =       {Matteo Bettini},
    title =        {Static traffic assignment using user equilibrium and system optimum},
    howpublished = {GitHub},
    year =         {2021},
    url =          {https://github.com/MatteoBettini/Traffic-Assignment-Frank-Wolfe-2021}
}
"""
class FlowTransportNetwork:

    def __init__(self):
        self.linkSet = {}
        self.nodeSet = {}

        self.tripSet = {}
        self.zoneSet = {}
        self.originZones = {}

        self.networkx_graph = None

    def to_networkx(self):
        if self.networkx_graph is None:
            self.networkx_graph = nx.DiGraph([(int(begin),int(end)) for (begin,end) in self.linkSet.keys()])
        return self.networkx_graph

    def reset_flow(self):
        for link in self.linkSet.values():
            link.reset_flow()

    def reset(self):
        for link in self.linkSet.values():
            link.reset()


class Zone:
    def __init__(self, zoneId: str):
        self.zoneId = zoneId

        self.lat = 0
        self.lon = 0
        self.destList = []  # list of zone ids (strs)


class Node:
    """
    This class has attributes associated with any node
    """

    def __init__(self, nodeId: str):
        self.Id = nodeId

        self.lat = 0
        self.lon = 0

        self.outLinks = []  # list of node ids (strs)
        self.inLinks = []  # list of node ids (strs)

        # For Dijkstra
        self.label = np.inf
        self.pred = None


class Link:
    """
    This class has attributes associated with any link
    """

    def __init__(self,
                idx: np.int32,
                init_node: str,
                term_node: str,
                capacity: float,
                length: float,
                fft: float,
                b: float,
                power: float,
                speed_limit: float,
                toll: float,
                linkType
                ):
        self.idx = idx
        self.init_node = init_node
        self.term_node = term_node
        self.max_capacity = float(capacity)  # veh per hour
        self.length = float(length)  # Length
        self.fft = float(fft)  # Free flow travel time (min)
        self.beta = float(power)
        self.alpha = float(b)
        self.speedLimit = float(speed_limit)
        self.toll = float(toll)
        self.linkType = linkType

        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.flow = 0.0
        self.cost = self.fft

    # Method not used for assignment
    def modify_capacity(self, delta_percentage: float):
        assert -1 <= delta_percentage <= 1
        self.curr_capacity_percentage += delta_percentage
        self.curr_capacity_percentage = max(0, min(1, self.curr_capacity_percentage))
        self.capacity = self.max_capacity * self.curr_capacity_percentage

    def modify_base_capacity(self, delta_percentage: float):
        assert -1 <= delta_percentage <= 1
        self.curr_capacity_percentage = delta_percentage
        # print("-----")
        # print(delta_percentage)
        # print(f"max capacity {self.max_capacity}")
        # print(self.capacity)
        self.capacity = max(0, self.max_capacity + (self.max_capacity * delta_percentage))
        # print(self.capacity)

    def reset(self):
        self.curr_capacity_percentage = 1
        self.capacity = self.max_capacity
        self.reset_flow()

    def reset_flow(self):
        self.flow = 0.0
        self.cost = self.fft


class Demand:
    def __init__(self,
                init_node: str,
                term_node: str,
                demand: float
                ):
        self.fromZone = init_node
        self.toNode = term_node
        self.demand = float(demand)


def DijkstraHeap(origin, network: FlowTransportNetwork):
    """
    Calcualtes shortest path from an origin to all other destinations.
    The labels and preds are stored in node instances.
    """
    for n in network.nodeSet:
        network.nodeSet[n].label = np.inf
        network.nodeSet[n].pred = None
    network.nodeSet[origin].label = 0.0
    network.nodeSet[origin].pred = None
    SE = [(0, origin)]
    while SE:
        currentNode = heapq.heappop(SE)[1]
        currentLabel = network.nodeSet[currentNode].label
        for toNode in network.nodeSet[currentNode].outLinks:
            link = (currentNode, toNode)
            newNode = toNode
            newPred = currentNode
            existingLabel = network.nodeSet[newNode].label
            # print(network.linkSet[link].cost) # prints cost of link

            newLabel = currentLabel + network.linkSet[link].cost
            # print( 'new label : ' + str(newLabel) + ', existing label: ' + str(existingLabel))
            if newLabel < existingLabel:
                heapq.heappush(SE, (newLabel, newNode))
                network.nodeSet[newNode].label = newLabel
                network.nodeSet[newNode].pred = newPred


def BPRcostFunction(optimal: bool,
                    fft: float,
                    alpha: float,
                    flow: float,
                    capacity: float,
                    beta: float,
                    length: float,
                    maxSpeed: float
                    ) -> float:
    if capacity < 1e-3: ## avoid infinite costs for very low capacities
        return 60 * 8 ## TODO change this
    if optimal:
        return fft * (1 + (alpha * math.pow((flow * 1.0 / capacity), beta)) * (beta + 1))
    return fft * (1 + alpha * math.pow((flow * 1.0 / capacity), beta))


def constantCostFunction(optimal: bool,
                        fft: float,
                        alpha: float,
                        flow: float,
                        capacity: float,
                        beta: float,
                        length: float,
                        maxSpeed: float
                        ) -> float:
    if optimal:
        return fft + flow
    return fft


def greenshieldsCostFunction(optimal: bool,
                            fft: float,
                            alpha: float,
                            flow: float,
                            capacity: float,
                            beta: float,
                            length: float,
                            maxSpeed: float
                            ) -> float:
    if capacity < 1e-3:
        return np.finfo(np.float32).max
    if optimal:
        return (length * (capacity ** 2)) / (maxSpeed * (capacity - flow) ** 2)
    return length / (maxSpeed * (1 - (flow / capacity)))


def updateTravelTime(network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction):
    """
    This method updates the travel time on the links with the current flow
    """
    for l in network.linkSet:
        network.linkSet[l].cost = costFunction(optimal,
                                            network.linkSet[l].fft,
                                            network.linkSet[l].alpha,
                                            network.linkSet[l].flow,
                                            network.linkSet[l].capacity,
                                            network.linkSet[l].beta,
                                            network.linkSet[l].length,
                                            network.linkSet[l].speedLimit
                                            )
    # print("\n checkcapacity_1_updateTravelTIme_run_time_capacities" + str([network.linkSet[l].capacity for l in network.linkSet]))


def findAlpha(x_bar, network: FlowTransportNetwork, optimal: bool = False, costFunction=BPRcostFunction):
    """
    This uses unconstrained optimization to calculate the optimal step size required
    for the Frank-Wolfe Algorithm
    """

    def df(alpha):
        assert 0 <= alpha <= 1
        sum_derivative = 0
        for l in network.linkSet:
            tmpFlow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow
            tmpCost = costFunction(optimal,
                        network.linkSet[l].fft,
                        network.linkSet[l].alpha,
                        tmpFlow,
                        network.linkSet[l].capacity,
                        network.linkSet[l].beta,
                        network.linkSet[l].length,
                        network.linkSet[l].speedLimit
            )
            sum_derivative += (x_bar[l] - network.linkSet[l].flow) * tmpCost
        return sum_derivative  # Only return a single scalar value

    # Initial checks for the signs of the function at the endpoints of the bracket
    f0 = df(0)
    f1 = df(1)

    if f0 * f1 > 0:
        return 0.1

    sol = scipy.optimize.root_scalar(df, x0=0.5, bracket=(0, 1))


    assert 0 <= sol.root <= 1
    return sol.root



def tracePreds(dest, network: FlowTransportNetwork):
    """
    This method traverses predecessor nodes in order to create a shortest path
    """
    prevNode = network.nodeSet[dest].pred
    spLinks = []
    while prevNode is not None:
        spLinks.append((prevNode, dest))
        dest = prevNode
        prevNode = network.nodeSet[dest].pred
    return spLinks


def loadAON(network: FlowTransportNetwork, computeXbar: bool = True):
    """
    This method produces auxiliary flows for all or nothing loading.
    """
    x_bar = {l: 0.0 for l in network.linkSet}
    SPTT = 0.0
    for r in network.originZones:
        DijkstraHeap(r, network=network)
        for s in network.zoneSet[r].destList:
            dem = network.tripSet[r, s].demand

            if dem <= 0:
                continue

            SPTT = SPTT + network.nodeSet[s].label * dem
            # print("--------Load All or Nothing")
            # print('label' + str(network.nodeSet[s].label))
            # print(network.nodeSet[s].label)
            # print("SPTT: " + str(SPTT) + ', OD: ' + str(r) + ',' + str(s) ) ## check if OD pairs AON has shortest path or not
            if computeXbar and r != s:
                for spLink in tracePreds(s, network):
                    x_bar[spLink] = x_bar[spLink] + dem

    return SPTT, x_bar


def readDemand(demand_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in demand_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        demand = row["demand"]

        network.tripSet[init_node, term_node] = Demand(init_node, term_node, demand)
        if init_node not in network.zoneSet:
            network.zoneSet[init_node] = Zone(init_node)
        if term_node not in network.zoneSet:
            network.zoneSet[term_node] = Zone(term_node)
        if term_node not in network.zoneSet[init_node].destList:
            network.zoneSet[init_node].destList.append(term_node)

    # print(len(network.tripSet), "OD pairs")
    # print(len(network.zoneSet), "OD zones")


def readNetwork(network_df: pd.DataFrame, network: FlowTransportNetwork):
    for index, row in network_df.iterrows():

        init_node = str(int(row["init_node"]))
        term_node = str(int(row["term_node"]))
        capacity = row["capacity"]
        length = row["length"]
        free_flow_time = row["free_flow_time"]
        b = row["b"]
        power = row["power"]
        speed = row["speed"]
        toll = row["toll"]
        link_type = row["link_type"]

        network.linkSet[init_node, term_node] = Link(
            idx=index,
            init_node=init_node,
            term_node=term_node,
            capacity=capacity,
            length=length,
            fft=free_flow_time,
            b=b,
            power=power,
            speed_limit=speed,
            toll=toll,
            linkType=link_type
        )
        if init_node not in network.nodeSet:
            network.nodeSet[init_node] = Node(init_node)
        if term_node not in network.nodeSet:
            network.nodeSet[term_node] = Node(term_node)
        if term_node not in network.nodeSet[init_node].outLinks:
            network.nodeSet[init_node].outLinks.append(term_node)
        if init_node not in network.nodeSet[term_node].inLinks:
            network.nodeSet[term_node].inLinks.append(init_node)

    # print(len(network.nodeSet), "nodes")
    # print(len(network.linkSet), "links")


def get_TSTT(network: FlowTransportNetwork, costFunction=BPRcostFunction, use_max_capacity: bool = True):
    TSTT = round(sum([network.linkSet[a].flow * costFunction(optimal=False,
                                                            fft=network.linkSet[
                                                                a].fft,
                                                            alpha=network.linkSet[
                                                                a].alpha,
                                                            flow=network.linkSet[
                                                                a].flow,
                                                            capacity=network.linkSet[
                                                                a].max_capacity if use_max_capacity else network.linkSet[
                                                                a].capacity,
                                                            beta=network.linkSet[
                                                                a].beta,
                                                            length=network.linkSet[
                                                                a].length,
                                                            maxSpeed=network.linkSet[
                                                                a].speedLimit
                                                            ) for a in
                    network.linkSet]), 9)
    return TSTT


def  assignment_loop(network: FlowTransportNetwork,
                    algorithm: str = "FW",
                    systemOptimal: bool = False,
                    costFunction=BPRcostFunction,
                    accuracy: float = 0.001,
                    maxIter: int = 1000,
                    maxTime: int = 60,
                    verbose: bool = True,
                    return_detailed: bool =False
):
    """
    For explaination of the algorithm see Chapter 7 of:
    https://sboyles.github.io/blubook.html
    PDF:
    https://sboyles.github.io/teaching/ce392c/book.pdf
    """
    network.reset_flow()

    iteration_number = 1
    gap = np.inf
    TSTT = np.inf
    assignmentStartTime = time.time()
    # print(f'gap: {gap}')
    # print(f'accuracy: {accuracy}')
    # Check if desired accuracy is reached
    while gap > accuracy:
        # print(f'algorithm: {algorithm}')
        # Get x_bar throug all-or-nothing assignment
        _, x_bar = loadAON(network=network)

        if algorithm == "MSA" or iteration_number == 1:
            alpha = (1 / iteration_number)
            # print(f'iteration number: {iteration_number}')
        elif algorithm == "FW":
            # If using Frank-Wolfe determine the step size alpha by solving a nonlinear equation

            alpha = findAlpha(x_bar,
                            network=network,
                            optimal=systemOptimal,
                            costFunction=costFunction)
        else:
            print("Terminating the program.....")
            print("The solution algorithm ", algorithm, " does not exist!")
            raise TypeError('Algorithm must be MSA or FW')

        # Apply flow improvement
        for l in network.linkSet:
            network.linkSet[l].flow = alpha * x_bar[l] + (1 - alpha) * network.linkSet[l].flow

        # Compute the new travel time
        updateTravelTime(network=network,
                        optimal=systemOptimal,
                        costFunction=costFunction)

        # Compute the relative gap
        SPTT, _ = loadAON(network=network, computeXbar=False)
        SPTT = round(SPTT, 9)
        TSTT = round(sum([network.linkSet[a].flow * network.linkSet[a].cost for a in
                        network.linkSet]), 9)

        # print("TSTT, SPTT, Max capacity",TSTT, SPTT,  max([l.capacity for l in network.linkSet.values()]))
        gap = (TSTT / SPTT) - 1
        if abs(TSTT - SPTT) > 1e-3 and gap < 0:
            print("Error, gap is less than 0, this should not happen")
            print("TSTT", "SPTT", TSTT, SPTT)

            # Uncomment for debug
            # print("Capacities:", [l.capacity for l in network.linkSet.values()])
            # print("Flows:", [l.flow for l in network.linkSet.values()])

        # Compute the real total travel time (which in the case of system optimal rounting is different from the TSTT above)
        TSTT = get_TSTT(network=network, costFunction=costFunction)

        iteration_number += 1
        if iteration_number > maxIter:
            if verbose:
                print(
                    "The assignment did not converge to the desired gap and the max number of iterations has been reached")
                print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
                print("Current gap:", round(gap, 5))
            return TSTT
        if time.time() - assignmentStartTime > maxTime:
            if verbose:
                print("The assignment did not converge to the desired gap and the max time limit has been reached")
                print("Assignment did ", iteration_number, "iterations")
                print("Current gap:", round(gap, 5))
            return TSTT

    if verbose:
        print("Assignment converged in ", iteration_number, "iterations")
        print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
        print("Current gap:", round(gap, 5))

    if return_detailed:
        traffic_links_results = {
            "E": list(network.linkSet.keys()),
            "flow": [network.linkSet[a].flow * network.linkSet[a].cost for a in network.linkSet],
            "cost": [network.linkSet[a].cost for a in network.linkSet]
        }
        traffic_links_res_df = pd.DataFrame(traffic_links_results).set_index("E")
        # print(traffic_links_res_df)
        return TSTT, traffic_links_res_df

    return TSTT


def writeResults(network: FlowTransportNetwork, output_file: str, costFunction=BPRcostFunction,
                systemOptimal: bool = False, verbose: bool = True):
    outFile = open(output_file, "w")
    TSTT = get_TSTT(network=network, costFunction=costFunction)
    if verbose:
        print("\nTotal system travel time:", f'{TSTT} secs')
    tmpOut = "Total Travel Time:\t" + str(TSTT)
    outFile.write(tmpOut + "\n")
    tmpOut = "Cost function used:\t" + BPRcostFunction.__name__
    outFile.write(tmpOut + "\n")
    tmpOut = ["User equilibrium (UE) or system optimal (SO):\t"] + ["SO" if systemOptimal else "UE"]
    outFile.write("".join(tmpOut) + "\n\n")
    tmpOut = "init_node\tterm_node\tflow\ttravelTime"
    outFile.write(tmpOut + "\n")
    for i in network.linkSet:
        tmpOut = str(network.linkSet[i].init_node) + "\t" + str(
            network.linkSet[i].term_node) + "\t" + str(
            network.linkSet[i].flow) + "\t" + str(costFunction(False,
                                                            network.linkSet[i].fft,
                                                            network.linkSet[i].alpha,
                                                            network.linkSet[i].flow,
                                                            network.linkSet[i].max_capacity,
                                                            network.linkSet[i].beta,
                                                            network.linkSet[i].length,
                                                            network.linkSet[i].speedLimit
                                                            ))
        outFile.write(tmpOut + "\n")
    outFile.close()


def load_network(
                in_net_name: str = None,
                net_file: str = None,
                demand_file: str = None,
                in_net_df: pd.DataFrame = None,
                in_demand_df: pd.DataFrame = None,
                force_net_reprocess: bool = False,
                verbose: bool = True
                ) -> FlowTransportNetwork:
    readStart = time.time()

    if net_file is None and in_demand_df is None or in_net_df is None:
        raise ValueError("Need to provide either net tntp or net and demand pd files as input")

    if demand_file is None and net_file is not None:
        demand_file = '_'.join(net_file.split("_")[:-1] + ["trips.tntp"])
        net_name = net_file.split("/")[-1].split("_")[0]

    if in_net_name is None:
        net_name = "Unknown"
    else:
        net_name = in_net_name


    if verbose:
        print(f"Loading network {net_name}...")

    if in_net_df is None or in_demand_df is None:
        if demand_file is None:
            demand_file = '_'.join(net_file.split("_")[:-1] + ["trips.tntp"])

        net_name = net_file.split("/")[-1].split("_")[0]

        if verbose:
            print(f"Loading network {net_name}...")
        # Only call import_network if defaults are not provided
        net_df, demand_df = import_network(
            net_file,
            demand_file,
            force_reprocess=force_net_reprocess
        )
    else:
        # Use the provided defaults
        net_df, demand_df = in_net_df, in_demand_df

    network = FlowTransportNetwork()

    readDemand(demand_df, network=network)
    readNetwork(net_df, network=network)

    network.originZones = set([k[0] for k in network.tripSet])

    if verbose:
        print("Network", net_name, "loaded")
        print("Reading the network data took", round(time.time() - readStart, 2), "secs\n")

    return network


def computeAssignment(
    net_file: str = None,
    demand_file: str = None,
    in_network: FlowTransportNetwork = None,
    algorithm: str = "FW",  # FW or MSA
    costFunction=BPRcostFunction,
    systemOptimal: bool = False,
    accuracy: float = 0.01,
    maxIter: int = 100000,
    maxTime: int = 6000,
    results_file: str = None,
    force_net_reprocess: bool = False,
    verbose: bool = True,
    write_results: bool = False,
    return_detailed: bool = False
) -> float:
    """
    This function computes the user equilibrium (UE) or system optimal (SO) traffic assignment.
    The network should be in the tntp format, as described at https://github.com/bstabler/TransportationNetworks.

    :param net_file: Path to the network (net) file in tntp format.
    :param demand_file: Path to the demand (trips) file in tntp format, defaults to None.
    :param algorithm: Traffic assignment algorithm:
        - "FW": Frank-Wolfe algorithm.
        - "MSA": Method of successive averages.
    :param costFunction: Cost function used to compute travel time on edges (e.g., BPRcostFunction, greenshieldsCostFunction).
    :param systemOptimal: Whether to compute system optimal flows (default: False).
    :param accuracy: Desired precision gap for assignment.
    :param maxIter: Maximum number of iterations for the algorithm.
    :param maxTime: Maximum time (in seconds) for the assignment.
    :param results_file: Path to save results, defaults to None (will be derived from the network file name).
    :param force_net_reprocess: Force reprocessing of network files, defaults to False.
    :param verbose: Whether to print detailed logs, defaults to True.
    :return: Total system travel time.
    """


    # Load network if not provided
    network = in_network or load_network(
        net_file=net_file,
        demand_file=demand_file,
        verbose=verbose,
        force_net_reprocess=force_net_reprocess
    )

    if verbose:
        print("Computing assignment...")

    # Run assignment algorithm
    res = assignment_loop(
        network=network,
        algorithm=algorithm,
        systemOptimal=systemOptimal,
        costFunction=costFunction,
        accuracy=accuracy,
        maxIter=maxIter,
        maxTime=maxTime,
        verbose=verbose,
        return_detailed=return_detailed
    )
    if type(res) == tuple:
        return res

    TSTT = res

    # Write results to file if specified
    if write_results:
        # Derive default results file name if not provided
        results_file = results_file or '_'.join(net_file.split("_")[:-1] + ["flow.tntp"])

        writeResults(
            network=network,
            output_file=results_file,
            costFunction=costFunction,
            systemOptimal=systemOptimal,
            verbose=verbose
        )
    return TSTT

class TrafficMonetaryValues:
    """
    Source:
        Federal Highway Administration, “Work Zone Road User Costs - Concepts and Applications:
        Chapter 2. Work zone road user costs.” Last accessed Aug 12, 2023:
        https://ops.fhwa.dot.gov/wz/resources/publications/fhwahop12005/sec2.htm.
        Data in this class uses this source unless stated otherwise
    """
    BUSINESS_PERSONAL_RATIOS = np.array([
        [0.958, 0.042],
        [0.95, 0.05],
        [0.942, 0.058],
        [0.919, 0.081],
        [0.937, 0.063]

    ])
    MEAN_BUSINESS_PERSONAL_RATIO = np.mean(BUSINESS_PERSONAL_RATIOS, axis=0)
    STD_BUSINESS_PERSONAL_RATIO = np.std(BUSINESS_PERSONAL_RATIOS, axis=0)

    # TT_COST_VALUES = { ## source: https://www.vtpi.org/tca/tca0502.pdf
    #     'local': {
    #         'personal': 13.6,
    #         'business': 25.40
    #     },
    #     'intercity': {
    #         'personal': 19.0,
    #         'business': 63.20,
    #         'truck': 27.20,
    #         'bus': 28.30,
    #         'train': 46.10,
    #         'airplane': 86.70
    #     }
    # }

    def personal(
        self,
        med_income: float = 52000.0, ## median annual income of the area
        avg_veh_occupancy: float = 1.67, ## average vehicle occupancy for personal travel
    ) -> float:
        time_value_per_pers = 0.5 * med_income / 2080
        return time_value_per_pers * avg_veh_occupancy

    def buisness(
        self,
        time_value_per_pers: float = 46.84, ## average hourly employment cost from : https://www.bls.gov/charts/employer-costs-for-employee-compensation/costs-per-hour.htm#
        avg_veh_occupancy: float = 1.24
    ) -> float:
        return time_value_per_pers * avg_veh_occupancy

    @classmethod
    def compute_yearly_delay_cost(
        cls,
        delay_time: float, ## MTT(t) - MTT(0), MTT: Mean Travel Time in hours
        sample: bool = True
    ):
        if sample:
            sampled_business_personal_ratio = np.random.normal(
                loc=cls.MEAN_BUSINESS_PERSONAL_RATIO,
                scale=cls.STD_BUSINESS_PERSONAL_RATIO
            )
            sampled_business_personal_ratio = np.clip(sampled_business_personal_ratio, 0, 1)
        else:
            sampled_business_personal_ratio = np.array([0.95, 0.05])

        hourly_time_value_personal = cls.personal(cls)
        hourly_time_value_business = cls.buisness(cls)

        avg_time_value = np.dot(sampled_business_personal_ratio, [hourly_time_value_personal, hourly_time_value_business])

        return 365 * round(avg_time_value * delay_time, 3)




# print(
#     f'Traffic delay cost in $ is: ' + str(TrafficMonetaryValues.compute_yearly_delay_cost(delay_time=30))
# )



class TrafficAccessor:
    def __init__(self, parent_instance):
        self._parent = parent_instance

    def make_traffic_dfs(self):
        net_df, demand_df = import_network(
        self._parent._roads_traffic_net_tntp_f,
        self._parent._roads_traffic_demand_tntp_f
    )
        return net_df, demand_df

    def make_traffic_net(self, net_df, demand_df) -> FlowTransportNetwork:
        traffic_net = load_network(
            in_net_name="Anaheim, USA",
            in_net_df=net_df,
            in_demand_df=demand_df,
            force_net_reprocess=False,
            verbose=self._parent.verbose
        )
        self._log(f'Successfully loaded net df and demand df. Successfully converted them to {traffic_net}')
        return traffic_net

    def user_equilibrium(
        self,
        traffic_net: FlowTransportNetwork,
        return_traffic_links_res: bool = False
    ):
        """
        Computes the user equilibrium for the traffic network.
        Time: Minutes, Distance: Feet, Speed: Feet per minute, Cost: minutes
        """
        results = computeAssignment(
            in_network=traffic_net,
            verbose=self._parent.verbose,
            return_detailed=return_traffic_links_res,
            accuracy=0.01,
            maxTime=3
        )

        if return_traffic_links_res:
            return results

        total_UE_time = results

        self._log(f"UE = {math.ceil(total_UE_time)}")
        return total_UE_time

    def load_network_nodes_gdf(self,
        nodes_gdf: gpd.GeoDataFrame
    ):
        self._parent._roads_traffic_nodes_gdf = nodes_gdf

    def map_traffic_links(self):
        """
        Maps traffic network edges to drivable road network edges by adding a 'traffic_link_index'
        column to the drivable_gdf, indicating which traffic edge each drivable edge is part of.
        """
        traffic_gdf = self._parent._traffic_links_gdf
        drivable_gdf = self._parent._roads_study_gdf

        # Extract all unique nodes from the drivable network
        all_drivable_coords = []
        for geom in drivable_gdf.geometry:
            coords = list(geom.coords)
            all_drivable_coords.extend([coords[0], coords[-1]])

        # Remove duplicate nodes
        unique_drivable_coords = list(set(all_drivable_coords))

        # Prepare coordinates for BallTree
        node_coords_array = np.array([(x, y) for (x, y) in unique_drivable_coords])
        ball_tree = BallTree(node_coords_array, leaf_size=2)

        # Function to find the closest drivable node for a given point
        def get_closest_node(point_coords):
            query_point = np.array([[point_coords[0], point_coords[1]]])
            _, ind = ball_tree.query(query_point, k=1)
            return unique_drivable_coords[ind[0][0]]

        # Build a graph from the drivable network
        drivable_graph = nx.Graph()
        for idx, row in drivable_gdf.iterrows():
            line = row.geometry
            start = line.coords[0]
            end = line.coords[-1]
            length = line.length
            drivable_graph.add_edge(start, end, weight=length, edge_index=idx)

        # Initialize the traffic_link_index column
        drivable_gdf[StudyRoadSchema.TRAFFIC_LINK_INDEX] = None

        # Process each traffic edge
        for traffic_idx, traffic_row in traffic_gdf.iterrows():
            traffic_line = traffic_row.geometry
            traffic_start = traffic_line.coords[0]
            traffic_end = traffic_line.coords[-1]

            try:
                closest_start = get_closest_node(traffic_start)
                closest_end = get_closest_node(traffic_end)
            except:
                continue

            try:
                node_path = nx.shortest_path(drivable_graph, source=closest_start, target=closest_end, weight='weight')
            except nx.NetworkXNoPath:
                continue

            edge_indices = []
            for i in range(len(node_path) - 1):
                u = node_path[i]
                v = node_path[i + 1]
                edge_data = drivable_graph.get_edge_data(u, v)
                if edge_data:
                    edge_indices.append(edge_data['edge_index'])

            drivable_gdf.loc[edge_indices, StudyRoadSchema.TRAFFIC_LINK_INDEX] = traffic_idx

    def step_traffic_calc_net(
        self,
        roads_objs: list[Road]
    ) -> None:
        """
        Calculate and apply capacity reductions from study roads to a traffic network dataframe.
        Each road is mapped to a traffic link using `map_traffic_links()`

        **ATTENTION: roads_study_gdf needs `StudyRoadSchema.TRAFFIC_LINK_INDEX` column name.**
        **ATTENTION: roads_study_gdf needs 'capacity' column name, see https://github.com/bstabler/TransportationNetworks**

        Args:
            roads_study_gdf: GeoDataFrame containing study roads with capacity reduction data
            net_df: DataFrame containing the traffic network

        Returns:
            None
        """
        # Step 1: Group the roads by their traffic_idx and find the maximum capacity reduction for each group
        traffic_idx_capacity_reduction = {}
        # capacities = sum([self._parent.traffic_calc_net.linkSet[l].capacity for l in self._parent.traffic_calc_net.linkSet])
        # print('b ' + str(capacities))
        for road_obj in roads_objs:
            traffic_idx = road_obj.traffic_idx
            delta_percentage = - road_obj.capacity_reduction  # Convert to percentage reduction

            # Store the maximum capacity reduction for each traffic_idx
            if traffic_idx not in traffic_idx_capacity_reduction:
                traffic_idx_capacity_reduction[traffic_idx] = delta_percentage
            else:
                # Keep the minimum (max negative) reduction for roads with the same traffic_idx
                traffic_idx_capacity_reduction[traffic_idx] = min(
                    traffic_idx_capacity_reduction[traffic_idx], delta_percentage
                )

        # Step 2: Apply the maximum capacity reduction to the respective links
        for link in self._parent.traffic_calc_net.linkSet:
            link_idx = self._parent.traffic_calc_net.linkSet[link].idx

            # Check if this link's idx matches any of the traffic_idx
            if link_idx in traffic_idx_capacity_reduction:
                max_delta_percentage = traffic_idx_capacity_reduction[link_idx]
                # Apply the maximum capacity reduction to the link
                ## UNCOMMENT TO DEBUG:
                ## if capacity reduction mathces previous and new capacity then traffic assignment works
                # print('------')
                # print(max_delta_percentage)
                # print(self._parent.traffic_calc_net.linkSet[link].capacity)
                self._parent.traffic_calc_net.linkSet[link].modify_base_capacity(max_delta_percentage)
                # print(self._parent.traffic_calc_net.linkSet[link].capacity)


    def map_rep_road_to_traffic_cap(
            roads_study_gdf: gpd.GeoDataFrame,
            net_df: pd.DataFrame
    ) -> pd.DataFrame:
        net_df_copy = net_df.copy()
        for idx, row in roads_study_gdf.iterrows():
            traffic_idx = row[StudyRoadSchema.TRAFFIC_LINK_INDEX]
            if traffic_idx in net_df_copy.index:
                new_capacity = net_df_copy.loc[traffic_idx, 'capacity'] - math.ceil(
                    net_df_copy.loc[traffic_idx, 'capacity'] * row[StudyRoadSchema.CAPACITY_REDUCTION]
                )
                net_df_copy.loc[traffic_idx, 'capacity'] = new_capacity
        return net_df_copy

    @staticmethod
    def update_capacities(
        roads_study_gdf: gpd.GeoDataFrame,
        buildings_study_gdf: gpd.GeoDataFrame,
        recalculate: bool=True,
        plot: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Updates the capacity reductions for roads due to damage state and debris from connected buildings.

        Args:
            roads_study_gdf (gpd.GeoDataFrame): GeoDataFrame containing study roads.
            buildings_study_gdf (gpd.GeoDataFrame): GeoDataFrame containing study buildings.
            recalculate (bool, optional): Flag to force recalculation of debris impact. Defaults to True.

        Returns:
            Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: Updated GeoDataFrames for roads and buildings.
        """
        def viz_debris_road_interdep(roads_gdf, buildings_gdf):
            roads_gdf = roads_gdf.to_crs(epsg=3857)  # Web Mercator (meters)

            # Offset road centerlines
            def offset_road(road, width):
                """Offset road centerline by half its width on both sides."""
                left_offset = road.parallel_offset(width / 2, side="left")
                right_offset = road.parallel_offset(width / 2, side="right")
                return left_offset, right_offset

            left_roads, right_roads = [], []
            for _, row in roads_gdf.iterrows():
                left, right = offset_road(row.geometry, (row.width/2))
                if left and not left.is_empty:
                    left_roads.append(left)
                if right and not right.is_empty:
                    right_roads.append(right)

            # Convert roads back to original CRS
            roads_gdf = roads_gdf.to_crs(epsg=4326)
            left_roads_gdf = gpd.GeoDataFrame(geometry=left_roads, crs=roads_gdf.crs)
            right_roads_gdf = gpd.GeoDataFrame(geometry=right_roads, crs=roads_gdf.crs)

            # Convert WKT strings to Shapely geometries for debris
            debris_col = StudyBuildingSchema.DEBRIS_GEOM
            if isinstance(buildings_gdf[debris_col][0], str):
                buildings_gdf[debris_col] = buildings_gdf[debris_col].apply(wkt.loads)
            debris_gdf = gpd.GeoDataFrame(geometry=buildings_gdf[debris_col], crs=buildings_gdf.crs)
            debris_gdf = debris_gdf.loc[buildings_gdf[StudyBuildingSchema.DAMAGE_STATE] > 2]
            buildings_and_debris_gdf = buildings_gdf.copy()

            fig, ax = plt.subplots(figsize=(20, 20))

            # Convert to Web Mercator (EPSG:3857) for base map
            roads_gdf = roads_gdf.to_crs(epsg=3857)
            buildings_gdf = buildings_gdf.to_crs(epsg=3857)
            debris_gdf = debris_gdf.to_crs(epsg=3857)

            # Plot elements
            buildings_gdf.plot(ax=ax, color="blue", alpha=0.5, edgecolor="black")
            left_roads_gdf.plot(ax=ax, color="green", linewidth=1)
            right_roads_gdf.plot(ax=ax, color="green", linewidth=1)
            debris_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

            # Add index labels to buildings (centroid of each building)
            for idx, row in buildings_gdf.iterrows():
                centroid = row.geometry.centroid
                ax.text(centroid.x, centroid.y, str(idx), fontsize=10, ha='center', va='center', color='white',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            # Add index labels to roads (midpoint of each road)
            for idx, row in roads_gdf.iterrows():
                midpoint = row.geometry.interpolate(0.5, normalized=True)  # Midpoint of the line
                ax.text(midpoint.x, midpoint.y, str(idx), fontsize=10, ha='center', va='center', color='black',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

            plt.show()

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
                access_road_idx_col = StudyBuildingSchema.ACCESS_ROAD_IDX if StudyBuildingSchema.ACCESS_ROAD_IDX in study_buildings_gdf.columns else StudyBuildingSchema.ACCESS_ROAD_IDX[:10]

                focus_buildings = study_buildings_gdf[study_buildings_gdf[access_road_idx_col] == road_idx]

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
            access_road_idx_col = StudyBuildingSchema.ACCESS_ROAD_IDX if StudyBuildingSchema.ACCESS_ROAD_IDX in study_buildings_gdf.columns else StudyBuildingSchema.ACCESS_ROAD_IDX[:10]
            focus_buildings = study_buildings_gdf[study_buildings_gdf[access_road_idx_col] == road_idx]

            # Create GeoDataFrames
            focus_buildings_gdf = gpd.GeoDataFrame(geometry=focus_buildings.geometry, crs=study_buildings_gdf.crs)
            # focus_buildings.loc[:, StudyBuildingSchema.DEBRIS_GEOM] = focus_buildings[StudyBuildingSchema.DEBRIS_GEOM].apply(wkt.loads)
            # Filter focus_buildings to include only 'Extensive' or 'Complete' damage states
            focus_buildings = focus_buildings[focus_buildings[StudyBuildingSchema.DAMAGE_STATE].isin([DamageStates.to_int('Extensive'), DamageStates.to_int('Complete')])]
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
                recalculate=recalculate,
                verbose=False
            )
            # print(capacity_reduction_ds)
            # print(capacity_reduction_debris)
            roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_RED_DS] = capacity_reduction_ds
            roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_RED_DEBRIS] = max_capacity_reduction_debris
            roads_study_gdf.loc[idx, StudyRoadSchema.CAPACITY_REDUCTION] = max(capacity_reduction_ds, max_capacity_reduction_debris)

            if recalculate:
                # Update buildings with capacity reduction values from dictionary
                for building_id, capacity_reduction in capacity_reduction_debris_dict.items():
                    buildings_study_gdf.loc[building_id, StudyBuildingSchema.CAPACITY_REDUCTION] = capacity_reduction
        if plot:
            viz_debris_road_interdep(buildings_gdf=buildings_study_gdf, roads_gdf=roads_study_gdf)

        return buildings_study_gdf, roads_study_gdf

    def plot_networks(self, figsize: Tuple[float, float] = (10, 10),
        colour_roads: str = '#1f77b4',  # Subtle blue
        colour_traffic: str = '#ff7f0e',  # Subtle orange
        show_road_labels: bool = False
    ):
        traffic_gdf = self._parent._traffic_links_gdf
        drivable_gdf = self._parent._roads_study_gdf

        # Set a minimalist style using seaborn
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Plot drivable roads with subtle blue color
        drivable_gdf.plot(ax=ax, color=colour_roads, linewidth=1.5, label='Drivable Roads')

        # Plot traffic links with subtle orange color and dashed lines
        traffic_gdf.plot(ax=ax, color=colour_traffic, linestyle='dashed', linewidth=2, label='Traffic Links')

        # Filter mapped edges (those with a traffic_link_index)
        mapped_edges = drivable_gdf.dropna(subset=[StudyRoadSchema.TRAFFIC_LINK_INDEX])

        # Plot mapped edges in a muted green color with a thicker line
        mapped_edges.plot(ax=ax, color='#2ca02c', linewidth=4, label='Mapped Edges', alpha=0.5)

        # Add labels for each drivable road with a cleaner font style
        if show_road_labels:
            for idx, row in drivable_gdf.iterrows():
                centroid = row.geometry.centroid
                ax.text(centroid.x, centroid.y, str(row[StudyRoadSchema.LINKNWID]),
                        fontsize=8, color='#1f77b4', ha='center', va='center', weight='light')

        # Set plot bounds around the mapped edges
        ax.set_xlim(mapped_edges.total_bounds[0], mapped_edges.total_bounds[2])
        ax.set_ylim(mapped_edges.total_bounds[1], mapped_edges.total_bounds[3])

        # Add gridlines for better reference
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

        # Display the legend with more minimalist design
        plt.legend(frameon=False, loc='upper right', fontsize=10)

        # Set the plot title with a more professional font style
        plt.title("Drivable Network and Mapped Traffic Links", fontsize=14, weight='bold')

        # Show the plot with a cleaner style
        plt.tight_layout()  # Adjust layout for better spacing
        plt.show()

    def _log(self, message: str) -> None:
        """
        Log messages if verbose mode is enabled.

        Args:
            message (str): Message to log
        """
        if self._parent.verbose:
            print(message)

# if __name__ == '__main__':

#     # This is an example usage for calculating System Optimal and User Equilibrium with Frank-Wolfe

#     net_file = str(PathUtils.anaheim_net_file)


#     for i in range(10):
#         total_system_travel_time_equilibrium = computeAssignment(net_file=net_file,
#                                                             algorithm="MSA",
#                                                             costFunction=BPRcostFunction,
#                                                             systemOptimal=False,
#                                                             verbose=True,
#                                                             accuracy=0.01,
#                                                             maxIter=1000,
#                                                             maxTime=60000)

#         print("UE =", total_system_travel_time_equilibrium)
