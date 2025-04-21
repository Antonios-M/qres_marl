from .building_funcs import Building
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


class Resilience:
    def __init__(
        self,
        sum_initial_income: float,
        sum_current_income: float,
        sum_current_critical_funcs: float,
        sum_initial_critical_funcs: float,
        sum_current_beds: int,
        sum_initial_beds: int,
        sum_current_doctors: int,
        sum_initial_doctors: int,
        costs: np.ndarray,
        w_econ: float = 0.1,
        w_crit: float = 0.45,
        w_health: float = 0.45,
        w_health_bed: float = 0.5,
        w_health_doc: float = 0.5,

    ):
        self._sum_initial_income = sum_initial_income
        self._sum_current_income = sum_current_income
        self._sum_current_critical_funcs = sum_current_critical_funcs
        self._sum_initial_critical_funcs = sum_initial_critical_funcs
        self._sum_current_beds = sum_current_beds
        self._sum_initial_beds = sum_initial_beds
        self._sum_current_doctors = sum_current_doctors
        self._sum_initial_doctors = sum_initial_doctors
        self._costs = costs
        self._w_econ = w_econ
        self._w_crit = w_crit
        self._w_health = w_health
        self._w_bed = w_health_bed
        self._w_doc = w_health_doc

    def step(self,
        sum_income: float,
        sum_critical_funcs: float,
        sum_beds: int,
        sum_doctors: int,
        costs: np.ndarray,

    ) -> None:
        """Update functionality """
        self._sum_current_income = sum_income
        self._sum_current_critical_funcs = sum_critical_funcs
        self._sum_current_beds = sum_beds
        self._sum_current_doctors = sum_doctors
        self._costs = costs

    @property
    def q_community(self) -> float:
        """Calculate the overall community functionality"""
        return self._w_econ * self.q_econ + self._w_crit * self.q_crit + self._w_health * self.q_health

    @property
    def q_community_decomp(self) -> Tuple[float, float, float]:
        return (
            self.q_community,
            self._w_econ * self.q_econ,
            self._w_crit * self.q_crit,
            self._w_health * self.q_health,
        )
    # @property
    # def q_econ(self) -> float:
    #     """
    #     "Calculate the economic functionality
    #     q_econ(t) = BCR(t) = (q_inc(t) - q_inc(t-1)) / Σ(costs(t))""
    #     """
    #     bcr = (self._sum_current_income - np.sum(self._costs)) / self._sum_initial_income
    #     return bcr

    @property
    def q_econ_components(self) -> dict:
        """
        Decomposes the economic functionality (BCR) into:
        - Income contribution
        - Cost contributions (negative)
        Returns a dict where all values sum up to q_econ.
        """
        income_term = self._sum_current_income / self._sum_initial_income
        cost_names = ['building_repair', 'road_repair', 'traffic_delay', 'relocation']
        cost_terms = {
            name: -cost / self._sum_initial_income
            for name, cost in zip(cost_names, self._costs)
        }
        components = {'income': income_term}
        components.update(cost_terms)
        return components

    @property
    def q_econ(self) -> float:
        """
        Total economic functionality (BCR), equal to the sum of components.
        """
        return (sum(self.q_econ_components.values()))

    @property
    def q_crit(self) -> float:
        """Calculate the critical functionality"""
        return self._sum_current_critical_funcs / self._sum_initial_critical_funcs

    @property
    def q_health(
        self,
    ) -> float:
        q_beds = 0.0 if self._sum_initial_beds == 0 else self._sum_current_beds / self._sum_initial_beds
        q_doctors = 0.0 if self._sum_initial_doctors == 0 else self._sum_current_doctors / self._sum_initial_doctors
        q_health = self._w_bed * q_beds + self._w_doc * q_doctors
        return q_health


