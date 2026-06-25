"""Camada de problema: Instance (CSV -> dados) e ODCPlacementProblem (física da avaliação)."""

from .instance import Instance, load_instance, load_instance_sites
from .odc_problem import ODCPlacementProblem, FairODCProblem

__all__ = [
    "Instance",
    "load_instance",
    "load_instance_sites",
    "ODCPlacementProblem",
    "FairODCProblem",
]
