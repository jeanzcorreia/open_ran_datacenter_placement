"""Otimizadores.
Fase 2: base (interface) + nsga2 (baseline reprodução).
Fase 3: fair (NSGA-II/III, MOEA/D, random, greedy no modo justo)."""

from .base import Optimizer, ParetoSet, extract_front_X, feasible_nd_front
from .nsga2 import NSGA2Optimizer
from .fair import (
    FAIR_OPTIMIZERS,
    GreedyFair,
    MOEADFair,
    NSGA2Fair,
    NSGA3Fair,
    RandomSearchFair,
)

__all__ = [
    "Optimizer",
    "ParetoSet",
    "extract_front_X",
    "feasible_nd_front",
    "NSGA2Optimizer",
    "NSGA2Fair",
    "NSGA3Fair",
    "MOEADFair",
    "RandomSearchFair",
    "GreedyFair",
    "FAIR_OPTIMIZERS",
]
