"""Avaliação multiobjetivo (Fase 3): métricas de Pareto, fronteira de referência, runner."""

from .metrics import hypervolume, igd_plus, spacing, spread, normalize, METRIC_NAMES
from .reference_front import build_reference_front, ReferenceFront

__all__ = [
    "hypervolume",
    "igd_plus",
    "spacing",
    "spread",
    "normalize",
    "METRIC_NAMES",
    "build_reference_front",
    "ReferenceFront",
]
