"""
src/eval/reference_front.py — Fronteira de referência = união dos não-dominados VIÁVEIS de
TODOS os métodos × seeds (CLAUDE.md §8). Fornece (ideal, nadir) para normalização e HV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


@dataclass
class ReferenceFront:
    F: np.ndarray        # (n_ref, n_obj) pontos não-dominados (espaço bruto de objetivos)
    ideal: np.ndarray    # mínimo por objetivo
    nadir: np.ndarray    # máximo por objetivo (sobre a fronteira de referência)
    n_sources: int       # nº de fronteiras viáveis que contribuíram


def build_reference_front(paretosets) -> ReferenceFront:
    """Constrói a fronteira de referência a partir de uma lista de ParetoSet.

    Usa apenas pontos VIÁVEIS; faz a união e extrai o conjunto não-dominado global."""
    feasibleF = []
    for ps in paretosets:
        feas = np.asarray(ps.feasible, dtype=bool)
        if feas.any():
            feasibleF.append(np.atleast_2d(ps.F)[feas])
    if not feasibleF:
        raise RuntimeError("Nenhuma solução viável em nenhum método — fronteira de referência vazia.")
    allF = np.vstack(feasibleF)
    nd = NonDominatedSorting().do(allF, only_non_dominated_front=True)
    ref = np.unique(allF[nd].round(9), axis=0)
    return ReferenceFront(
        F=ref,
        ideal=ref.min(axis=0),
        nadir=ref.max(axis=0),
        n_sources=len(feasibleF),
    )
