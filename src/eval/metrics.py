"""
src/eval/metrics.py — Métricas de Pareto (modo justo).

Todas operam no espaço de objetivos NORMALIZADO por (ideal, nadir) da fronteira de
referência, para comparabilidade entre objetivos de escalas diferentes (f1 = nº de ODCs,
inteiro ~3..55; f2 = distância média em km ~0..3). Convenção de MINIMIZAÇÃO.

- HV   (Hypervolume, pymoo.indicators.hv.HV): MAIOR é melhor. Ponto de referência fixo
       `ref_point` em espaço normalizado (default (1.1, 1.1) — nadir + 10% de margem).
- IGD+ (pymoo.indicators.igd_plus.IGDPlus contra a fronteira de referência): MENOR é melhor.
- Spacing (Schott): uniformidade do espaçamento; MENOR é melhor.
- Spread (Δ de Deb, 2 objetivos): MENOR é melhor (usa os extremos da fronteira de referência).

Pontos inviáveis devem ser excluídos ANTES (o runner só passa fronteiras viáveis).
"""

from __future__ import annotations

import numpy as np

METRIC_NAMES = ["HV", "IGD+", "spacing", "spread"]
HV_REF_POINT = np.array([1.1, 1.1])


def normalize(F: np.ndarray, ideal: np.ndarray, nadir: np.ndarray) -> np.ndarray:
    """(F - ideal) / (nadir - ideal), com proteção a span zero."""
    F = np.atleast_2d(np.asarray(F, dtype=float))
    span = np.asarray(nadir, dtype=float) - np.asarray(ideal, dtype=float)
    span = np.where(span == 0, 1.0, span)
    return (F - np.asarray(ideal, dtype=float)) / span


def hypervolume(F_norm: np.ndarray, ref_point: np.ndarray = HV_REF_POINT) -> float:
    """HV em espaço normalizado (maior é melhor). Retorna 0.0 se vazio."""
    from pymoo.indicators.hv import HV

    F_norm = np.atleast_2d(F_norm)
    if F_norm.size == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref_point, dtype=float))(F_norm))


def igd_plus(F_norm: np.ndarray, ref_front_norm: np.ndarray) -> float:
    """IGD+ contra a fronteira de referência normalizada (menor é melhor)."""
    from pymoo.indicators.igd_plus import IGDPlus

    F_norm = np.atleast_2d(F_norm)
    if F_norm.size == 0:
        return float("nan")
    return float(IGDPlus(np.atleast_2d(ref_front_norm))(F_norm))


def spacing(F_norm: np.ndarray) -> float:
    """Métrica de Spacing de Schott (menor = mais uniforme). Distância L1 ao vizinho mais
    próximo dentro da própria fronteira. Indefinida (<2 pts) -> nan."""
    F = np.atleast_2d(F_norm)
    n = F.shape[0]
    if n < 2:
        return float("nan")
    # distância L1 (Manhattan) par-a-par
    diff = np.abs(F[:, None, :] - F[None, :, :]).sum(axis=2)
    np.fill_diagonal(diff, np.inf)
    d = diff.min(axis=1)
    d_bar = d.mean()
    return float(np.sqrt(((d - d_bar) ** 2).sum() / (n - 1)))


def spread(F_norm: np.ndarray, ref_front_norm: np.ndarray) -> float:
    """Spread Δ de Deb (2 objetivos; menor = melhor distribuição).

    Δ = (d_f + d_l + Σ|d_i − d̄|) / (d_f + d_l + (N−1)·d̄),
    onde d_i são distâncias euclidianas entre soluções consecutivas (ordenadas por f1) e
    d_f, d_l são as distâncias das soluções extremas obtidas aos extremos da fronteira de
    referência. <2 pts -> nan."""
    F = np.atleast_2d(F_norm)
    n = F.shape[0]
    if n < 2:
        return float("nan")
    order = np.argsort(F[:, 0])
    Fs = F[order]
    d = np.sqrt(((Fs[1:] - Fs[:-1]) ** 2).sum(axis=1))
    d_bar = d.mean()
    ref = np.atleast_2d(ref_front_norm)
    # extremos da fronteira de referência (mínimo de cada objetivo)
    ext0 = ref[np.argmin(ref[:, 0])]
    ext1 = ref[np.argmin(ref[:, 1])]
    d_f = float(np.sqrt(((Fs[0] - ext0) ** 2).sum()))
    d_l = float(np.sqrt(((Fs[-1] - ext1) ** 2).sum()))
    denom = d_f + d_l + (n - 1) * d_bar
    if denom == 0:
        return 0.0
    return float((d_f + d_l + np.abs(d - d_bar).sum()) / denom)
