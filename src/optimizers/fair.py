"""
src/optimizers/fair.py — Otimizadores no MODO JUSTO (Fase 3), todos sob a interface
`Optimizer.solve(instance, budget, seed) -> ParetoSet`.

Convenção comum: cada método busca, e a fronteira reportada é SEMPRE reavaliada no problema
VERDADEIRO (`FairODCProblem`) e reduzida ao conjunto VIÁVEL não-dominado — garantindo que
todos os métodos sejam comparados no mesmo espaço de objetivos/viabilidade.

Métodos:
  NSGA2Fair, NSGA3Fair  — pymoo, restrições nativas (G contínuo).
  MOEADFair             — pymoo MOEA/D NÃO suporta restrições; busca com penalidade estática,
                          fronteira reavaliada no problema verdadeiro.
  RandomSearchFair      — piso: amostra binária (Bernoulli 0.5, = init dos EAs), sem seleção.
  GreedyFair            — construção gulosa determinística sobre os sites (uma solução por k).
"""

from __future__ import annotations

import time

import numpy as np

from ..problem.instance import Instance
from ..problem.odc_problem import FairODCProblem
from .base import Optimizer, ParetoSet, extract_front_X, feasible_nd_front


def _pymoo_version() -> str:
    try:
        import pymoo

        return pymoo.__version__
    except Exception:  # pragma: no cover
        return "unknown"


def _make_pareto(name, instance, seed, budget, X, F, feas, extra_meta) -> ParetoSet:
    meta = dict(pymoo_version=_pymoo_version())
    meta.update(extra_meta)
    return ParetoSet(
        X=X, F=F, feasible=feas, method=name, instance=instance.name, seed=seed,
        budget=budget, meta=meta,
    )


# ------------------------------------------------------------------ EAs (pymoo)
class _FairEA(Optimizer):
    """Base dos baselines evolutivos no modo justo."""

    name = "ea"

    def __init__(self, max_distance: float = 11.0, max_capacity: float = 1000.0, pop_size: int = 300):
        self.max_distance = max_distance
        self.max_capacity = max_capacity
        self.pop_size = pop_size

    def _make_algorithm(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _search_problem(self, fair: FairODCProblem):
        """Problema que o algoritmo VÊ (com restrições nativas, por padrão)."""
        return fair.to_pymoo()

    def solve(self, instance: Instance, budget: int = 60, seed: int = 1) -> ParetoSet:
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination

        fair = FairODCProblem(instance, self.max_distance, self.max_capacity)
        search = self._search_problem(fair)
        algo = self._make_algorithm()

        t0 = time.time()
        res = minimize(search, algo, get_termination("n_gen", budget), seed=seed, verbose=False)
        elapsed = time.time() - t0

        X, src = extract_front_X(res)
        F, G = fair.evaluate_population(X)            # objetivos/viabilidade VERDADEIROS
        Xf, Ff, feas = feasible_nd_front(X, F, G)
        evaluator = getattr(getattr(res, "algorithm", None), "evaluator", None)
        n_eval = int(evaluator.n_eval) if evaluator is not None else None
        return _make_pareto(
            self.name, instance, seed, budget, Xf, Ff, feas,
            dict(pop_size=self.pop_size, n_gen=budget, n_eval=n_eval,
                 elapsed_sec=round(elapsed, 3), front_source=src,
                 max_distance=self.max_distance, max_capacity=self.max_capacity),
        )


class NSGA2Fair(_FairEA):
    name = "nsga2"

    def _make_algorithm(self):
        from pymoo.algorithms.moo.nsga2 import NSGA2

        return NSGA2(pop_size=self.pop_size)


class NSGA3Fair(_FairEA):
    name = "nsga3"

    def _make_algorithm(self):
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.util.ref_dirs import get_reference_directions

        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=self.pop_size - 1)
        return NSGA3(ref_dirs=ref_dirs)


class MOEADFair(_FairEA):
    name = "moead"

    def __init__(self, max_distance=11.0, max_capacity=1000.0, pop_size=300, penalty=1000.0):
        super().__init__(max_distance, max_capacity, pop_size)
        self.penalty = penalty

    def _search_problem(self, fair: FairODCProblem):
        return fair.to_pymoo_penalized(penalty=self.penalty)   # MOEA/D não suporta G nativo

    def _make_algorithm(self):
        from pymoo.algorithms.moo.moead import MOEAD
        from pymoo.util.ref_dirs import get_reference_directions

        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=self.pop_size - 1)
        return MOEAD(ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)


# ------------------------------------------------------------------ Random (piso)
class RandomSearchFair(Optimizer):
    name = "random"

    def __init__(self, max_distance=11.0, max_capacity=1000.0, pop_size=300):
        self.max_distance = max_distance
        self.max_capacity = max_capacity
        self.pop_size = pop_size

    def solve(self, instance: Instance, budget: int = 60, seed: int = 1) -> ParetoSet:
        fair = FairODCProblem(instance, self.max_distance, self.max_capacity)
        rng = np.random.default_rng(seed)
        n_eval = self.pop_size * budget          # mesmo orçamento de avaliações dos EAs
        t0 = time.time()
        X = (rng.random((n_eval, fair.n_var)) > 0.5).astype(float)   # Bernoulli 0.5 (= init dos EAs)
        F, G = fair.evaluate_population(X)
        elapsed = time.time() - t0
        Xf, Ff, feas = feasible_nd_front(X, F, G)
        return _make_pareto(
            self.name, instance, seed, budget, Xf, Ff, feas,
            dict(pop_size=self.pop_size, n_gen=budget, n_eval=n_eval,
                 elapsed_sec=round(elapsed, 3), front_source="random-sample",
                 max_distance=self.max_distance, max_capacity=self.max_capacity),
        )


# ------------------------------------------------------------------ Greedy (opcional)
class GreedyFair(Optimizer):
    """Construção gulosa determinística sobre os SITES (não-contaminada): a cada passo
    adiciona o site que mais reduz a distância média de fronthaul, gerando uma solução por
    k=1..n_sites. `seed`/`budget` ignorados (determinístico)."""

    name = "greedy"

    def __init__(self, max_distance=11.0, max_capacity=1000.0):
        self.max_distance = max_distance
        self.max_capacity = max_capacity

    def solve(self, instance: Instance, budget: int = 0, seed: int = 0) -> ParetoSet:
        fair = FairODCProblem(instance, self.max_distance, self.max_capacity)
        D = instance.distances                 # (n_clients, n_sites)
        n_clients, n_sites = D.shape
        t0 = time.time()
        selected: list[int] = []
        remaining = list(range(n_sites))
        cur_min = np.full(n_clients, np.inf)
        rows = []
        for _ in range(n_sites):
            best_s, best_mean, best_newmin = None, None, None
            for s in remaining:
                newmin = np.minimum(cur_min, D[:, s])
                m = float(newmin.mean())
                if best_mean is None or m < best_mean:
                    best_mean, best_s, best_newmin = m, s, newmin
            selected.append(best_s)
            remaining.remove(best_s)
            cur_min = best_newmin
            x = np.zeros(n_sites)
            x[selected] = 1.0
            rows.append(x.copy())
        X = np.array(rows)
        F, G = fair.evaluate_population(X)
        elapsed = time.time() - t0
        Xf, Ff, feas = feasible_nd_front(X, F, G)
        return _make_pareto(
            self.name, instance, seed, budget, Xf, Ff, feas,
            dict(deterministic=True, n_eval=n_sites, elapsed_sec=round(elapsed, 3),
                 front_source="greedy-construction",
                 max_distance=self.max_distance, max_capacity=self.max_capacity),
        )


FAIR_OPTIMIZERS = {
    "nsga2": NSGA2Fair,
    "nsga3": NSGA3Fair,
    "moead": MOEADFair,
    "random": RandomSearchFair,
    "greedy": GreedyFair,
}
