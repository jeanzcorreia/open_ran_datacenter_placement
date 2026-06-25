"""
src/optimizers/nsga2.py — Baseline NSGA-II (pymoo), espelhando a configuração do parser
original (odc_placement_parser.py):

    algorithm  = NSGA2(pop_size=300)                       # operadores DEFAULT do pymoo (reais)
    termination = DefaultSingleObjectiveTermination(
        xtol=1e-8, cvtol=1e-8, ftol=1e-8, period=60, n_max_gen=<budget>)
    res = minimize(problem, algorithm, termination, seed=<seed>, save_history=True)

Diferença DELIBERADA vs. original: PERSISTIMOS a fronteira de Pareto. O original descarta
res.X/res.F; aqui reconstruímos o conjunto não-dominado a partir de `res.opt`
(= algorithm.opt) com fallback para res.X/res.F e para a união do histórico, e o
devolvemos num ParetoSet (HARD RULE Fase 2).
"""

from __future__ import annotations

import time

import numpy as np

from ..problem.instance import Instance
from ..problem.odc_problem import ODCPlacementProblem
from .base import Optimizer, ParetoSet


class NSGA2Optimizer(Optimizer):
    name = "nsga2"

    def __init__(
        self,
        max_distance: float = 11.0,
        max_capacity: float = 1000.0,
        obj_weights: tuple[float, float, float] = (0.0, 0.0, 1.0),
        pop_size: int = 300,
        period: int = 60,
    ):
        self.max_distance = max_distance
        self.max_capacity = max_capacity
        self.obj_weights = tuple(obj_weights)
        self.pop_size = pop_size
        self.period = period

    def solve(self, instance: Instance, budget: int = 60, seed: int = 1) -> ParetoSet:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.termination.default import DefaultSingleObjectiveTermination

        problem_phys = ODCPlacementProblem(
            instance,
            max_distance=self.max_distance,
            max_capacity=self.max_capacity,
            obj_weights=self.obj_weights,
        )
        pymoo_problem = problem_phys.to_pymoo()

        algorithm = NSGA2(pop_size=self.pop_size)
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-8,
            ftol=1e-8,
            period=self.period,
            n_max_gen=budget,
        )

        t0 = time.time()
        res = minimize(
            pymoo_problem,
            algorithm,
            termination=termination,
            seed=seed,
            verbose=False,
            save_history=True,
        )
        elapsed = time.time() - t0

        X, F = self._extract_front(res)
        # Recalcula G/viabilidade sobre o problema (robusto e independente da versão do pymoo).
        _, G = problem_phys.evaluate_population(X)
        feasible = np.all(G <= 0, axis=1)

        n_gen = len(res.history) if res.history is not None else None
        n_eval = int(res.algorithm.evaluator.n_eval) if hasattr(res, "algorithm") else None

        meta = dict(
            pop_size=self.pop_size,
            period=self.period,
            n_max_gen=budget,
            n_gen=n_gen,
            n_eval=n_eval,
            elapsed_sec=round(elapsed, 3),
            obj_weights=self.obj_weights,
            max_distance=self.max_distance,
            max_capacity=self.max_capacity,
            pymoo_version=_pymoo_version(),
            front_source=self._front_source,
        )
        return ParetoSet(
            X=X,
            F=F,
            feasible=feasible,
            method=self.name,
            instance=instance.name,
            seed=seed,
            budget=budget,
            meta=meta,
        )

    # ------------------------------------------------------------------ helpers
    def _extract_front(self, res) -> tuple[np.ndarray, np.ndarray]:
        """Reconstrói o conjunto não-dominado de forma robusta.

        Prioridade: res.opt (algorithm.opt) -> res.X/res.F -> união do histórico.
        Guarda a fonte usada em self._front_source para registro."""
        # 1) res.opt (Population dos ótimos)
        opt = getattr(res, "opt", None)
        if opt is not None and len(opt) > 0:
            X = np.atleast_2d(opt.get("X"))
            F = np.atleast_2d(opt.get("F"))
            if X.size and F.size:
                self._front_source = "res.opt"
                return X, F

        # 2) res.X / res.F
        if res.X is not None and res.F is not None:
            X = np.atleast_2d(res.X)
            F = np.atleast_2d(res.F)
            self._front_source = "res.X/res.F"
            return X, F

        # 3) união não-dominada do histórico
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        Xs, Fs = [], []
        for entry in (res.history or []):
            pop = entry.opt if getattr(entry, "opt", None) is not None else entry.pop
            Xs.append(pop.get("X"))
            Fs.append(pop.get("F"))
        if not Xs:
            raise RuntimeError("Nenhuma solução recuperável de res (opt/X/history vazios).")
        Xall = np.vstack(Xs)
        Fall = np.vstack(Fs)
        nd = NonDominatedSorting().do(Fall, only_non_dominated_front=True)
        self._front_source = "history-union"
        return Xall[nd], Fall[nd]

    _front_source: str = "unknown"


def _pymoo_version() -> str:
    try:
        import pymoo

        return pymoo.__version__
    except Exception:  # pragma: no cover
        return "unknown"
