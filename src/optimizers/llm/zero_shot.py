"""
src/optimizers/llm/zero_shot.py — Ablação ZERO-SHOT: o LLM propõe UMA heurística `place_odcs`
(sem laço evolutivo, sem reflexão). Isola o ganho da evolução reflexiva do ReEvo.

Para reduzir variância, faz `n_samples` propostas independentes e fica com a de maior HV
interno (continua sendo "sem evolução": nenhuma reflexão/crossover/mutação).
"""

from __future__ import annotations

import time

import numpy as np

from ...problem.odc_problem import FairODCProblem
from ..base import Optimizer, ParetoSet
from . import prompts
from .heuristic_runtime import HeuristicInstance
from .reevo import build_dense_sweep, build_sweep, evaluate_heuristic, extract_code


class ZeroShotOptimizer(Optimizer):
    name = "zero_shot"

    def __init__(self, llm_client, max_distance=11.0, max_capacity=1000.0, n_samples=4, verbose=True):
        self.llm = llm_client
        self.max_distance = max_distance
        self.max_capacity = max_capacity
        self.n_samples = n_samples
        self.verbose = verbose

    def solve(self, instance, budget: int = None, seed: int = 1) -> ParetoSet:
        fair = FairODCProblem(instance, self.max_distance, self.max_capacity)
        hinst = HeuristicInstance.from_instance(instance, self.max_distance, self.max_capacity)
        sweep = build_sweep(instance, self.max_capacity)
        dist_max = float(hinst.distances.max())
        t0 = time.time()

        best = None
        for i in range(self.n_samples):
            r = self.llm.complete("generate", prompts.SYSTEM, prompts.generate_user(),
                                  context={"index": i})
            code = extract_code(r.text)
            score, front, n_valid = evaluate_heuristic(code, instance, fair, hinst, sweep, dist_max)
            if best is None or score > best[0]:
                best = (score, front, code, n_valid)
            if self.verbose:
                print(f"  zero_shot sample {i+1}/{self.n_samples}: HV={score:.4f}")

        score, front, code, n_valid = best
        # fronteira FINAL com sweep DENSO (cobertura comparável aos baselines)
        dense = build_dense_sweep(instance, self.max_capacity)
        _ds, dfront, dn = evaluate_heuristic(code, instance, fair, hinst, dense, dist_max, per_call_timeout=2.0)
        if dfront is not None:
            Xf, Ff, feas = dfront
            n_valid = dn
        elif front is not None:
            Xf, Ff, feas = front
        else:
            Xf = np.zeros((1, instance.n_var)); Ff = np.zeros((1, 2)); feas = np.array([False])
        meta = dict(internal_hv=score, heuristic_code=code, n_samples=self.n_samples,
                    n_valid_sweep=n_valid, sweep_points=len(sweep),
                    elapsed_sec=round(time.time() - t0, 2),
                    llm_backend=self.llm.name, llm_usage=self.llm.usage.to_dict())
        return ParetoSet(X=Xf, F=Ff, feasible=feas, method=self.name,
                         instance=instance.name, seed=seed, budget=0, meta=meta)
