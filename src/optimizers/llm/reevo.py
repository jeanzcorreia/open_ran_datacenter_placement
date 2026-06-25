"""
src/optimizers/llm/reevo.py — Otimizador-LLM estilo ReEvo (NeurIPS 2024) para o MODO JUSTO.

Laço: gerar população de HEURÍSTICAS (código `place_odcs`) -> avaliar (HV da fronteira varrida,
no FairODCProblem, penalizada por inviabilidade) -> refletir (curto/longo prazo) -> evoluir
(crossover/mutação guiados) -> selecionar com elitismo -> repetir. Roda em sandbox; loga
chamadas/tokens/custo. Interface Optimizer.solve(instance, budget, seed) -> ParetoSet.
"""

from __future__ import annotations

import math
import re
import time

import numpy as np

from ...problem.odc_problem import FairODCProblem
from ..base import Optimizer, ParetoSet, feasible_nd_front
from . import prompts
from .heuristic_runtime import HeuristicInstance, SandboxError, run_heuristic

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)

# Direções distintas para a geração inicial -> população DIVERSA (e prompts distintos, não
# colapsados pelo cache). Uma por indivíduo (idx % len).
IDEA_HINTS = [
    "greedy incremental que adiciona o site que mais reduz a distância média (k-median)",
    "ordene os sites pela demanda de CPU atribuída e cubra primeiro os clientes mais carregados",
    "farthest-first: espalhe os ODCs para garantir cobertura de todos a <= 11 km",
    "equilibre a carga por ODC (minimize a carga máxima) respeitando a capacidade de 1000 cores",
    "p-median com troca local (swap) para refinar a seleção após uma construção gulosa",
    "cobertura de conjunto: use o menor nº de ODCs que mantém todo cliente a <= 11 km",
    "híbrido: semeie pelos sites de maior demanda e refine por distância, desempatando por carga",
    "agrupe clientes por proximidade e escolha medoides (sites reais) como ODCs",
]


def extract_code(text: str) -> str:
    m = _CODE_RE.search(text or "")
    code = m.group(1) if m else (text or "")
    return code.strip()


def build_sweep(instance, max_capacity, max_points=14):
    n_sites = instance.n_var
    demand = float(np.sum(instance.cpu_cores))
    min_odc = max(1, math.ceil(demand / max_capacity))
    if min_odc >= n_sites:
        return [n_sites]
    step = max(1, (n_sites - min_odc) // (max_points - 1))
    sweep = sorted(set(range(min_odc, n_sites + 1, step)) | {n_sites})
    return sweep


def build_dense_sweep(instance, max_capacity):
    """Sweep DENSO (todo n_active de min_viável..n_sites) — para a fronteira FINAL, comparável
    em cobertura com os baselines. Usado só na avaliação final (não no laço de evolução)."""
    n_sites = instance.n_var
    demand = float(np.sum(instance.cpu_cores))
    min_odc = max(1, math.ceil(demand / max_capacity))
    return list(range(min_odc, n_sites + 1))


def instance_hv(F: np.ndarray, n_sites: int, dist_max: float) -> float:
    """HV normalizado por bounds fixos da instância (ranking interno de heurísticas)."""
    from pymoo.indicators.hv import HV

    if F is None or len(F) == 0:
        return 0.0
    span = np.array([max(n_sites, 1), max(dist_max, 1e-9)])
    Fn = np.atleast_2d(F) / span
    Fn = Fn[(Fn[:, 0] <= 1.0) & (Fn[:, 1] <= 1.0)]
    if len(Fn) == 0:
        return 0.0
    return float(HV(ref_point=np.array([1.05, 1.05]))(Fn))


class Heuristic:
    __slots__ = ("code", "score", "pareto", "origin", "n_valid")

    def __init__(self, code, score, pareto, origin, n_valid):
        self.code = code
        self.score = score
        self.pareto = pareto
        self.origin = origin
        self.n_valid = n_valid


def evaluate_heuristic(code, instance, fair, hinst, sweep, dist_max, per_call_timeout=1.2):
    """Varre n_active, monta a fronteira no problema VERDADEIRO, devolve (score, ParetoSet, n_valid)."""
    Xs = []
    for n in sweep:
        try:
            idx = run_heuristic(code, hinst, n, timeout=per_call_timeout)
        except SandboxError:
            continue
        x = np.zeros(hinst.n_sites)
        x[idx] = 1.0
        Xs.append(x)
    if not Xs:
        return 0.0, None, 0
    X = np.array(Xs)
    F, G = fair.evaluate_population(X)
    Xf, Ff, feas = feasible_nd_front(X, F, G)
    score = instance_hv(Ff[feas] if feas.any() else Ff, hinst.n_sites, dist_max) if feas.any() else 0.0
    return score, (Xf, Ff, feas), len(Xs)


# ---------------------------------------------------------------- adaptador para apply (transfer)
def heuristic_to_pareto(code, instance, max_distance, max_capacity, method, instance_name, meta=None):
    """Aplica um CÓDIGO de heurística fixo a uma instância e devolve um ParetoSet (modo justo).
    Usado na transferência Natal->Manaus (sem re-evoluir)."""
    fair = FairODCProblem(instance, max_distance, max_capacity)
    hinst = HeuristicInstance.from_instance(instance, max_distance, max_capacity)
    sweep = build_dense_sweep(instance, max_capacity)   # transfer: fronteira densa (comparação justa)
    dist_max = float(hinst.distances.max())
    score, front, n_valid = evaluate_heuristic(code, instance, fair, hinst, sweep, dist_max, per_call_timeout=2.0)
    if front is None:
        Xf = np.zeros((1, instance.n_var)); Ff = np.zeros((1, 2)); feas = np.array([False])
    else:
        Xf, Ff, feas = front
    m = dict(internal_hv=score, n_valid_sweep=n_valid, sweep_points=len(sweep))
    m.update(meta or {})
    return ParetoSet(X=Xf, F=Ff, feasible=feas, method=method, instance=instance_name,
                     seed=0, budget=0, meta=m)


class ReEvoOptimizer(Optimizer):
    name = "reevo"

    def __init__(self, llm_client, max_distance=11.0, max_capacity=1000.0,
                 pop_size=8, generations=6, elite=2, verbose=True):
        self.llm = llm_client
        self.max_distance = max_distance
        self.max_capacity = max_capacity
        self.pop_size = pop_size
        self.generations = generations
        self.elite = elite
        self.verbose = verbose

    # ---- operadores (LLM ou offline) ----
    def _gen(self, idx, hint=None):
        if hint is None:
            hint = IDEA_HINTS[idx % len(IDEA_HINTS)]
        r = self.llm.complete("generate", prompts.SYSTEM, prompts.generate_user(hint),
                              context={"index": idx})
        return extract_code(r.text)

    def _reflect_short(self, better, worse):
        r = self.llm.complete(
            "reflect_short", prompts.SYSTEM,
            prompts.reflect_short_user(better.code, better.score, worse.code, worse.score),
            context={"op": "reflect_short"}, model=self.llm.reflection_model)
        return r.text.strip()

    def _reflect_long(self, shorts):
        r = self.llm.complete("reflect_long", prompts.SYSTEM, prompts.reflect_long_user(shorts),
                              context={"op": "reflect_long"}, model=self.llm.reflection_model)
        return r.text.strip()

    def _crossover(self, a, b, long_refl):
        r = self.llm.complete("crossover", prompts.SYSTEM,
                              prompts.crossover_user(a.code, b.code, long_refl),
                              context={"parents": [a.origin, b.origin]})
        return extract_code(r.text)

    def _mutate(self, p, long_refl):
        r = self.llm.complete("mutate", prompts.SYSTEM, prompts.mutate_user(p.code, long_refl),
                              context={"origin": p.origin})
        return extract_code(r.text)

    def solve(self, instance, budget: int = None, seed: int = 1) -> ParetoSet:
        fair = FairODCProblem(instance, self.max_distance, self.max_capacity)
        hinst = HeuristicInstance.from_instance(instance, self.max_distance, self.max_capacity)
        sweep = build_sweep(instance, self.max_capacity)
        dist_max = float(hinst.distances.max())
        t0 = time.time()

        def ev(code, origin):
            score, front, n_valid = evaluate_heuristic(code, instance, fair, hinst, sweep, dist_max)
            return Heuristic(code, score, front, origin, n_valid)

        # população inicial
        pop = []
        for i in range(self.pop_size):
            try:
                pop.append(ev(self._gen(i), "seed"))
            except Exception as e:
                if self.verbose:
                    print(f"  [gen-init {i}] falhou: {e}")
        if not pop:
            raise RuntimeError("Nenhuma heurística inicial válida.")
        pop.sort(key=lambda h: h.score, reverse=True)
        if self.verbose:
            print(f"  init: melhor HV={pop[0].score:.4f} ({len(pop)} heurísticas)")

        shorts, long_refl = [], ""
        hv_curve = [pop[0].score]
        n_cross = max(1, self.pop_size // 2)
        n_mut = max(1, self.pop_size // 2)

        for g in range(self.generations):
            valid = [h for h in pop if h.score > 0]
            if len(valid) >= 2:
                try:
                    shorts.append(self._reflect_short(valid[0], valid[-1]))
                    long_refl = self._reflect_long(shorts)
                except Exception as e:
                    if self.verbose:
                        print(f"  [reflect g{g}] {e}")

            offspring = []
            top = pop[: max(self.elite, 2)]
            for j in range(n_cross):
                a = top[j % len(top)]
                b = top[(j + 1) % len(top)]
                try:
                    offspring.append(ev(self._crossover(a, b, long_refl), "crossover"))
                except Exception as e:
                    if self.verbose:
                        print(f"  [crossover g{g}.{j}] {e}")
            for j in range(n_mut):
                p = pop[j % max(self.elite, 1)]
                try:
                    offspring.append(ev(self._mutate(p, long_refl), "mutate"))
                except Exception as e:
                    if self.verbose:
                        print(f"  [mutate g{g}.{j}] {e}")

            pop = sorted(pop + offspring, key=lambda h: h.score, reverse=True)[: self.pop_size]
            hv_curve.append(pop[0].score)
            if self.verbose:
                print(f"  gen {g+1}/{self.generations}: melhor HV={pop[0].score:.4f} "
                      f"(origem={pop[0].origin}) | chamadas={self.llm.usage.calls} "
                      f"custo=${self.llm.usage.cost_usd:.3f}")

        best = pop[0]
        # Fronteira FINAL com sweep DENSO (cobertura comparável aos baselines; o laço usou sweep coarse p/ velocidade).
        dense = build_dense_sweep(instance, self.max_capacity)
        _ds, dfront, dn = evaluate_heuristic(best.code, instance, fair, hinst, dense, dist_max, per_call_timeout=2.0)
        if dfront is not None:
            Xf, Ff, feas = dfront
        elif best.pareto is not None:
            Xf, Ff, feas = best.pareto
        else:
            Xf, Ff, feas = np.zeros((1, instance.n_var)), np.zeros((1, 2)), np.array([False])
        meta = dict(
            pop_size=self.pop_size, generations=self.generations, elite=self.elite,
            best_origin=best.origin, internal_hv=best.score, hv_curve=hv_curve,
            heuristic_code=best.code, n_valid_sweep=best.n_valid,
            sweep_points=len(sweep), dense_sweep_points=len(dense), final_front_points=dn,
            elapsed_sec=round(time.time() - t0, 2),
            llm_backend=self.llm.name, llm_usage=self.llm.usage.to_dict(),
        )
        return ParetoSet(X=Xf, F=Ff, feasible=feas, method=self.name,
                         instance=instance.name, seed=seed, budget=self.generations, meta=meta)
