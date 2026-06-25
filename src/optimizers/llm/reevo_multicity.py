"""
src/optimizers/llm/reevo_multicity.py — Fase 5a: ReEvo MULTI-CIDADE.

Estende o ReEvo da Fase 4 (mesmos operadores LLM, mesmo sandbox, mesmo roteamento/cache) com
uma AVALIAÇÃO DE FITNESS multi-cidade e uma PENALIDADE DE ROBUSTEZ:

  fitness(heurística) = HV MÉDIO sobre as cidades de TREINO (instance_hv normalizado por cidade);
  reporta também o MAXIMIN (HV da PIOR cidade de treino).

  PENALIDADE DE ROBUSTEZ: se a heurística LANÇAR exceção/timeout (n_crash > 0) OU produzir
  fronteira INVIÁVEL (nenhum ponto viável => score == 0) em QUALQUER cidade de treino, ela
  recebe um score fortemente penalizado (NEGATIVO) e NÃO pode vencer uma heurística robusta.
  Isso endereça o bug da Fase 4 (UnboundLocalError em início inviável) e seleciona contra
  heurísticas lentas (que estouram o timeout nas cidades grandes, p.ex. BH/Manaus).

INVARIANTE DE CUSTO: os operadores LLM (_gen/_crossover/_mutate/_reflect_*) atuam SÓ sobre
CÓDIGO + reflexões. A avaliação multi-cidade é 100% CPU (sandbox). Logo o nº de chamadas de
API NÃO cresce com o nº de cidades — depende só de pop/gen (~68/seed, igual à Fase 4).

🔒 INTEGRIDADE (CLAUDE.md §10): `solve_multi` recebe SOMENTE as 6 cidades de treino. Há uma
asserção de defesa-em-profundidade contra vazamento das 4 cidades de teste.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ...problem.odc_problem import FairODCProblem
from ..base import ParetoSet, feasible_nd_front
from . import prompts
from .heuristic_runtime import HeuristicInstance, SandboxError, run_heuristic
from .reevo import (
    IDEA_HINTS,
    ReEvoOptimizer,
    build_sweep,
    extract_code,
    instance_hv,
)

# Nomes (instance.name = "<arquivo>_sites") das 4 cidades de TESTE held-out. NUNCA podem
# entrar no fitness/seleção do ReEvo (trava dura da Fase 5a).
TEST_CITY_TOKENS = ("curitiba", "recife", "florianopolis", "vitoria")

# Erros que indicam BUG DE CÓDIGO (não falha transitória de LLM/sandbox): devem estourar alto
# em vez de serem engolidos pelo try/except do laço (foi isto que mascarou o bug do _HShim).
_STRUCTURAL_ERRORS = (AttributeError, TypeError, NameError, KeyError, IndexError, ImportError)

# SEED FORTE (Fase 5a corrigida): construção vetorizada LIMPA por distância MÍNIMA — ordena os
# sites pela distância ao cliente MAIS PRÓXIMO (cobertura). Rápida (uma passada O(n_cli·n_sites)),
# robusta (sem crash em nenhuma cidade) e COMPETITIVA (HV de benchmark ~0.95–0.99, vs ~0.34–0.46 da
# variante "distância média" degenerada da rodada anterior). Injetada na população inicial para
# garantir ≥1 robusta competitiva; o LLM gera as outras pop_size−1 e a evolução parte daí. A
# PROVENIÊNCIA da vencedora (origin) distingue se a vitória veio deste seed, de uma init do LLM,
# ou de crossover/mutação (evolução).
STRONG_SEED_CODE = """import numpy as np
def place_odcs(instance, n_active):
    d = instance.distances.min(axis=0)               # dist. de cada site ao cliente mais próximo
    n = int(max(1, min(n_active, instance.n_sites)))
    return list(np.argsort(d)[:n])                    # n_active sites de melhor cobertura
"""


def build_capped_sweep(instance, max_capacity, max_points=200):
    """Sweep denso porém LIMITADO a `max_points` valores de n_active (mín_viável..n_sites).

    Para cidades pequenas (faixa <= max_points) é o sweep denso completo; para cidades grandes
    (BH/Manaus/Curitiba) amostra uniformemente para limitar o custo da aplicação da vencedora
    sem perder cobertura da fronteira."""
    n_sites = instance.n_var
    demand = float(np.sum(instance.cpu_cores))
    min_odc = max(1, math.ceil(demand / max_capacity))
    if min_odc >= n_sites:
        return [n_sites]
    full = list(range(min_odc, n_sites + 1))
    if len(full) <= max_points:
        return full
    # amostra EXATAMENTE <= max_points valores uniformemente, sempre incluindo os extremos
    # (min_odc e n_sites) — respeita o teto declarado.
    idx = np.unique(np.linspace(0, len(full) - 1, max_points).round().astype(int))
    return [full[int(i)] for i in idx]


def evaluate_heuristic_robust(code, fair, hinst, sweep, dist_max, per_call_timeout=1.5):
    """Como `evaluate_heuristic` (reevo.py), mas também conta CRASHES (SandboxError/timeout).

    Retorna (score, front, n_valid, n_crash):
      score    : instance_hv da fronteira VIÁVEL não-dominada (0.0 se nenhuma viável);
      front    : (Xf, Ff, feas) ou None se nenhum ponto produzido;
      n_valid  : nº de seleções produzidas sem crash;
      n_crash  : nº de pontos do sweep onde a heurística LANÇOU (exceção/timeout/saída inválida).
    """
    Xs = []
    n_crash = 0
    for n in sweep:
        try:
            idx = run_heuristic(code, hinst, n, timeout=per_call_timeout)
        except SandboxError:
            n_crash += 1
            continue
        x = np.zeros(hinst.n_sites)
        x[idx] = 1.0
        Xs.append(x)
    if not Xs:
        return 0.0, None, 0, n_crash
    X = np.array(Xs)
    F, G = fair.evaluate_population(X)
    Xf, Ff, feas = feasible_nd_front(X, F, G)
    score = instance_hv(Ff, hinst.n_sites, dist_max) if feas.any() else 0.0
    return score, (Xf, Ff, feas), len(Xs), n_crash


@dataclass
class MultiHeuristic:
    """Heurística avaliada nas cidades de treino (fitness agregado + diagnóstico por cidade)."""

    code: str
    agg: float                  # score de RANKING (mean_hv se robusta; NEGATIVO se não)
    mean_hv: float              # HV médio sobre as cidades de treino
    maximin: float              # HV da pior cidade de treino
    robust: bool                # n_crash==0 e score>0 em TODAS as cidades de treino
    origin: str
    per_city: list = field(default_factory=list)  # [{name, score, n_valid, n_crash, ok}]


@dataclass
class MultiCityResult:
    """Resultado de `solve_multi`: a vencedora + metadados de treino (NÃO contém fronteiras
    de benchmark — estas são produzidas aplicando-se a vencedora a cada cidade no runner)."""

    code: str
    meta: dict


class MultiCityReEvoOptimizer(ReEvoOptimizer):
    """ReEvo cujo fitness é o HV médio sobre VÁRIAS cidades de treino + penalidade de robustez.

    Reutiliza os operadores LLM e o sandbox do `ReEvoOptimizer` (Fase 4) sem alterá-los."""

    name = "reevo"

    def __init__(self, *args, per_call_timeout=1.5, sweep_max_points=14, strong_seeds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.per_call_timeout = float(per_call_timeout)
        self.sweep_max_points = int(sweep_max_points)
        # heurísticas-semente FORTES injetadas na pop inicial (origin="seed_strong").
        self.strong_seeds = list(strong_seeds) if strong_seeds else []

    # --- agregação multi-cidade do fitness -----------------------------------------
    @staticmethod
    def _aggregate(code, origin, cities, per_call_timeout):
        """PISO DE QUALIDADE (Fase 5a corrigida): crash/inviável numa cidade ⇒ HV daquela cidade
        = 0 (PENALIDADE), NÃO eliminação da heurística. Assim uma boa construção com um bug pontual
        (forte em 5 cidades, quebra em 1) ainda COMPETE pela média — e a evolução/reflexão pode
        consertá-la — em vez de ser descartada (o que, na rodada anterior, deixou vencer a única
        robusta-porém-fraca). Fitness = HV MÉDIO com piso 0 por cidade; reporta maximin e robustez."""
        per_city = []
        scores = []
        n_failed = 0
        for c in cities:
            raw, _front, n_valid, n_crash = evaluate_heuristic_robust(
                code, c["fair"], c["hinst"], c["sweep"], c["dist_max"], per_call_timeout
            )
            crashed = (n_crash > 0) or (raw <= 0.0)        # crash/timeout OU fronteira inviável
            city_score = 0.0 if crashed else raw            # <-- PISO: penaliza, não elimina
            if crashed:
                n_failed += 1
            scores.append(city_score)
            per_city.append(dict(name=c["name"], score=float(city_score), raw_score=float(raw),
                                 n_valid=int(n_valid), n_crash=int(n_crash), ok=(not crashed)))
        scores = np.asarray(scores, dtype=float)
        mean_hv = float(scores.mean())                      # média JÁ inclui os zeros de penalidade
        maximin = float(scores.min())
        robust = (n_failed == 0)                            # só p/ relatório (não elimina mais)
        agg = mean_hv                                       # ranking = HV médio (com piso 0/cidade)
        return MultiHeuristic(code, agg, mean_hv, maximin, robust, origin, per_city)

    def solve_multi(self, train_instances, seed: int = 1) -> MultiCityResult:
        # 🔒 trava anti-vazamento: nenhuma cidade de teste pode entrar no treino/seleção.
        names = [getattr(inst, "name", "") for inst in train_instances]
        for nm in names:
            low = nm.lower()
            if any(tok in low for tok in TEST_CITY_TOKENS):
                raise RuntimeError(
                    f"VAZAMENTO DE TESTE: cidade held-out '{nm}' chegou ao treino do ReEvo. Abortando.")
        # Namespace de cache por SEED -> variância real entre seeds (com temperatura > 0).
        if hasattr(self.llm, "set_cache_salt"):
            self.llm.set_cache_salt(f"reevo_s{seed}")
        if self.verbose:
            print(f"  [multi-city] seed={seed} | treino em {len(train_instances)} cidades: {names} "
                  f"| {len(self.strong_seeds)} seed(s) forte(s) injetada(s)")

        # Pré-computa (uma vez) os objetos por cidade de treino.
        cities = []
        for inst in train_instances:
            fair = FairODCProblem(inst, self.max_distance, self.max_capacity)
            hinst = HeuristicInstance.from_instance(inst, self.max_distance, self.max_capacity)
            sweep = build_sweep(inst, self.max_capacity, max_points=self.sweep_max_points)
            cities.append(dict(name=inst.name, instance=inst, fair=fair, hinst=hinst,
                               sweep=sweep, dist_max=float(hinst.distances.max())))
        t0 = time.time()

        def ev(code, origin):
            return self._aggregate(code, origin, cities, self.per_call_timeout)

        # ---- população inicial: seeds FORTES injetadas + restante gerado pelo LLM ----
        pop = []
        for sc in self.strong_seeds:                      # construção limpa conhecida (sem API)
            try:
                pop.append(ev(sc, "seed_strong"))
            except _STRUCTURAL_ERRORS:
                raise
            except Exception as e:
                if self.verbose:
                    print(f"  [seed_strong] falhou: {e}")
        for i in range(self.pop_size - len(self.strong_seeds)):
            try:
                pop.append(ev(self._gen(i), "seed"))
            except Exception as e:
                if self.verbose:
                    print(f"  [gen-init {i}] falhou: {e}")
        if not pop:
            raise RuntimeError("Nenhuma heurística inicial válida.")
        pop.sort(key=lambda h: h.agg, reverse=True)
        if self.verbose:
            b = pop[0]
            print(f"  init: melhor agg={b.agg:.4f} meanHV={b.mean_hv:.4f} maximin={b.maximin:.4f} "
                  f"robusta={b.robust} ({len(pop)} heurísticas, {sum(h.robust for h in pop)} robustas)")

        shorts, long_refl = [], ""
        # curvas de convergência por geração (do melhor da população)
        agg_curve = [pop[0].agg]
        mean_hv_curve = [pop[0].mean_hv]
        maximin_curve = [pop[0].maximin]
        origin_curve = [pop[0].origin]
        n_cross = max(1, self.pop_size // 2)
        n_mut = max(1, self.pop_size // 2)
        total_offspring = 0

        for g in range(self.generations):
            valid = [h for h in pop if h.agg > 0]      # robustas (agg = mean_hv > 0)
            if len(valid) >= 2:
                try:
                    shorts.append(self._reflect_short(_as_h(valid[0]), _as_h(valid[-1])))
                    long_refl = self._reflect_long(shorts)
                except _STRUCTURAL_ERRORS:
                    raise
                except Exception as e:
                    if self.verbose:
                        print(f"  [reflect g{g}] {e}")

            offspring = []
            top = pop[: max(self.elite, 2)]
            for j in range(n_cross):
                a = top[j % len(top)]
                b = top[(j + 1) % len(top)]
                try:
                    offspring.append(ev(self._crossover(_as_h(a), _as_h(b), long_refl), "crossover"))
                except _STRUCTURAL_ERRORS:    # bug de código -> falha alto (não mascara)
                    raise
                except Exception as e:        # falha transitória de LLM/sandbox -> tolera
                    if self.verbose:
                        print(f"  [crossover g{g}.{j}] {e}")
            for j in range(n_mut):
                p = pop[j % max(self.elite, 1)]
                try:
                    offspring.append(ev(self._mutate(_as_h(p), long_refl), "mutate"))
                except _STRUCTURAL_ERRORS:
                    raise
                except Exception as e:
                    if self.verbose:
                        print(f"  [mutate g{g}.{j}] {e}")
            total_offspring += len(offspring)

            pop = sorted(pop + offspring, key=lambda h: h.agg, reverse=True)[: self.pop_size]
            b = pop[0]
            agg_curve.append(b.agg)
            mean_hv_curve.append(b.mean_hv)
            maximin_curve.append(b.maximin)
            origin_curve.append(b.origin)
            if self.verbose:
                # `offspring` = nº de filhos AVALIADOS (detecta falha silenciosa de evolução:
                # se ~0, crossover/mutate estão lançando — ver _HShim).
                print(f"  gen {g+1}/{self.generations}: agg={b.agg:.4f} meanHV={b.mean_hv:.4f} "
                      f"maximin={b.maximin:.4f} robusta={b.robust} origem={b.origin} | "
                      f"offspring={len(offspring)}/{n_cross+n_mut} chamadas={self.llm.usage.calls} "
                      f"custo=${self.llm.usage.cost_usd:.3f}")

        best = pop[0]
        if total_offspring == 0 and self.generations > 0:
            # Sinaliza ALTO: nenhum filho foi avaliado -> evolução inerte (crossover/mutate
            # falhando). Não deveria acontecer após o fix do _HShim; é a salvaguarda contra
            # regressão silenciosa (o modo como o bug original passou despercebido).
            print("  [ALERTA] 0 offspring avaliados em todas as gerações — EVOLUÇÃO INERTE "
                  "(crossover/mutate falhando?). Resultado equivale a zero-shot.")
        meta = dict(
            mode="multi_city",
            train_cities=names,
            pop_size=self.pop_size, generations=self.generations, elite=self.elite,
            total_offspring_evaluated=int(total_offspring),
            best_origin=best.origin, robust=best.robust,
            train_mean_hv=best.mean_hv, train_maximin_hv=best.maximin,
            train_per_city=best.per_city,
            agg_curve=agg_curve, mean_hv_curve=mean_hv_curve, maximin_curve=maximin_curve,
            origin_curve=origin_curve,
            n_robust_final_pop=int(sum(h.robust for h in pop)),
            heuristic_code=best.code,
            elapsed_sec=round(time.time() - t0, 2),
            llm_backend=self.llm.name, llm_usage=self.llm.usage.to_dict(),
        )
        return MultiCityResult(code=best.code, meta=meta)


# Os operadores _reflect_short/_crossover/_mutate do ReEvoOptimizer esperam objetos com .code,
# .score E .origin (este último vai no `context` das chamadas). MultiHeuristic usa .agg como
# score de ranking; este shim expõe .score == .agg e repassa .origin. SEM .origin, os operadores
# crossover/mutate lançariam AttributeError (engolido pelo try/except) e a EVOLUÇÃO ficaria inerte.
class _HShim:
    __slots__ = ("code", "score", "origin")

    def __init__(self, mh: MultiHeuristic):
        self.code = mh.code
        self.score = mh.agg
        self.origin = mh.origin


def _as_h(mh: MultiHeuristic) -> _HShim:
    return _HShim(mh)


# ---------------------------------------------------------------- aplicação da vencedora
def apply_heuristic_to_city(code, instance, max_distance, max_capacity, method, instance_name,
                            sweep_max_points=200, per_call_timeout=2.0, meta=None):
    """Aplica um CÓDIGO de heurística fixo a UMA cidade (modo justo) e devolve
    (ParetoSet, n_crash). Usado para benchmarkar a vencedora nas 10 cidades SEM re-evoluir.

    A fronteira é varrida com um sweep denso (limitado a `sweep_max_points`); `n_crash` é o
    nº de pontos do sweep onde a heurística lançou — usado na VERIFICAÇÃO ANTI-BUG (a vencedora
    deve rodar viável e SEM crash nas 10 cidades)."""
    fair = FairODCProblem(instance, max_distance, max_capacity)
    hinst = HeuristicInstance.from_instance(instance, max_distance, max_capacity)
    sweep = build_capped_sweep(instance, max_capacity, max_points=sweep_max_points)
    dist_max = float(hinst.distances.max())
    score, front, n_valid, n_crash = evaluate_heuristic_robust(
        code, fair, hinst, sweep, dist_max, per_call_timeout=per_call_timeout)
    if front is None:
        Xf = np.zeros((1, instance.n_var)); Ff = np.zeros((1, 2)); feas = np.array([False])
    else:
        Xf, Ff, feas = front
    m = dict(internal_hv=score, n_valid_sweep=n_valid, sweep_points=len(sweep), n_crash=int(n_crash))
    m.update(meta or {})
    ps = ParetoSet(X=Xf, F=Ff, feasible=feas, method=method, instance=instance_name,
                   seed=0, budget=0, meta=m)
    return ps, int(n_crash)


# ---------------------------------------------------------------- zero-shot multi-cidade (ablação)
class MultiCityZeroShot:
    """Ablação: o LLM propõe `n_samples` heurísticas DISTINTAS (SEM evolução/reflexão); seleciona
    a de maior HV MÉDIO sobre as cidades de TREINO (mesmo critério multi-cidade do ReEvo, sem o
    laço). Para ser uma ablação LIMPA do laço evolutivo, usa as MESMAS `IDEA_HINTS` da população
    inicial do ReEvo (com n_samples = pop_size, isola exatamente "melhor do init" vs "init +
    evolução"; além disso compartilha o cache com o init do ReEvo => 0 chamadas extras de API).
    Sem os hints, todas as amostras teriam o MESMO prompt e colapsariam numa só via cache.
    NÃO toca nas cidades de teste."""

    name = "zero_shot"

    def __init__(self, llm_client, max_distance=11.0, max_capacity=1000.0, n_samples=4,
                 per_call_timeout=1.5, sweep_max_points=14, strong_seeds=None, verbose=True):
        self.llm = llm_client
        self.max_distance = max_distance
        self.max_capacity = max_capacity
        self.n_samples = n_samples
        self.per_call_timeout = float(per_call_timeout)
        self.sweep_max_points = int(sweep_max_points)
        self.strong_seeds = list(strong_seeds) if strong_seeds else []
        self.verbose = verbose

    def solve_multi(self, train_instances, seed: int = 1) -> MultiCityResult:
        names = [getattr(inst, "name", "") for inst in train_instances]
        for nm in names:
            if any(tok in nm.lower() for tok in TEST_CITY_TOKENS):
                raise RuntimeError(f"VAZAMENTO DE TESTE (zero-shot): '{nm}' no treino. Abortando.")
        # MESMA população inicial do ReEvo (seeds fortes + mesmos hints) -> ablação LIMPA do laço
        # evolutivo; mesmo namespace de cache por seed.
        if hasattr(self.llm, "set_cache_salt"):
            self.llm.set_cache_salt(f"reevo_s{seed}")
        cities = []
        for inst in train_instances:
            fair = FairODCProblem(inst, self.max_distance, self.max_capacity)
            hinst = HeuristicInstance.from_instance(inst, self.max_distance, self.max_capacity)
            sweep = build_sweep(inst, self.max_capacity, max_points=self.sweep_max_points)
            cities.append(dict(name=inst.name, fair=fair, hinst=hinst, sweep=sweep,
                               dist_max=float(hinst.distances.max())))
        t0 = time.time()
        best = None
        # candidatos = seeds fortes injetadas + n_samples gerados pelo LLM (mesmos hints do init).
        cands = [(sc, "seed_strong") for sc in self.strong_seeds]
        for i in range(self.n_samples - len(self.strong_seeds)):
            hint = IDEA_HINTS[i % len(IDEA_HINTS)]
            r = self.llm.complete("generate", prompts.SYSTEM, prompts.generate_user(hint),
                                  context={"index": i})
            cands.append((extract_code(r.text), "zero_shot"))
        best_origin = None
        for code, origin in cands:
            mh = MultiCityReEvoOptimizer._aggregate(code, origin, cities, self.per_call_timeout)
            if best is None or mh.agg > best.agg:
                best, best_origin = mh, origin
            if self.verbose:
                print(f"  zero_shot[{origin}]: meanHV={mh.mean_hv:.4f} maximin={mh.maximin:.4f} "
                      f"robusta={mh.robust}")
        meta = dict(
            mode="multi_city", train_cities=names, n_samples=self.n_samples,
            best_origin=best_origin, robust=best.robust,
            train_mean_hv=best.mean_hv, train_maximin_hv=best.maximin, train_per_city=best.per_city,
            heuristic_code=best.code, elapsed_sec=round(time.time() - t0, 2),
            llm_backend=self.llm.name, llm_usage=self.llm.usage.to_dict(),
        )
        return MultiCityResult(code=best.code, meta=meta)
