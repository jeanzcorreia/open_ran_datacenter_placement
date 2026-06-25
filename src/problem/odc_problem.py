"""
src/problem/odc_problem.py — Avaliadores do placement de ODCs.

Dois modos, claramente separados:

  • ODCPlacementProblem  (MODO REPRODUÇÃO, Fase 2) — física EXATA do parser original:
      candidatos = centróides KMeans; F (3,) = [-cap·w0, n_odc·w1, dist·w2] (pesos);
      G (2,) = flags BINÁRIAS 0/1 [capacidade, distância]. NÃO alterar (validado).

  • FairODCProblem       (MODO JUSTO, Fase 3) — MO genuíno:
      candidatos = TODOS os sites únicos (n_var = n_sites); genótipo binário (limiar 0.5);
      F (2,) = [nº de ODCs ativos, distância média de fronthaul] (AMBOS minimizados);
      G (2,) = restrições com MAGNITUDE CONTÍNUA (viável ⇔ ambas == 0):
          g_cap  = Σ_ODC max(0, carga_ODC − capacity)      (sobrecarga total, cores)
          g_dist = Σ_cli max(0, dist_atribuída − max_km)     (excesso total de distância, km)
      Params do cenário: max_km=11, capacity=1000, cpuper100=14.

Ambos compartilham a mesma atribuição (cliente -> ODC ativo mais próximo, Haversine
pré-computado, sem dedupe de clientes) via `assign_clients`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance


# --------------------------------------------------------------------- atribuição comum
def assign_clients(distances: np.ndarray, cpu_cores: np.ndarray, x, threshold: float = 0.5):
    """Atribui cada cliente ao ODC ATIVO mais próximo (igual a evaluate_trial do original).

    Retorna (sel, nearest, client_dist, caps, fiber_per_odc); se nenhum ODC ativo,
    `sel.size == 0` e os demais são None. `caps`/`fiber_per_odc` indexados na ordem de `sel`.
    """
    x = np.asarray(x, dtype=float).ravel()
    sel = np.where(x > threshold)[0]
    if sel.size == 0:
        return sel, None, None, None, None
    sub = distances[:, sel]                     # (n_clients, |sel|)
    nearest = sub.argmin(axis=1)                # índice em `sel`
    client_dist = sub[np.arange(distances.shape[0]), nearest]
    caps = np.zeros(sel.size)
    np.add.at(caps, nearest, cpu_cores)
    fiber = np.zeros(sel.size)
    np.add.at(fiber, nearest, client_dist)
    return sel, nearest, client_dist, caps, fiber


@dataclass
class EvalResult:
    """Resultado da avaliação de UMA solução (contrato do otimizador-LLM)."""

    F: np.ndarray              # objetivos (3, no repro; 2, no justo)
    feasible: bool             # G == 0
    info: dict


# ============================================================ MODO REPRODUÇÃO (Fase 2)
class ODCPlacementProblem:
    """Avaliador do modo REPRODUÇÃO (física exata do parser). Ver docs/PHASE2_REPRO.md."""

    def __init__(
        self,
        instance: Instance,
        max_distance: float = 11.0,
        max_capacity: float = 1000.0,
        obj_weights: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ):
        self.instance = instance
        self.max_distance = float(max_distance)
        self.max_capacity = float(max_capacity)
        self.obj_weights = tuple(float(w) for w in obj_weights)
        self.distances = instance.distances
        self.cpu_cores = instance.cpu_cores
        self.n_clients = instance.n_clients
        self.n_var = instance.n_var
        self.n_obj = 3
        self.n_constr = 2

    def _evaluate_one(self, x: np.ndarray) -> dict:
        sel, nearest, client_dist, caps, fiber = assign_clients(self.distances, self.cpu_cores, x)
        if sel.size == 0:
            return dict(
                g_cap=1, g_dist=1, total_cap=0.0, n_active=0, avg_dist=float("inf"),
                sel=sel, caps=np.zeros(0), fiber_per_odc=np.zeros(0),
                assign=np.zeros(0, dtype=int), client_dist=np.zeros(0),
            )
        g_cap = 0 if (caps.max() <= self.max_capacity) else 1
        g_dist = 0 if (client_dist.max() <= self.max_distance) else 1
        return dict(
            g_cap=g_cap, g_dist=g_dist, total_cap=float(caps.sum()), n_active=int(sel.size),
            avg_dist=float(client_dist.mean()), sel=sel, caps=caps, fiber_per_odc=fiber,
            assign=nearest, client_dist=client_dist,
        )

    def _weighted_F(self, total_cap: float, n_active: int, avg_dist: float) -> np.ndarray:
        w0, w1, w2 = self.obj_weights
        # Peso ZERO zera o termo (evita inf*0 = nan; ver PHASE2_REPRO §5b).
        f0 = 0.0 if w0 == 0 else -total_cap * w0
        f1 = 0.0 if w1 == 0 else n_active * w1
        f2 = 0.0 if w2 == 0 else avg_dist * w2
        return np.array([f0, f1, f2], dtype=float)

    def evaluate(self, x: np.ndarray) -> EvalResult:
        r = self._evaluate_one(x)
        F = self._weighted_F(r["total_cap"], r["n_active"], r["avg_dist"])
        feasible = (r["g_cap"] == 0) and (r["g_dist"] == 0)
        if r["n_active"] > 0:
            nonempty = r["caps"] > 0
            n_nonempty = int(nonempty.sum())
            mean_cap = float(r["caps"][nonempty].mean()) if n_nonempty else 0.0
            mean_fiber_per_odc = float(r["fiber_per_odc"][nonempty].mean()) if n_nonempty else 0.0
            max_fiber = float(r["client_dist"].max())
        else:
            n_nonempty, mean_cap, mean_fiber_per_odc, max_fiber = 0, 0.0, 0.0, float("inf")
        info = dict(
            n_odc=r["n_active"], n_odc_nonempty=n_nonempty, total_cap=r["total_cap"],
            mean_cap_per_odc=mean_cap, mean_fiber_km=r["avg_dist"], max_fiber_km=max_fiber,
            mean_fiber_per_odc_km=mean_fiber_per_odc, viol=(r["g_cap"], r["g_dist"]),
        )
        return EvalResult(F=F, feasible=feasible, info=info)

    def evaluate_population(self, X: np.ndarray):
        X = np.atleast_2d(X)
        n = X.shape[0]
        F = np.zeros((n, 3))
        G = np.zeros((n, 2))
        for i in range(n):
            r = self._evaluate_one(X[i])
            F[i] = self._weighted_F(r["total_cap"], r["n_active"], r["avg_dist"])
            G[i, 0] = r["g_cap"]
            G[i, 1] = r["g_dist"]
        return F, G

    def to_pymoo(self):
        from pymoo.core.problem import Problem

        odcp = self

        class _PymooODCProblem(Problem):
            def __init__(self):
                super().__init__(n_var=odcp.n_var, n_obj=3, n_constr=2, xl=0, xu=1)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = odcp.evaluate_population(X)
                out["F"] = F
                out["G"] = G

        return _PymooODCProblem()


# ================================================================== MODO JUSTO (Fase 3)
class FairODCProblem:
    """Avaliador do MODO JUSTO: MO genuíno [nº ODCs, distância média] com restrições de
    magnitude contínua. Candidatos = sites únicos (use `load_instance_sites`)."""

    def __init__(self, instance: Instance, max_distance: float = 11.0, max_capacity: float = 1000.0):
        self.instance = instance
        self.max_distance = float(max_distance)
        self.max_capacity = float(max_capacity)
        self.distances = instance.distances
        self.cpu_cores = instance.cpu_cores
        self.n_clients = instance.n_clients
        self.n_var = instance.n_var
        self.n_obj = 2
        self.n_constr = 2
        # sentinela FINITA para o caso degenerado "sem ODC" (evita inf/nan nas métricas):
        self._dist_sentinel = float(self.distances.max()) if self.distances.size else 1.0e3
        self._total_demand = float(self.cpu_cores.sum())

    def _evaluate_one(self, x: np.ndarray) -> dict:
        sel, nearest, client_dist, caps, fiber = assign_clients(self.distances, self.cpu_cores, x)
        if sel.size == 0:
            # Sem ODC: maximamente inviável, mas com valores FINITOS (não entra na fronteira).
            big = float(self.n_clients) * self._dist_sentinel
            return dict(
                f1=0.0, f2=self._dist_sentinel, g_cap=self._total_demand, g_dist=big,
                n_active=0, n_nonempty=0, total_overload=self._total_demand,
                total_dist_excess=big, max_load=0.0, mean_fiber=self._dist_sentinel,
            )
        overload = float(np.maximum(0.0, caps - self.max_capacity).sum())
        dist_excess = float(np.maximum(0.0, client_dist - self.max_distance).sum())
        return dict(
            f1=float(sel.size),                       # nº de ODCs ativos (selecionados)
            f2=float(client_dist.mean()),             # distância média de fronthaul
            g_cap=overload,
            g_dist=dist_excess,
            n_active=int(sel.size),
            n_nonempty=int((caps > 0).sum()),
            total_overload=overload,
            total_dist_excess=dist_excess,
            max_load=float(caps.max()),
            mean_fiber=float(client_dist.mean()),
        )

    def evaluate(self, x: np.ndarray) -> EvalResult:
        r = self._evaluate_one(x)
        F = np.array([r["f1"], r["f2"]], dtype=float)
        feasible = (r["g_cap"] == 0.0) and (r["g_dist"] == 0.0)
        info = dict(
            n_odc=r["n_active"],
            n_odc_nonempty=r["n_nonempty"],
            mean_fiber_km=r["mean_fiber"],
            total_overload=r["total_overload"],
            total_dist_excess=r["total_dist_excess"],
            max_load_per_odc=r["max_load"],
            viol=(r["g_cap"], r["g_dist"]),
        )
        return EvalResult(F=F, feasible=feasible, info=info)

    def evaluate_population(self, X: np.ndarray):
        X = np.atleast_2d(X)
        n = X.shape[0]
        F = np.zeros((n, 2))
        G = np.zeros((n, 2))
        for i in range(n):
            r = self._evaluate_one(X[i])
            F[i, 0] = r["f1"]
            F[i, 1] = r["f2"]
            G[i, 0] = r["g_cap"]
            G[i, 1] = r["g_dist"]
        return F, G

    def to_pymoo(self):
        from pymoo.core.problem import Problem

        fp = self

        class _PymooFairProblem(Problem):
            def __init__(self):
                super().__init__(n_var=fp.n_var, n_obj=2, n_constr=2, xl=0, xu=1)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = fp.evaluate_population(X)
                out["F"] = F
                out["G"] = G

        return _PymooFairProblem()

    def to_pymoo_penalized(self, penalty: float = 1000.0):
        """Variante SEM restrições, com penalidade estática somada aos DOIS objetivos.

        Necessária só para o MOEA/D do pymoo 0.6.1.3, que NÃO suporta restrições. A busca
        usa esta penalização; ao extrair a fronteira, as soluções são reavaliadas com o
        `evaluate`/`evaluate_population` VERDADEIRO (objetivos e viabilidade idênticos aos
        demais métodos), preservando a comparação justa. Soluções viáveis (violação 0) ficam
        com F inalterado; inviáveis recebem `penalty·(g_cap+g_dist)` em ambos objetivos."""
        from pymoo.core.problem import Problem

        fp = self

        class _PymooFairPenalized(Problem):
            def __init__(self):
                super().__init__(n_var=fp.n_var, n_obj=2, n_constr=0, xl=0, xu=1)

            def _evaluate(self, X, out, *args, **kwargs):
                F, G = fp.evaluate_population(X)
                viol = np.maximum(0.0, G).sum(axis=1, keepdims=True)
                out["F"] = F + penalty * viol

        return _PymooFairPenalized()
