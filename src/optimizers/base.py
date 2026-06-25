"""
src/optimizers/base.py — Interface comum dos otimizadores e o contêiner ParetoSet.

ParetoSet PERSISTE a fronteira (X e F) — algo que o parser original NÃO faz (ele só salva
a melhor solução da última geração). Persistir é HARD RULE da Fase 2 (CLAUDE.md §9.5),
necessário para HV/IGD+ na Fase 3.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ParetoSet:
    """Conjunto de soluções não-dominadas devolvido por um otimizador.

    X        : (n_sol, n_var) variáveis de decisão (reais em [0,1])
    F        : (n_sol, n_obj) objetivos (mesma convenção do problema; F[0] já negativo)
    feasible : (n_sol,) bool — viabilidade (G == 0)
    method   : nome do otimizador
    instance : nome da instância
    seed     : seed usada
    budget   : nº máximo de gerações / orçamento
    meta     : dict livre (n_eval, tempo, params do cenário, etc.)
    """

    X: np.ndarray
    F: np.ndarray
    feasible: np.ndarray
    method: str
    instance: str
    seed: int
    budget: int
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.X = np.atleast_2d(np.asarray(self.X, dtype=float))
        self.F = np.atleast_2d(np.asarray(self.F, dtype=float))
        self.feasible = np.asarray(self.feasible, dtype=bool).ravel()

    @property
    def n_solutions(self) -> int:
        return self.F.shape[0]

    def save(self, out_dir: str) -> str:
        """Salva X, F, feasible (.npz) + metadados (.json). Retorna o diretório."""
        os.makedirs(out_dir, exist_ok=True)
        np.savez(
            os.path.join(out_dir, "pareto.npz"),
            X=self.X,
            F=self.F,
            feasible=self.feasible,
        )
        meta = {
            "method": self.method,
            "instance": self.instance,
            "seed": self.seed,
            "budget": self.budget,
            "n_solutions": self.n_solutions,
            "n_var": int(self.X.shape[1]),
            "n_obj": int(self.F.shape[1]),
            "meta": _jsonable(self.meta),
        }
        with open(os.path.join(out_dir, "pareto_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        return out_dir

    @classmethod
    def load(cls, out_dir: str) -> "ParetoSet":
        data = np.load(os.path.join(out_dir, "pareto.npz"))
        with open(os.path.join(out_dir, "pareto_meta.json")) as fh:
            meta = json.load(fh)
        return cls(
            X=data["X"],
            F=data["F"],
            feasible=data["feasible"],
            method=meta["method"],
            instance=meta["instance"],
            seed=meta["seed"],
            budget=meta["budget"],
            meta=meta.get("meta", {}),
        )


def _jsonable(obj):
    """Converte recursivamente np.* / tuplas para tipos serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def extract_front_X(res):
    """Recupera as VARIÁVEIS de decisão da fronteira de forma robusta.

    Prioridade: res.opt -> res.X/res.F -> união não-dominada do histórico. Retorna
    (X, source). Os OBJETIVOS devem ser recomputados no problema verdadeiro pelo chamador
    (importante p/ MOEA/D, cuja busca usa F penalizado)."""
    import numpy as np

    opt = getattr(res, "opt", None)
    if opt is not None and len(opt) > 0:
        X = np.atleast_2d(opt.get("X"))
        if X.size:
            return X, "res.opt"
    if res.X is not None:
        return np.atleast_2d(res.X), "res.X"
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
    return Xall[nd], "history-union"


def feasible_nd_front(X, F, G):
    """Reduz (X, F) ao conjunto VIÁVEL e NÃO-DOMINADO (G<=0 em todas as colunas).

    Retorna (X_front, F_front, feasible_mask). Se não houver viável, devolve (X, F) cru com
    máscara toda-False (excluído da fronteira de referência e com HV 0)."""
    import numpy as np
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    feas = np.all(np.asarray(G) <= 0, axis=1)
    if not feas.any():
        return np.atleast_2d(X), np.atleast_2d(F), feas
    Xf, Ff = np.atleast_2d(X)[feas], np.atleast_2d(F)[feas]
    nd = NonDominatedSorting().do(Ff, only_non_dominated_front=True)
    Xnd, Fnd = Xf[nd], Ff[nd]
    # A fronteira é um CONJUNTO no espaço de objetivos: dedup por vetor F único (várias
    # soluções X podem mapear no mesmo (n_odc, dist); duplicatas distorceriam spacing/spread).
    _, uniq = np.unique(Fnd.round(9), axis=0, return_index=True)
    uniq = np.sort(uniq)
    return Xnd[uniq], Fnd[uniq], np.ones(len(uniq), dtype=bool)


class Optimizer(ABC):
    """Interface-contrato (CLAUDE.md §6): solve(instance, budget, seed) -> ParetoSet."""

    name: str = "optimizer"

    @abstractmethod
    def solve(self, instance, budget: int, seed: int) -> ParetoSet:  # pragma: no cover
        ...
