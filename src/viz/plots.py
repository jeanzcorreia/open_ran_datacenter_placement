"""
src/viz/plots.py — Plota as fronteiras de Pareto do MODO JUSTO (Natal) já persistidas em
results/phase3/. Eixo X = nº de ODCs (f1), eixo Y = distância média de fronthaul km (f2).

Uso: python -m src.viz.plots
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..eval.reference_front import build_reference_front
from ..optimizers.base import ParetoSet

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PH3 = os.path.join(REPO, "results", "phase3", "Natal_sites")
METHODS = ["nsga2", "nsga3", "moead", "random", "greedy"]
_STYLE = {
    "nsga2": ("tab:blue", "o"),
    "nsga3": ("tab:green", "s"),
    "moead": ("tab:orange", "^"),
    "random": ("tab:red", "x"),
    "greedy": ("tab:purple", "D"),
}


def _load_seed1(method):
    d = os.path.join(PH3, method, "seed1")
    return ParetoSet.load(d) if os.path.isdir(d) else None


def plot_fronts(out_path=None):
    sets = {m: _load_seed1(m) for m in METHODS}
    sets = {m: ps for m, ps in sets.items() if ps is not None}
    if not sets:
        raise RuntimeError(f"Nenhuma fronteira em {PH3}. Rode `python -m src.eval.runner` antes.")

    ref = build_reference_front(list(sets.values()))

    fig, ax = plt.subplots(figsize=(8, 6))
    rf = ref.F[np.argsort(ref.F[:, 0])]
    ax.plot(rf[:, 0], rf[:, 1], "-", color="black", lw=1.0, alpha=0.5, label="fronteira de referência", zorder=1)
    for m, ps in sets.items():
        feas = np.asarray(ps.feasible, dtype=bool)
        if not feas.any():
            continue
        F = np.atleast_2d(ps.F)[feas]
        F = F[np.argsort(F[:, 0])]
        c, mk = _STYLE.get(m, ("gray", "."))
        ax.scatter(F[:, 0], F[:, 1], s=22, color=c, marker=mk, alpha=0.8, label=f"{m} (|{int(feas.sum())}|)", zorder=2)

    ax.set_xlabel("nº de ODCs ativos  (f1, minimizar)")
    ax.set_ylabel("distância média de fronthaul [km]  (f2, minimizar)")
    ax.set_title("Fronteiras de Pareto — Natal, modo justo (seed 1)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    if out_path is None:
        out_path = os.path.join(PH3, "fronts.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    p = plot_fronts()
    print(f"Plot salvo em: {p}")


if __name__ == "__main__":
    main()
