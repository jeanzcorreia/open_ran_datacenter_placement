"""
src/reproduce_natal.py — Harness de REPRODUÇÃO (Fase 2).

Roda o baseline NSGA-II encapsulado (src/) sobre Natal, nos cenários de candidatos
k ∈ {27, 18, 13, 0} (RUs/2, RUs/3, RUs/4 e "ODCs=O-RUs"; RUs = 55 sites únicos), e compara
as métricas operacionais com os resultados dos autores em Results/ — nº de ODCs ativos,
capacidade média/ODC e distância média cliente↔ODC (proxy de fibra).

Uso:
    python -m src.reproduce_natal              # roda todos os cenários
    python -m src.reproduce_natal --k 27       # só um cenário
    python -m src.reproduce_natal --no-run     # só recomputa a referência de Results/

(Será substituído por src/eval/runner.py na Fase 3.)
"""

from __future__ import annotations

import argparse
import ast
import os

import numpy as np
import pandas as pd

from .optimizers.nsga2 import NSGA2Optimizer
from .problem.instance import load_instance
from .problem.odc_problem import ODCPlacementProblem

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NATAL_CSV = os.path.join(REPO, "CityData", "Natal.csv")
RESULTS_DIR = os.path.join(REPO, "Results", "results_Placement_Natal_Case_1_2_odcs")
OUT_DIR = os.path.join(REPO, "results", "phase2")

# Cenário do paper (das CAMPANHAS, não dos defaults da CLI):
CPU_PER_100MHZ = 14
MAX_DISTANCE = 11.0
MAX_CAPACITY = 1000.0
OBJ_WEIGHTS = (0.0, 0.0, 1.0)   # wcpu=0, wodc=0, wd=1 (cenário de distância pura)
POP_SIZE = 300
N_GEN = 60

# k -> índice de Sim em Results/ (grupo de pesos (0,0,1)); ver PHASE1/PHASE2.
K_TO_SIM = {0: 0, 27: 1, 18: 2, 13: 3}
# Referência APROXIMADA citada do paper (Natal). Ver PHASE2_REPRO §divergências.
PAPER_NATAL = dict(n_odc="~13", mean_fiber_km="~3.76", mean_cap_per_odc="~105")


def aggregate_results_reference(k: int) -> dict | None:
    """Lê Results/ e agrega (média sobre JOBs) as métricas operacionais do cenário k."""
    sim = K_TO_SIM[k]
    if not os.path.isdir(RESULTS_DIR):
        return None
    jobs = [d for d in os.listdir(RESULTS_DIR) if d.startswith("JOB")]
    n_act, mean_cap, client_dist, fiber_per_odc = [], [], [], []
    for j in jobs:
        cap = os.path.join(RESULTS_DIR, j, f"Sim_{sim}", "df_capacities.csv")
        fib = os.path.join(RESULTS_DIR, j, f"Sim_{sim}", "df_fiberlength.csv")
        ca = os.path.join(RESULTS_DIR, j, f"Sim_{sim}", "df_client_association.csv")
        if not (os.path.exists(cap) and os.path.exists(fib) and os.path.exists(ca)):
            continue
        dc = pd.read_csv(cap)
        df = pd.read_csv(fib)
        nclients = len(pd.read_csv(ca))
        active = dc[dc["capacities"] > 0]
        n_act.append(len(active))
        mean_cap.append(active["capacities"].mean())
        client_dist.append(df["fiberlength"].sum() / nclients)
        nz = df[df["fiberlength"] > 0]["fiberlength"]
        fiber_per_odc.append(nz.mean() if len(nz) else np.nan)
    if not n_act:
        return None
    return dict(
        n_runs=len(n_act),
        n_odc=float(np.mean(n_act)),
        n_odc_std=float(np.std(n_act)),
        mean_cap_per_odc=float(np.mean(mean_cap)),
        mean_fiber_km=float(np.nanmean(client_dist)),
        mean_fiber_per_odc_km=float(np.nanmean(fiber_per_odc)),
    )


def representative_solution(pareto, problem: ODCPlacementProblem):
    """Solução representativa = entre as VIÁVEIS, a de menor distância (F[2])."""
    F = pareto.F
    feas = pareto.feasible
    idx_pool = np.where(feas)[0] if feas.any() else np.arange(len(F))
    best = idx_pool[np.argmin(F[idx_pool, 2])]
    x = pareto.X[best]
    return x, problem.evaluate(x)


def run_scenario(k: int, seed: int = 1) -> dict:
    inst = load_instance(NATAL_CSV, k=k, cpu_per_100mhz=CPU_PER_100MHZ)
    problem = ODCPlacementProblem(inst, MAX_DISTANCE, MAX_CAPACITY, OBJ_WEIGHTS)
    opt = NSGA2Optimizer(MAX_DISTANCE, MAX_CAPACITY, OBJ_WEIGHTS, pop_size=POP_SIZE, period=N_GEN)
    pareto = opt.solve(inst, budget=N_GEN, seed=seed)

    # persiste a fronteira
    scen_dir = os.path.join(OUT_DIR, inst.name)
    pareto.save(scen_dir)

    x, ev = representative_solution(pareto, problem)
    ref = aggregate_results_reference(k)

    return dict(
        instance=inst.summary(),
        k=k,
        n_var=inst.n_var,
        seed=seed,
        front_size=pareto.n_solutions,
        front_source=pareto.meta.get("front_source"),
        n_gen=pareto.meta.get("n_gen"),
        n_eval=pareto.meta.get("n_eval"),
        elapsed_sec=pareto.meta.get("elapsed_sec"),
        ours=dict(
            n_odc_selected=ev.info["n_odc"],
            n_odc_active=ev.info["n_odc_nonempty"],
            mean_cap_per_odc=ev.info["mean_cap_per_odc"],
            mean_fiber_km=ev.info["mean_fiber_km"],
            mean_fiber_per_odc_km=ev.info["mean_fiber_per_odc_km"],
            max_fiber_km=ev.info["max_fiber_km"],
            feasible=ev.feasible,
            total_cap=ev.info["total_cap"],
        ),
        results_ref=ref,
        saved_to=scen_dir,
    )


def _fmt(v, nd=4):
    # np.int64 NÃO é subclasse de int (np.float64 é de float) — inclui os tipos numpy.
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="rodar só um cenário (27/18/13/0)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-run", action="store_true", help="só recomputar a referência de Results/")
    args = ap.parse_args()

    ks = [args.k] if args.k is not None else [27, 18, 13, 0]

    if args.no_run:
        for k in ks:
            print(k, aggregate_results_reference(k))
        return

    rows = []
    for k in ks:
        print(f"\n=== Natal cenário k={k} (RUs/{ {27:2,18:3,13:4,0:'(=O-RUs)'}.get(k) }) ===")
        r = run_scenario(k, seed=args.seed)
        rows.append(r)
        o, ref = r["ours"], r["results_ref"]
        print(f"  n_var (candidatos)        : {r['n_var']}")
        print(f"  gerações / avaliações     : {r['n_gen']} / {r['n_eval']}  ({r['elapsed_sec']}s)")
        print(f"  fronteira (|nd|, fonte)    : {r['front_size']} ({r['front_source']})")
        print(f"  NOSSO  n_odc ativos        : {o['n_odc_active']}  (selecionados={o['n_odc_selected']})")
        print(f"  NOSSO  cap média/ODC       : {_fmt(o['mean_cap_per_odc'],2)} cores")
        print(f"  NOSSO  dist média cli->ODC : {_fmt(o['mean_fiber_km'])} km   (viável={o['feasible']})")
        if ref:
            print(f"  Results n_odc ativos       : {_fmt(ref['n_odc'],2)} (±{_fmt(ref['n_odc_std'],2)}, n={ref['n_runs']})")
            print(f"  Results cap média/ODC      : {_fmt(ref['mean_cap_per_odc'],2)} cores")
            print(f"  Results dist média cli->ODC: {_fmt(ref['mean_fiber_km'])} km")

    # tabela markdown comparativa
    print("\n\n### Tabela de reprodução (Natal)\n")
    print("| Cenário (k) | n_odc nosso | n_odc Results | cap/ODC nosso | cap/ODC Results | dist nosso (km) | dist Results (km) | Δdist |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        o, ref = r["ours"], r["results_ref"]
        ddist = abs(o["mean_fiber_km"] - ref["mean_fiber_km"]) if ref else float("nan")
        print(
            f"| RUs/{ {27:2,18:3,13:4,0:'∞'}.get(r['k']) } (k={r['k']}→{r['n_var']}) "
            f"| {o['n_odc_active']} | {_fmt(ref['n_odc'],1) if ref else '-'} "
            f"| {_fmt(o['mean_cap_per_odc'],2)} | {_fmt(ref['mean_cap_per_odc'],2) if ref else '-'} "
            f"| {_fmt(o['mean_fiber_km'])} | {_fmt(ref['mean_fiber_km']) if ref else '-'} "
            f"| {_fmt(ddist)} |"
        )
    print(f"\nReferência (paper, Natal, aprox.): {PAPER_NATAL}")
    print(f"Artefatos salvos em: {OUT_DIR}")


if __name__ == "__main__":
    main()
