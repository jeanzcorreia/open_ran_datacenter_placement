"""
src/eval/runner_llm.py — Fase 4: avalia o otimizador-LLM (ReEvo) e a ablação zero-shot no
MODO JUSTO (Natal), contra os baselines da Fase 3, e testa a GENERALIZAÇÃO Natal->Manaus
(aplica a melhor heurística evoluída em Manaus, sem re-evoluir).

Uso:
    python -m src.eval.runner_llm                       # usa configs/experiment.yaml
    python -m src.eval.runner_llm --config <path> --backend offline|anthropic|auto
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml

from ..optimizers.fair import FAIR_OPTIMIZERS
from ..optimizers.llm.llm_client import BudgetExceeded, build_llm_client
from ..optimizers.llm.reevo import ReEvoOptimizer, heuristic_to_pareto
from ..optimizers.llm.zero_shot import ZeroShotOptimizer
from ..problem.instance import load_instance_sites
from .metrics import hypervolume, igd_plus, normalize, spacing, spread
from .reference_front import build_reference_front
from .runner import _ci95, _fmt_ci

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "results", "phase4")
BASELINES = ["nsga2", "nsga3", "moead", "greedy", "random"]


def _csv(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def run_baselines(instance, seeds, md, mc, pop=300, ngen=60):
    out = []
    for m in BASELINES:
        cls = FAIR_OPTIMIZERS[m]
        opt = cls(md, mc) if m == "greedy" else cls(md, mc, pop_size=pop)
        ms = [seeds[0]] if m == "greedy" else seeds
        for s in ms:
            out.append(opt.solve(instance, budget=ngen, seed=s))
    return out


def metrics_for(ps, ref, refF_norm):
    feas = np.asarray(ps.feasible, dtype=bool)
    if not feas.any():
        return dict(HV=0.0, IGDp=float("nan"), spacing=float("nan"), spread=float("nan"), n=0)
    Fn = normalize(np.atleast_2d(ps.F)[feas], ref.ideal, ref.nadir)
    return dict(HV=hypervolume(Fn), IGDp=igd_plus(Fn, refF_norm),
                spacing=spacing(Fn), spread=spread(Fn, refF_norm), n=int(feas.sum()))


def aggregate(runs, ref, refF_norm):
    """Agrega métricas por método (média ± IC95% sobre runs)."""
    by = {}
    for ps in runs:
        by.setdefault(ps.method, []).append(metrics_for(ps, ref, refF_norm))
    agg = {}
    for m, ms in by.items():
        agg[m] = {k: _ci95([d[k] for d in ms]) for k in ("HV", "IGDp", "spacing", "spread", "n")}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "experiment.yaml"))
    ap.add_argument("--backend", default=None, help="override llm.backend (offline|anthropic|auto)")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.backend:
        cfg.setdefault("llm", {})["backend"] = args.backend

    P = cfg["problem"]
    md, mc, cpu = P["max_distance"], P["max_capacity"], P["cpu_per_100mhz"]
    train_csv, transfer_csv = _csv(P["train_csv"]), _csv(P["transfer_csv"])
    base_seeds = cfg["eval"]["baseline_seeds"]
    os.makedirs(OUT_DIR, exist_ok=True)

    # clientes LLM separados (usage limpo por método); cache de respostas compartilhado.
    reevo_client = build_llm_client(cfg, log_path=os.path.join(OUT_DIR, "reevo_calls.jsonl"))
    zs_client = build_llm_client(cfg, log_path=os.path.join(OUT_DIR, "zeroshot_calls.jsonl"))
    print(f"LLM backend: {reevo_client.name} | model={reevo_client.model} "
          f"reflection={reevo_client.reflection_model}")

    # ---------------- TREINO: Natal ----------------
    natal = load_instance_sites(train_csv, cpu_per_100mhz=cpu)
    print(f"\n=== Treino: {natal.name} ({natal.n_unique_sites} sites) ===")
    print("Baselines (Fase 3)...")
    runs = run_baselines(natal, base_seeds, md, mc)

    print("\nReEvo (otimizador-LLM)...")
    rc = cfg["reevo"]
    reevo = ReEvoOptimizer(reevo_client, md, mc, pop_size=rc["pop_size"],
                           generations=rc["generations"], elite=rc["elite"])
    try:
        ps_reevo = reevo.solve(natal, seed=rc["seed"])
    except BudgetExceeded as e:
        print(f"  budget LLM excedido: {e}"); raise
    runs.append(ps_reevo)
    ps_reevo.save(os.path.join(OUT_DIR, "Natal", "reevo"))

    print("\nZero-shot (ablação)...")
    zs = ZeroShotOptimizer(zs_client, md, mc, n_samples=4)
    ps_zs = zs.solve(natal, seed=1)
    runs.append(ps_zs)
    ps_zs.save(os.path.join(OUT_DIR, "Natal", "zero_shot"))

    ref = build_reference_front(runs)
    refF_norm = normalize(ref.F, ref.ideal, ref.nadir)
    agg = aggregate(runs, ref, refF_norm)

    # ---------------- tabela Natal ----------------
    print(f"\nFronteira de referência (Natal): {len(ref.F)} pts | ideal={ref.ideal.tolist()} "
          f"nadir={[round(x,3) for x in ref.nadir.tolist()]}")
    print("\n### Tabela Natal (modo justo) — HV/IGD+/spacing/spread + custo LLM\n")
    print("| Método | HV ↑ | IGD+ ↓ | Spacing ↓ | Spread ↓ | |front| | chamadas LLM | tokens saída | custo US$ | tempo s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    order = ["reevo", "zero_shot", "greedy", "nsga2", "nsga3", "moead", "random"]
    rows = {}
    for m in order:
        if m not in agg:
            continue
        a = agg[m]
        llm = ""
        toks = cost = calls = tsec = "-"
        ps = next((r for r in runs if r.method == m), None)
        if m in ("reevo", "zero_shot") and ps is not None:
            u = ps.meta.get("llm_usage", {})
            calls = u.get("calls", "-")
            toks = u.get("output_tokens", "-")
            cost = f"{u.get('cost_usd', 0):.3f}"
            tsec = ps.meta.get("elapsed_sec", "-")
        print(f"| {m} | {_fmt_ci(a['HV'])} | {_fmt_ci(a['IGDp'])} | {_fmt_ci(a['spacing'])} "
              f"| {_fmt_ci(a['spread'])} | {_fmt_ci(a['n'],1)} | {calls} | {toks} | {cost} | {tsec} |")
        rows[m] = a

    # ---------------- TRANSFERÊNCIA: Natal -> Manaus ----------------
    print(f"\n=== Transferência: melhor heurística do ReEvo (Natal) aplicada em Manaus ===")
    best_code = ps_reevo.meta["heuristic_code"]
    manaus = load_instance_sites(transfer_csv, cpu_per_100mhz=cpu)
    print(f"Manaus: {manaus.n_unique_sites} sites, {manaus.n_clients} clientes")
    manaus_runs = run_baselines(manaus, base_seeds, md, mc)
    ps_transfer = heuristic_to_pareto(best_code, manaus, md, mc, "reevo_transfer", manaus.name,
                                      meta={"source": "evolved_on_Natal"})
    manaus_runs.append(ps_transfer)
    ps_transfer.save(os.path.join(OUT_DIR, "Manaus", "reevo_transfer"))

    ref_m = build_reference_front(manaus_runs)
    refF_m = normalize(ref_m.F, ref_m.ideal, ref_m.nadir)
    agg_m = aggregate(manaus_runs, ref_m, refF_m)

    print(f"\nFronteira de referência (Manaus): {len(ref_m.F)} pts")
    print("\n### Transferência Natal->Manaus (modo justo)\n")
    print("| Método | HV ↑ | IGD+ ↓ | |front| |")
    print("|---|---|---|---|")
    for m in ["reevo_transfer", "greedy", "nsga2", "nsga3", "moead", "random"]:
        if m not in agg_m:
            continue
        a = agg_m[m]
        print(f"| {m} | {_fmt_ci(a['HV'])} | {_fmt_ci(a['IGDp'])} | {_fmt_ci(a['n'],1)} |")

    # ---------------- persistência ----------------
    with open(os.path.join(OUT_DIR, "winning_heuristic.py"), "w") as fh:
        fh.write(f"# Heurística vencedora evoluída pelo ReEvo (origem={ps_reevo.meta['best_origin']}, "
                 f"HV interno={ps_reevo.meta['internal_hv']:.4f}, backend={ps_reevo.meta['llm_backend']})\n")
        fh.write(best_code + "\n")
    summary = {
        "backend": reevo_client.name,
        "model": reevo_client.model, "reflection_model": reevo_client.reflection_model,
        "natal": {m: {k: list(v) for k, v in a.items()} for m, a in agg.items()},
        "manaus_transfer": {m: {k: list(v) for k, v in a.items()} for m, a in agg_m.items()},
        "reevo_meta": {k: ps_reevo.meta[k] for k in
                       ("best_origin", "internal_hv", "hv_curve", "n_valid_sweep", "elapsed_sec",
                        "llm_usage") if k in ps_reevo.meta},
        "zero_shot_meta": {k: ps_zs.meta[k] for k in ("internal_hv", "llm_usage", "elapsed_sec")
                           if k in ps_zs.meta},
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nHeurística vencedora: {os.path.join(OUT_DIR, 'winning_heuristic.py')}")
    print(f"Resumo: {os.path.join(OUT_DIR, 'summary.json')}")
    print(f"Curva de convergência (HV/geração): {ps_reevo.meta['hv_curve']}")


if __name__ == "__main__":
    main()
