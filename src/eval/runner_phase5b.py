"""
src/eval/runner_phase5b.py — Fase 5b: OBJETIVO POR LINGUAGEM NATURAL (balanceamento de carga).

Mostra que adicionar um objetivo (balanceamento de carga) descrevendo-o em PORTUGUÊS no prompt faz
o LLM REGENERAR a heurística para considerá-lo — SEM tocar na função de fitness — enquanto os
baselines exigiriam editar a fitness à mão.

Duas condições, mesma(s) cidade(s) de treino, mesmo 7B local, MESMO setup (só o prompt difere):
  (A) BASELINE-NL : ReEvo com o system prompt ORIGINAL (nº ODCs + distância média).
  (B) BALANCED-NL : ReEvo com o system prompt + a descrição de balanceamento (nl_objectives).
Sem seed forte injetada nesta experiência: a vencedora É a heurística GERADA pelo LLM (comparação
direta de código A vs B). O fitness do ReEvo continua HV(nº ODCs, distância) nas duas — o efeito de
balanceamento é medido FORA do fitness (índice de desbalanceamento).

Uso:
    python -m src.eval.runner_phase5b --config configs/phase5a.yaml --out results/phase5b --seeds 1,2
    python -m src.eval.runner_phase5b --smoke --out results/phase5b_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import yaml

from ..optimizers.fair import GreedyFair
from ..optimizers.llm import prompts
from ..optimizers.llm.heuristic_runtime import HeuristicInstance, HeuristicSandbox, SandboxError, validate_code
from ..optimizers.llm.llm_client import BudgetExceeded, build_llm_client
from ..optimizers.llm.nl_objectives import assign_loads, balanced_system, load_imbalance
from ..optimizers.llm.reevo import IDEA_HINTS, extract_code
from ..optimizers.llm.reevo_multicity import MultiCityReEvoOptimizer, build_capped_sweep
from ..problem.instance import load_instance_sites
from ..problem.odc_problem import FairODCProblem

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _csv(city_key):
    return os.path.join(REPO, "data", "processed", f"{city_key}.csv")


# --------------------------------------------------------- ReEvo com system prompt por condição
class _PromptedReEvo(MultiCityReEvoOptimizer):
    """MultiCityReEvoOptimizer que usa um `system_prompt` configurável em TODOS os operadores LLM
    (geração/reflexão/crossover/mutação). Não altera reevo.py — só sobrescreve os operadores."""

    def __init__(self, *a, system_prompt=None, **k):
        super().__init__(*a, **k)
        self._system = system_prompt or prompts.SYSTEM

    def _gen(self, idx, hint=None):
        if hint is None:
            hint = IDEA_HINTS[idx % len(IDEA_HINTS)]
        r = self.llm.complete("generate", self._system, prompts.generate_user(hint), context={"index": idx})
        return extract_code(r.text)

    def _reflect_short(self, better, worse):
        r = self.llm.complete("reflect_short", self._system,
                              prompts.reflect_short_user(better.code, better.score, worse.code, worse.score),
                              context={"op": "reflect_short"}, model=self.llm.reflection_model)
        return r.text.strip()

    def _reflect_long(self, shorts):
        r = self.llm.complete("reflect_long", self._system, prompts.reflect_long_user(shorts),
                              context={"op": "reflect_long"}, model=self.llm.reflection_model)
        return r.text.strip()

    def _crossover(self, a, b, long_refl):
        r = self.llm.complete("crossover", self._system, prompts.crossover_user(a.code, b.code, long_refl),
                              context={"parents": [a.origin, b.origin]})
        return extract_code(r.text)

    def _mutate(self, p, long_refl):
        r = self.llm.complete("mutate", self._system, prompts.mutate_user(p.code, long_refl),
                              context={"origin": p.origin})
        return extract_code(r.text)


# --------------------------------------------------------- avaliação: fronteira + desbalanceamento
def evaluate_front_balance(code, instance, hinst, fair, max_capacity, sweep_max_points, timeout):
    """Varre n_active; para cada ponto roda a heurística (sandbox de PROCESSO) e mede f1=nº ODCs,
    f2=distância média, viabilidade E o desbalanceamento de carga. Retorna lista de registros."""
    recs = []
    try:
        validate_code(code)
    except SandboxError:
        return recs
    sweep = build_capped_sweep(instance, max_capacity, max_points=sweep_max_points)
    with HeuristicSandbox(hinst) as sb:
        for n in sweep:
            try:
                idx = sb.run(code, n, timeout)
            except SandboxError:
                continue
            x = np.zeros(hinst.n_sites)
            x[idx] = 1.0
            ev = fair.evaluate(x)
            loads, sel, _, _ = assign_loads(idx, hinst)
            imb = load_imbalance(loads)
            recs.append(dict(n_active=int(n), n_odc=int(sel.size),
                             f1=float(ev.F[0]), f2=float(ev.F[1]), feasible=bool(ev.feasible),
                             max_over_mean=imb["max_over_mean"], cv=imb["cv"], jain=imb["jain"],
                             max_load=imb["max_load"], mean_load=imb["mean_load"]))
    return recs


def greedy_front_balance(instance, hinst, fair, md, mc):
    """Fronteira do greedy (referência) com desbalanceamento por solução."""
    gs = GreedyFair(md, mc).solve(instance)
    feas = np.asarray(gs.feasible, dtype=bool).ravel()
    recs = []
    for i in range(np.atleast_2d(gs.X).shape[0]):
        if not feas[i]:
            continue
        x = np.atleast_2d(gs.X)[i]
        sel = np.where(np.asarray(x) > 0.5)[0]
        if sel.size == 0:
            continue
        loads, s, _, _ = assign_loads(sel, hinst)
        imb = load_imbalance(loads)
        F = np.atleast_2d(gs.F)[i]
        recs.append(dict(n_odc=int(sel.size), f1=float(F[0]), f2=float(F[1]), feasible=True,
                         max_over_mean=imb["max_over_mean"], cv=imb["cv"], jain=imb["jain"]))
    return recs


def _feasible_summary(recs):
    """Média do desbalanceamento e da distância sobre os pontos VIÁVEIS de uma fronteira."""
    f = [r for r in recs if r["feasible"] and np.isfinite(r.get("max_over_mean", np.nan))]
    if not f:
        return dict(n_feasible=0, mean_imbalance=float("nan"), mean_jain=float("nan"),
                    mean_f2=float("nan"), min_n_odc=None, max_n_odc=None)
    return dict(
        n_feasible=len(f),
        mean_imbalance=float(np.mean([r["max_over_mean"] for r in f])),
        mean_jain=float(np.mean([r["jain"] for r in f])),
        mean_f2=float(np.mean([r["f2"] for r in f])),
        min_n_odc=int(min(r["n_odc"] for r in f)),
        max_n_odc=int(max(r["n_odc"] for r in f)),
    )


def _matched_comparison(recs_a, recs_b):
    """Compara A vs B no MESMO nº de ODCs (n_odc): para cada n_odc viável nos dois, devolve
    (n_odc, f2_A, f2_B, imb_A, imb_B). Permite ver o trade-off a igual nº de ODCs."""
    def by_nodc(recs):
        d = {}
        for r in recs:
            if r["feasible"] and np.isfinite(r.get("max_over_mean", np.nan)):
                d.setdefault(r["n_odc"], r)   # 1º viável por n_odc
        return d
    A, B = by_nodc(recs_a), by_nodc(recs_b)
    rows = []
    for k in sorted(set(A) & set(B)):
        rows.append(dict(n_odc=k, f2_A=A[k]["f2"], f2_B=B[k]["f2"],
                         imb_A=A[k]["max_over_mean"], imb_B=B[k]["max_over_mean"]))
    return rows


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "phase5a.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase5b"))
    ap.add_argument("--cities", default="Natal,BeloHorizonte")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gen", type=int, default=6)
    ap.add_argument("--loop-timeout", type=float, default=1.5,
                    help="timeout (s) por chamada no laço (5b: generoso p/ a heurística balanceada sobreviver)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    P = cfg["problem"]
    md, mc, cpu = P["max_distance"], P["max_capacity"], P["cpu_per_100mhz"]
    EV = cfg.get("eval", {})
    loop_to = float(args.loop_timeout)
    loop_sweep = EV.get("loop_sweep_points", 8)
    apply_to = EV.get("apply_per_call_timeout", 2.5)
    sweep_max = EV.get("sweep_max_points", 200)

    cities = [c for c in args.cities.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    pop, gen = args.pop, args.gen
    if args.smoke:
        cities = ["Natal"]
        seeds = [1]
        pop, gen = 6, 2
        sweep_max = 40
        print("[SMOKE 5b] 1 cidade, pop=6 gen=2, 1 seed.")

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    ckpt_dir = os.path.join(OUT, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("=" * 78)
    print("FASE 5b — OBJETIVO NL (balanceamento de carga): A (sem) vs B (com)")
    print(f"cidades={cities} seeds={seeds} pop={pop} gen={gen} | timeouts loop={loop_to}s apply={apply_to}s")
    print("=" * 78)

    insts = {}
    hinsts = {}
    fairs = {}
    for c in cities:
        inst = load_instance_sites(_csv(c), cpu_per_100mhz=cpu)
        insts[c] = inst
        hinsts[c] = HeuristicInstance.from_instance(inst, md, mc)
        fairs[c] = FairODCProblem(inst, md, mc)
        print(f"  {c:16}: {inst.n_unique_sites} sites, {inst.n_clients} clientes")
    train_insts = [insts[c] for c in cities]

    CONDITIONS = {
        "A_baseline": prompts.SYSTEM,                       # prompt ORIGINAL
        "B_balanced": balanced_system(prompts.SYSTEM),      # prompt + balanceamento (PT)
    }

    client = build_llm_client(cfg, log_path=os.path.join(OUT, "reevo5b_calls.jsonl"))
    print(f"\nLLM: {client.name} | model={client.model}")

    # ---- TREINO: ReEvo por (condição × seed), sem seed forte (vencedora = heurística do LLM) ----
    runs = {}   # cond -> seed -> {code, meta}
    for cond, system in CONDITIONS.items():
        runs[cond] = {}
        print(f"\n=== CONDIÇÃO {cond} ===")
        for s in seeds:
            ckpt = os.path.join(ckpt_dir, f"{cond}_seed{s}.json")
            if os.path.exists(ckpt):
                runs[cond][s] = json.load(open(ckpt, encoding="utf-8"))
                print(f"  [{cond} seed {s}] checkpoint -> pulando treino")
                continue
            opt = _PromptedReEvo(client, md, mc, pop_size=pop, generations=gen, elite=min(2, pop),
                                 per_call_timeout=loop_to, sweep_max_points=loop_sweep,
                                 strong_seeds=[], system_prompt=system, verbose=True)
            try:
                res = opt.solve_multi(train_insts, seed=s)
            except BudgetExceeded as e:
                print(f"  budget LLM excedido: {e}"); raise
            popu = [dict(code=h.code, agg=float(h.agg), origin=h.origin,
                         load_aware=("client_demand" in h.code))
                    for h in getattr(opt, "final_population", [])]
            runs[cond][s] = dict(code=res.code, meta=res.meta, population=popu)
            json.dump(runs[cond][s], open(ckpt, "w", encoding="utf-8"), default=float)
            m = res.meta
            print(f"  [{cond} seed {s}] origem={m['best_origin']} meanHV={m['train_mean_hv']:.4f} "
                  f"offspring={m.get('total_offspring_evaluated')} | salvo")

    # ---- representante por condição ----
    # A: vencedora por HV (não foi pedida p/ balancear). B: MELHOR heurística que REALMENTE computa
    # carga (client_demand) — é o que o prompt induziu; se nenhuma, cai p/ a vencedora HV (e sinaliza).
    def _all_pop(cond):
        out = []
        for s in seeds:
            for hh in runs[cond][s].get("population", []):
                out.append(dict(seed=s, **hh))
        return out

    rep, rep_note = {}, {}
    for cond in CONDITIONS:
        bs = max(seeds, key=lambda s: runs[cond][s]["meta"]["train_mean_hv"])
        hv_winner = dict(seed=bs, code=runs[cond][bs]["code"], meta=runs[cond][bs]["meta"])
        if cond == "B_balanced":
            cand = [h for h in _all_pop(cond) if h["load_aware"] and h["agg"] > 0.0]
            if cand:
                best = max(cand, key=lambda h: h["agg"])
                rep[cond] = dict(seed=best["seed"], code=best["code"], meta=runs[cond][best["seed"]]["meta"])
                rep_note[cond] = f"melhor heuristica load-aware (usa client_demand) agg={best['agg']:.4f}"
            else:
                rep[cond] = hv_winner
                rep_note[cond] = "NENHUMA heuristica de B computou carga (client_demand) -> usando vencedora HV"
        else:
            rep[cond] = hv_winner
            rep_note[cond] = "vencedora por HV"
        with open(os.path.join(OUT, f"winner_{cond}.py"), "w", encoding="utf-8") as fh:
            fh.write(f"# representante {cond} -- {rep_note[cond]}\n{rep[cond]['code']}\n")
        print(f"  representante {cond}: {rep_note[cond]}")

    # ---- APLICAÇÃO: fronteira + desbalanceamento por cidade, A vs B (+ greedy referência) ----
    print("\n--- Aplicando vencedoras às cidades (fronteira + desbalanceamento) ---")
    fronts = {cond: {} for cond in CONDITIONS}
    greedy = {}
    summaries = {cond: {} for cond in CONDITIONS}
    matched = {}
    for c in cities:
        for cond in CONDITIONS:
            t0 = time.time()
            recs = evaluate_front_balance(rep[cond]["code"], insts[c], hinsts[c], fairs[c],
                                          mc, sweep_max, apply_to)
            fronts[cond][c] = recs
            summaries[cond][c] = _feasible_summary(recs)
            print(f"  {c:14} {cond:11}: {summaries[cond][c]['n_feasible']} viáveis | "
                  f"imbalance(med)={summaries[cond][c]['mean_imbalance']:.3f} "
                  f"f2(med)={summaries[cond][c]['mean_f2']:.3f} ({time.time()-t0:.0f}s)")
        greedy[c] = greedy_front_balance(insts[c], hinsts[c], fairs[c], md, mc)
        gs = _feasible_summary(greedy[c])
        print(f"  {c:14} {'greedy':11}: {gs['n_feasible']} viáveis | imbalance(med)={gs['mean_imbalance']:.3f} "
              f"f2(med)={gs['mean_f2']:.3f}")
        matched[c] = _matched_comparison(fronts["A_baseline"][c], fronts["B_balanced"][c])

    # ---- persistência ----
    report = dict(
        cities=cities, seeds=seeds, pop=pop, gen=gen,
        instances={c: dict(n_sites=insts[c].n_unique_sites, n_clients=insts[c].n_clients) for c in cities},
        conditions=list(CONDITIONS),
        winner_code={cond: rep[cond]["code"] for cond in CONDITIONS},
        winner_meta={cond: rep[cond]["meta"] for cond in CONDITIONS},
        per_seed={cond: {s: runs[cond][s]["meta"] for s in seeds} for cond in CONDITIONS},
        representative_note=rep_note,
        population={cond: _all_pop(cond) for cond in CONDITIONS},
        fronts=fronts, greedy=greedy,
        summaries=summaries,
        greedy_summary={c: _feasible_summary(greedy[c]) for c in cities},
        matched=matched,
        llm_usage=client.usage.to_dict(),
    )
    json.dump(report, open(os.path.join(OUT, "report_data_5b.json"), "w", encoding="utf-8"),
              indent=2, default=float)

    # ---- tabela resumo A vs B vs greedy ----
    print("\n### RESUMO — desbalanceamento (max/mean, menor=melhor) e distância média por cidade")
    print("| cidade | métrica | A (sem) | B (com) | greedy |")
    print("|---|---|---|---|---|")
    for c in cities:
        a, b, g = summaries["A_baseline"][c], summaries["B_balanced"][c], _feasible_summary(greedy[c])
        print(f"| {c} | imbalance (med) | {a['mean_imbalance']:.3f} | {b['mean_imbalance']:.3f} | {g['mean_imbalance']:.3f} |")
        print(f"| {c} | Jain (med, maior=melhor) | {a['mean_jain']:.3f} | {b['mean_jain']:.3f} | {g['mean_jain']:.3f} |")
        print(f"| {c} | dist média f2 (med) | {a['mean_f2']:.3f} | {b['mean_f2']:.3f} | {g['mean_f2']:.3f} |")
    print(f"\nProveniência vencedoras: " +
          ", ".join(f"{cond}=seed{rep[cond]['seed']}/{rep[cond]['meta']['best_origin']}" for cond in CONDITIONS))
    print(f"Dados: {os.path.join(OUT, 'report_data_5b.json')}")
    print("DONE_5B")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
