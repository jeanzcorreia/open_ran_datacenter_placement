"""
src/eval/runner_phase5.py — Fase 5a: ReEvo MULTI-CIDADE + benchmark nas 10 cidades +
avaliação de GENERALIZAÇÃO (treino 6 vs teste held-out 4).

Pipeline:
  1. TREINO (só nas 6 cidades de treino): MultiCityReEvoOptimizer evolui heurísticas-código;
     fitness = HV médio sobre as 6 cidades (instance_hv normalizado por cidade) + penalidade de
     robustez (crash/inviável em qualquer cidade => não pode vencer). Reporta mean/maximin.
     Ablação: MultiCityZeroShot (gera N e seleciona pelo mesmo critério, SEM evolução).
  2. BENCHMARK das 10 cidades (modo justo, sem cap): baselines POR CIDADE (nsga2/nsga3/moead/
     greedy/random, paralelos); a heurística VENCEDORA (treinada uma vez) é aplicada às 10 SEM
     re-evoluir; zero-shot idem. Fronteira-referência POR CIDADE = união dos viáveis não-dominados
     de todos os métodos naquela cidade. Métricas: HV, IGD+, spacing, spread. Fronteiras persistidas.
  3. GENERALIZAÇÃO: tabelas separando TREINO (6) vs TESTE (4); mean/maximin de HV do ReEvo por
     grupo + gap de generalização. VERIFICAÇÃO ANTI-BUG: a vencedora roda viável e SEM crash nas 10.

🔒 As 4 cidades de teste (Curitiba, Recife, Florianópolis, Vitória) NUNCA entram no treino/seleção
(o `solve_multi` recebe só as 6 de treino e ainda assim valida contra vazamento).

Uso:
    python -m src.eval.runner_phase5 --config configs/phase5a.yaml
    python -m src.eval.runner_phase5 --smoke --backend offline --out results/phase5a_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from ..optimizers.fair import FAIR_OPTIMIZERS
from ..optimizers.llm.llm_client import BudgetExceeded, build_llm_client
from ..optimizers.llm.reevo_multicity import (
    STRONG_SEED_CODE,
    MultiCityReEvoOptimizer,
    MultiCityZeroShot,
    apply_heuristic_to_city,
)
from ..problem.instance import load_instance_sites
from .metrics import hypervolume, igd_plus, normalize, spacing, spread
from .reference_front import build_reference_front
from .runner import _ci95, _fmt_ci

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASELINES = ["nsga2", "nsga3", "moead", "greedy", "random"]
METHOD_ORDER = ["reevo", "zero_shot", "greedy", "nsga2", "nsga3", "moead", "random"]


def _csv(city_key):
    return os.path.join(REPO, "data", "processed", f"{city_key}.csv")


# --------------------------------------------------------------------- baseline worker (paralelo)
def _baseline_worker(task):
    """Roda UM baseline (cidade × método × seed) num processo separado. Salva a fronteira e
    devolve um registro leve (F/feasible) para a métrica. Recarrega a instância do CSV (barato)."""
    (csv_path, city_key, method, seed, md, mc, cpu, pop, ngen, out_dir) = task
    inst = load_instance_sites(csv_path, cpu_per_100mhz=cpu)
    cls = FAIR_OPTIMIZERS[method]
    opt = cls(md, mc) if method == "greedy" else cls(md, mc, pop_size=pop)
    ps = opt.solve(inst, budget=ngen, seed=seed)
    ps.save(os.path.join(out_dir, city_key, method, f"seed{seed}"))
    feas = np.asarray(ps.feasible, dtype=bool)
    return dict(city=city_key, method=method, seed=int(seed),
                F=np.atleast_2d(ps.F).tolist(), feasible=feas.tolist(),
                n=int(feas.sum()), elapsed=ps.meta.get("elapsed_sec"))


# --------------------------------------------------------------------- views/métricas
class _FrontView:
    """Visão leve (F, feasible) para a fronteira-referência e métricas (sem precisar de X)."""

    def __init__(self, F, feasible, method):
        self.F = np.atleast_2d(np.asarray(F, dtype=float))
        self.feasible = np.asarray(feasible, dtype=bool).ravel()
        self.method = method


def _metrics_one(F, feasible, ref, refF_norm):
    feas = np.asarray(feasible, dtype=bool).ravel()
    F = np.atleast_2d(np.asarray(F, dtype=float))
    if not feas.any():
        return dict(HV=0.0, IGDp=float("nan"), spacing=float("nan"), spread=float("nan"), n=0)
    Fn = normalize(F[feas], ref.ideal, ref.nadir)
    return dict(HV=hypervolume(Fn), IGDp=igd_plus(Fn, refF_norm),
                spacing=spacing(Fn), spread=spread(Fn, refF_norm), n=int(feas.sum()))


def _agg_metrics(runs, ref, refF_norm):
    """runs: lista de (F, feasible). Retorna {metric: (mean, ci95)} agregando sobre as runs."""
    ms = [_metrics_one(F, feas, ref, refF_norm) for (F, feas) in runs]
    return {k: _ci95([d[k] for d in ms]) for k in ("HV", "IGDp", "spacing", "spread", "n")}


# --------------------------------------------------------------------- main
def _load_cfg(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "phase5a.yaml"))
    ap.add_argument("--backend", default=None, help="override llm.backend (routed|offline|...)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase5a"))
    ap.add_argument("--smoke", action="store_true",
                    help="config reduzida p/ validar o pipeline (cidades pequenas, baselines minúsculos)")
    ap.add_argument("--no-baselines", action="store_true", help="pula baselines (debug)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    if args.backend:
        cfg.setdefault("llm", {})["backend"] = args.backend
    OUT_DIR = args.out
    os.makedirs(OUT_DIR, exist_ok=True)

    P = cfg["problem"]
    md, mc, cpu = P["max_distance"], P["max_capacity"], P["cpu_per_100mhz"]
    EV = cfg["eval"]
    disp = cfg.get("display_names", {})
    train_keys = list(cfg["split"]["train"])
    test_keys = list(cfg["split"]["test"])

    pop, ngen = EV["pop_size"], EV["n_gen"]
    base_seeds = list(EV["baseline_seeds"])
    sweep_max = EV.get("sweep_max_points", 200)
    loop_to = EV.get("loop_per_call_timeout", 1.5)
    apply_to = EV.get("apply_per_call_timeout", 2.5)
    loop_sweep = EV.get("loop_sweep_points", 14)
    rc = cfg["reevo"]
    pop_size, generations, elite = rc["pop_size"], rc["generations"], rc["elite"]
    reevo_seeds = list(rc.get("seeds", [rc.get("seed", 1)]))   # 4 seeds -> média±IC
    max_workers = EV.get("max_workers", max(1, os.cpu_count() - 1))
    strong_seeds = [STRONG_SEED_CODE]    # seed forte (construção limpa por distância mínima)

    if args.smoke:
        train_keys = ["Natal", "CampoGrande"]
        test_keys = ["Vitoria"]
        pop, ngen, base_seeds = 40, 8, [1]
        pop_size, generations, elite = 4, 2, 2
        reevo_seeds = [1, 2]
        sweep_max, loop_to, apply_to, loop_sweep = 60, 0.6, 1.0, 8
        print("[SMOKE] cidades reduzidas + baselines minúsculos + 2 seeds.")

    # Ablação zero-shot = MELHOR da MESMA população inicial do ReEvo (seeds fortes + mesmos hints),
    # sem evolução -> isola o ganho do laço evolutivo (n_samples = pop_size, fixado no construtor).

    all_keys = train_keys + test_keys

    def dname(k):
        return disp.get(k, k)

    print("=" * 78)
    print("FASE 5a — ReEvo MULTI-CIDADE + benchmark 10 cidades + generalização")
    print(f"TREINO ({len(train_keys)}): {[dname(k) for k in train_keys]}")
    print(f"TESTE held-out ({len(test_keys)}): {[dname(k) for k in test_keys]}")
    print("=" * 78)

    # ============================================================ 1. TREINO (só 6 cidades)
    reevo_client = build_llm_client(cfg, log_path=os.path.join(OUT_DIR, "reevo_calls.jsonl"))
    zs_client = build_llm_client(cfg, log_path=os.path.join(OUT_DIR, "zeroshot_calls.jsonl"))
    print(f"\nLLM backend: {reevo_client.name} | model={reevo_client.model} "
          f"reflection={getattr(reevo_client, 'reflection_model', '-')}")

    print(f"\nCarregando {len(train_keys)} instâncias de TREINO...")
    train_insts = []
    for k in train_keys:
        inst = load_instance_sites(_csv(k), cpu_per_100mhz=cpu)
        print(f"  {dname(k):16s}: {inst.n_unique_sites} sites, {inst.n_clients} clientes")
        train_insts.append(inst)

    print(f"\n--- ReEvo multi-cidade (treino, {len(reevo_seeds)} seeds) + zero-shot por seed ---")
    reevo = MultiCityReEvoOptimizer(reevo_client, md, mc, pop_size=pop_size,
                                    generations=generations, elite=elite,
                                    per_call_timeout=loop_to, sweep_max_points=loop_sweep,
                                    strong_seeds=strong_seeds)
    zs = MultiCityZeroShot(zs_client, md, mc, n_samples=pop_size,
                           per_call_timeout=loop_to, sweep_max_points=loop_sweep,
                           strong_seeds=strong_seeds)
    t_train0 = time.time()
    reevo_runs = {}    # seed -> {"code", "meta"}
    zs_runs = {}
    for s in reevo_seeds:
        print(f"\n  === ReEvo seed {s} ===")
        try:
            rr = reevo.solve_multi(train_insts, seed=s)
        except BudgetExceeded as e:
            print(f"  budget LLM excedido (seed {s}): {e}"); raise
        reevo_runs[s] = dict(code=rr.code, meta=rr.meta)
        m = rr.meta
        print(f"  reevo seed {s}: meanHV={m['train_mean_hv']:.4f} maximin={m['train_maximin_hv']:.4f} "
              f"robusta={m['robust']} origem={m['best_origin']} offspring={m.get('total_offspring_evaluated','?')} "
              f"| chamadas={reevo_client.usage.calls}")
        zr = zs.solve_multi(train_insts, seed=s)
        zs_runs[s] = dict(code=zr.code, meta=zr.meta)
        print(f"  zero-shot seed {s}: meanHV={zr.meta['train_mean_hv']:.4f} origem={zr.meta['best_origin']}")

    # vencedora "representativa" = a de MAIOR train_mean_hv entre os seeds (para salvar/relatar o código)
    best_seed = max(reevo_seeds, key=lambda s: reevo_runs[s]["meta"]["train_mean_hv"])
    winner_code = reevo_runs[best_seed]["code"]
    train_meta = reevo_runs[best_seed]["meta"]
    origins = [reevo_runs[s]["meta"]["best_origin"] for s in reevo_seeds]
    print(f"\n  PROVENIÊNCIA por seed: {dict(zip(reevo_seeds, origins))}")
    print(f"  vencedora representativa: seed {best_seed} (origem={train_meta['best_origin']}, "
          f"meanHV={train_meta['train_mean_hv']:.4f})")

    # salva a vencedora representativa + as 4 vencedoras por seed
    with open(os.path.join(OUT_DIR, "winning_heuristic_multicity.py"), "w") as fh:
        fh.write(f"# Heurística vencedora REPRESENTATIVA (ReEvo MULTI-CIDADE, Fase 5a corrigida)\n"
                 f"# seed={best_seed} origem={train_meta['best_origin']} robusta={train_meta['robust']} "
                 f"train_meanHV={train_meta['train_mean_hv']:.4f} "
                 f"train_maximin={train_meta['train_maximin_hv']:.4f}\n"
                 f"# proveniência por seed: {dict(zip(reevo_seeds, origins))}\n")
        fh.write(winner_code + "\n")
    for s in reevo_seeds:
        with open(os.path.join(OUT_DIR, f"winner_seed{s}.py"), "w") as fh:
            mm = reevo_runs[s]["meta"]
            fh.write(f"# seed={s} origem={mm['best_origin']} meanHV={mm['train_mean_hv']:.4f} "
                     f"maximin={mm['train_maximin_hv']:.4f} robusta={mm['robust']}\n")
            fh.write(reevo_runs[s]["code"] + "\n")

    # ============================================================ 2. BENCHMARK das 10 cidades
    print(f"\nCarregando {len(test_keys)} instâncias de TESTE held-out...")
    instances = {k: inst for k, inst in zip(train_keys, train_insts)}
    for k in test_keys:
        inst = load_instance_sites(_csv(k), cpu_per_100mhz=cpu)
        print(f"  {dname(k):16s}: {inst.n_unique_sites} sites, {inst.n_clients} clientes")
        instances[k] = inst

    # --- aplica CADA vencedora-por-seed + zero-shot a TODAS as 10 (sem re-evoluir) ---
    print(f"\n--- Aplicando as {len(reevo_seeds)} vencedoras (e zero-shot) às 10 cidades ---")
    reevo_fronts = {k: [] for k in all_keys}   # k -> [(F,feas) por seed]
    zs_fronts = {k: [] for k in all_keys}
    winner_apply = {k: [] for k in all_keys}   # k -> [per-seed dict]
    for k in all_keys:
        inst = instances[k]
        grp = "train" if k in train_keys else "test"
        for s in reevo_seeds:
            ps_r, ncr = apply_heuristic_to_city(reevo_runs[s]["code"], inst, md, mc, "reevo", inst.name,
                                                sweep_max_points=sweep_max, per_call_timeout=apply_to,
                                                meta={"group": grp, "seed": s})
            ps_r.save(os.path.join(OUT_DIR, k, "reevo", f"seed{s}"))
            reevo_fronts[k].append((ps_r.F, ps_r.feasible))
            nfeas = int(np.asarray(ps_r.feasible).sum())
            winner_apply[k].append(dict(seed=s, n_crash=int(ncr), n_feasible=nfeas,
                                        origin=reevo_runs[s]["meta"]["best_origin"]))
            ps_z, _ = apply_heuristic_to_city(zs_runs[s]["code"], inst, md, mc, "zero_shot", inst.name,
                                              sweep_max_points=sweep_max, per_call_timeout=apply_to)
            ps_z.save(os.path.join(OUT_DIR, k, "zero_shot", f"seed{s}"))
            zs_fronts[k].append((ps_z.F, ps_z.feasible))
        crashes = sum(w["n_crash"] for w in winner_apply[k])
        feas_lo = min(w["n_feasible"] for w in winner_apply[k])
        flag = "OK" if (crashes == 0 and feas_lo >= 1) else "*** FALHA ***"
        print(f"  {dname(k):16s}: reevo Σn_crash={crashes} min|viável|={feas_lo} [{flag}]")

    # --- baselines por cidade (paralelo) ---
    baseline_runs = {k: {m: [] for m in BASELINES} for k in all_keys}
    if not args.no_baselines:
        tasks = []
        for k in all_keys:
            for m in BASELINES:
                seeds = [base_seeds[0]] if m == "greedy" else base_seeds
                for s in seeds:
                    tasks.append((_csv(k), k, m, s, md, mc, cpu, pop, ngen, OUT_DIR))
        print(f"\n--- Baselines: {len(tasks)} tarefas (cidade×método×seed) em até {max_workers} processos ---")
        t_base0 = time.time()
        parallel = EV.get("parallel", True) and max_workers > 1
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_baseline_worker, t): t for t in tasks}
                done = 0
                for fut in as_completed(futs):
                    r = fut.result()
                    baseline_runs[r["city"]][r["method"]].append((r["F"], r["feasible"]))
                    done += 1
                    if done % 20 == 0 or done == len(tasks):
                        print(f"    {done}/{len(tasks)} baselines concluídos "
                              f"({time.time()-t_base0:.0f}s)")
        else:
            for i, t in enumerate(tasks):
                r = _baseline_worker(t)
                baseline_runs[r["city"]][r["method"]].append((r["F"], r["feasible"]))
        print(f"  baselines: {time.time()-t_base0:.0f}s")

    # ============================================================ 3. MÉTRICAS POR CIDADE
    print("\n--- Métricas por cidade (fronteira-referência por cidade) ---")
    city_metrics = {}          # city -> method -> {metric: (mean, ci)}
    ref_info = {}
    skipped_cities = []
    for k in all_keys:
        method_runs = {}
        for m in BASELINES:
            if baseline_runs[k][m]:
                method_runs[m] = baseline_runs[k][m]
        method_runs["reevo"] = reevo_fronts[k]        # lista de (F,feas) sobre os seeds -> média±IC
        method_runs["zero_shot"] = zs_fronts[k]

        views = [_FrontView(F, feas, m) for m, runs in method_runs.items() for (F, feas) in runs]
        try:
            ref = build_reference_front(views)   # exige >=1 fronteira viável (qualquer método)
        except RuntimeError as e:
            # Defensivo: nenhum método produziu solução viável nesta cidade (não esperado —
            # greedy sempre produz). Registra e pula sem derrubar a rodada inteira.
            print(f"  [AVISO] {dname(k)}: fronteira-referência vazia ({e}); cidade PULADA nas métricas.")
            skipped_cities.append(k)
            continue
        refF_norm = normalize(ref.F, ref.ideal, ref.nadir)
        ref_info[k] = dict(n_points=int(len(ref.F)), ideal=ref.ideal.tolist(),
                           nadir=ref.nadir.tolist())
        city_metrics[k] = {m: _agg_metrics(runs, ref, refF_norm) for m, runs in method_runs.items()}

    # ---------------- impressão: tabelas por grupo ----------------
    def print_group(title, keys):
        print(f"\n### {title}\n")
        for k in keys:
            if k not in city_metrics:
                print(f"\n**{dname(k)}** — sem métricas (cidade pulada).")
                continue
            cm = city_metrics[k]
            print(f"\n**{dname(k)}** ({instances[k].n_unique_sites} sites; "
                  f"ref={ref_info[k]['n_points']} pts)\n")
            print("| Método | HV ↑ | IGD+ ↓ | Spacing ↓ | Spread ↓ | |front| |")
            print("|---|---|---|---|---|---|")
            for m in METHOD_ORDER:
                if m not in cm:
                    continue
                a = cm[m]
                print(f"| {m} | {_fmt_ci(a['HV'])} | {_fmt_ci(a['IGDp'])} | {_fmt_ci(a['spacing'])} "
                      f"| {_fmt_ci(a['spread'])} | {_fmt_ci(a['n'],1)} |")

    print_group(f"TREINO ({len(train_keys)} cidades)", train_keys)
    print_group(f"TESTE held-out ({len(test_keys)} cidades)", test_keys)

    # ---------------- generalização (resumo) ----------------
    def _vals(method, keys, metric):
        out = []
        for k in keys:
            if k in city_metrics and method in city_metrics[k]:
                v = city_metrics[k][method][metric][0]
                if not (isinstance(v, float) and np.isnan(v)):
                    out.append(float(v))
        return out

    def _mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    def group_stat(method, keys, metric="HV"):
        vals = _vals(method, keys, metric)
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.min(vals))   # (média, pior=maximin p/ HV)

    def _has(m):
        return any(k in city_metrics and m in city_metrics[k] for k in all_keys)

    print("\n\n### GENERALIZAÇÃO — média de HV por grupo (treino 6 vs teste held-out 4)\n")
    print("| Método | HV treino (média) | HV treino (maximin) | HV teste (média) | HV teste (maximin) | gap (treino−teste) |")
    print("|---|---|---|---|---|---|")
    gen_summary = {}
    for m in METHOD_ORDER:
        if not _has(m):
            continue
        tr_mean, tr_min = group_stat(m, train_keys)
        te_mean, te_min = group_stat(m, test_keys)
        gap = tr_mean - te_mean if not (np.isnan(tr_mean) or np.isnan(te_mean)) else float("nan")
        gen_summary[m] = dict(train_mean=tr_mean, train_maximin=tr_min,
                              test_mean=te_mean, test_maximin=te_min, gap=gap)
        print(f"| {m} | {tr_mean:.4f} | {tr_min:.4f} | {te_mean:.4f} | {te_min:.4f} | {gap:+.4f} |")

    print("\n### GENERALIZAÇÃO — média de IGD+ por grupo\n")
    print("| Método | IGD+ treino | IGD+ teste |")
    print("|---|---|---|")
    for m in METHOD_ORDER:
        if not _has(m):
            continue
        tr = _mean(_vals(m, train_keys, "IGDp"))
        te = _mean(_vals(m, test_keys, "IGDp"))
        print(f"| {m} | {tr:.4f} | {te:.4f} |")

    # ---------------- verificação anti-bug (todas as vencedoras × todas as cidades) -------------
    print(f"\n### VERIFICAÇÃO ANTI-BUG — as {len(reevo_seeds)} vencedoras rodam viáveis e SEM crash nas 10?\n")
    print("| Cidade | grupo | Σ n_crash (4 seeds) | min |viável| | OK? |")
    print("|---|---|---|---|---|")
    all_ok = True
    for k in all_keys:
        wl = winner_apply[k]
        crashes = sum(w["n_crash"] for w in wl)
        feas_lo = min(w["n_feasible"] for w in wl)
        ok = (crashes == 0 and feas_lo >= 1)
        all_ok = all_ok and ok
        grp = "treino" if k in train_keys else "teste"
        print(f"| {dname(k)} | {grp} | {crashes} | {feas_lo} | {'OK' if ok else 'FALHA'} |")
    print(f"\n**Todas as vencedoras viáveis e sem crash nas 10 cidades: "
          f"{'SIM ✔ (bug da Fase 4 corrigido)' if all_ok else 'NÃO ✗'}**")

    # provenance summary
    prov = {}
    for s in reevo_seeds:
        o = reevo_runs[s]["meta"]["best_origin"]; prov[o] = prov.get(o, 0) + 1
    print(f"\n**Proveniência das {len(reevo_seeds)} vencedoras:** {prov} "
          f"(seed_strong=injetada | seed=init LLM | crossover/mutate=EVOLUÇÃO agregou)")

    # ---------------- orçamento LLM ----------------
    ru = reevo_client.usage.to_dict()
    zu = zs_client.usage.to_dict()
    print("\n### Orçamento LLM (Fase 5a corrigida)\n")
    print(f"- Geração (ReEvo, {len(reevo_seeds)} seeds): {ru['calls']} chamadas de REDE | "
          f"in={ru['input_tokens']} out={ru['output_tokens']} | custo~US${ru['cost_usd']}")
    print(f"- Zero-shot: {zu['calls']} chamadas | out={zu['output_tokens']} | custo~US${zu['cost_usd']}")
    print(f"- offspring avaliados (por seed): "
          f"{[reevo_runs[s]['meta'].get('total_offspring_evaluated','?') for s in reevo_seeds]}")
    print(f"- INVARIANTE: nº de chamadas depende SÓ de pop/gen/seeds, NÃO do nº de cidades.")

    # ============================================================ persistência (report_data.json)
    report = dict(
        split=dict(train=train_keys, test=test_keys, display_names=disp),
        problem=dict(max_distance=md, max_capacity=mc, cpu_per_100mhz=cpu),
        config=dict(pop=pop, ngen=ngen, baseline_seeds=base_seeds, sweep_max_points=sweep_max,
                    reevo=dict(pop_size=pop_size, generations=generations, elite=elite, seeds=reevo_seeds),
                    routing="gemini-only(gen=2.5-flash, refl=2.5-flash-lite)", strong_seed=True),
        reevo_seeds=reevo_seeds,
        best_seed=best_seed,
        provenance={s: reevo_runs[s]["meta"]["best_origin"] for s in reevo_seeds},
        train_meta=train_meta,
        per_seed_train_meta={s: reevo_runs[s]["meta"] for s in reevo_seeds},
        zero_shot_meta={s: zs_runs[s]["meta"] for s in reevo_seeds},
        winner_code=winner_code,
        per_seed_winner_code={s: reevo_runs[s]["code"] for s in reevo_seeds},
        winner_apply=winner_apply,
        all_winner_feasible_no_crash=bool(all_ok),
        city_metrics={k: {m: {kk: list(vv) for kk, vv in a.items()} for m, a in cm.items()}
                      for k, cm in city_metrics.items()},
        ref_info=ref_info,
        generalization=gen_summary,
        llm_usage=dict(reevo=ru, zero_shot=zu),
        train_elapsed_sec=round(time.time() - t_train0, 1),
        instances={k: dict(n_sites=instances[k].n_unique_sites, n_clients=instances[k].n_clients)
                   for k in all_keys},
    )
    with open(os.path.join(OUT_DIR, "report_data.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nDados do relatório: {os.path.join(OUT_DIR, 'report_data.json')}")
    print(f"Heurística vencedora: {os.path.join(OUT_DIR, 'winning_heuristic_multicity.py')}")


if __name__ == "__main__":
    main()
