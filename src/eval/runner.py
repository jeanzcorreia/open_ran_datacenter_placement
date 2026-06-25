"""
src/eval/runner.py — Fase 3: roda todos os métodos × seeds em Natal (MODO JUSTO), constrói a
fronteira de referência (união dos não-dominados viáveis), computa HV / IGD+ / spacing /
spread (normalizados), e imprime a tabela comparativa (média ± IC95%). Persiste as
fronteiras em results/phase3/.

Uso:
    python -m src.eval.runner                       # todos os métodos, 5 seeds
    python -m src.eval.runner --seeds 1 2 3
    python -m src.eval.runner --methods nsga2 random
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..optimizers.fair import FAIR_OPTIMIZERS
from ..problem.instance import load_instance_sites
from .metrics import HV_REF_POINT, hypervolume, igd_plus, normalize, spacing, spread
from .reference_front import build_reference_front

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NATAL_CSV = os.path.join(REPO, "CityData", "Natal.csv")
OUT_DIR = os.path.join(REPO, "results", "phase3")

# Cenário do paper (modo justo):
CPU_PER_100MHZ = 14
MAX_DISTANCE = 11.0
MAX_CAPACITY = 1000.0
POP_SIZE = 300
N_GEN = 60
DEFAULT_METHODS = ["nsga2", "nsga3", "moead", "random", "greedy"]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]

# t de Student (0.975) por graus de liberdade, para IC95% com n pequeno.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 29: 2.045}


def _ci95(vals):
    v = np.asarray([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(v[0]), 0.0
    m = float(v.mean())
    sem = float(v.std(ddof=1) / np.sqrt(n))
    t = _T95.get(n - 1, 1.96)
    return m, t * sem


def run(methods, seeds, budget=N_GEN):
    inst = load_instance_sites(NATAL_CSV, cpu_per_100mhz=CPU_PER_100MHZ)
    print(f"Instância: {inst.summary()}")
    print(f"Cenário: max_km={MAX_DISTANCE}, capacity={MAX_CAPACITY}, pop={POP_SIZE}, n_gen={budget}\n")

    runs = []
    for mname in methods:
        cls = FAIR_OPTIMIZERS[mname]
        opt = cls(MAX_DISTANCE, MAX_CAPACITY) if mname == "greedy" else cls(MAX_DISTANCE, MAX_CAPACITY, pop_size=POP_SIZE)
        method_seeds = [seeds[0]] if mname == "greedy" else seeds  # greedy é determinístico
        for s in method_seeds:
            ps = opt.solve(inst, budget=budget, seed=s)
            ps.save(os.path.join(OUT_DIR, inst.name, mname, f"seed{s}"))
            feas = np.asarray(ps.feasible, dtype=bool)
            print(f"  {mname:7s} seed={s}: |front viável|={int(feas.sum()):3d}  "
                  f"n_eval={ps.meta.get('n_eval')}  t={ps.meta.get('elapsed_sec')}s  src={ps.meta.get('front_source')}")
            runs.append(dict(method=mname, seed=s, ps=ps))

    # ---------- fronteira de referência (união dos não-dominados viáveis) ----------
    ref = build_reference_front([r["ps"] for r in runs])
    refF_norm = normalize(ref.F, ref.ideal, ref.nadir)
    print(f"\nFronteira de REFERÊNCIA: {len(ref.F)} pts | ideal={ref.ideal.tolist()} nadir={ref.nadir.tolist()}"
          f" | n_odc {int(ref.F[:,0].min())}..{int(ref.F[:,0].max())} | HV ref_point(norm)={HV_REF_POINT.tolist()}")

    # ---------- métricas por run ----------
    for r in runs:
        ps = r["ps"]
        feas = np.asarray(ps.feasible, dtype=bool)
        if feas.any():
            Ff = np.atleast_2d(ps.F)[feas]
            Fn = normalize(Ff, ref.ideal, ref.nadir)
            r["HV"] = hypervolume(Fn)
            r["IGD+"] = igd_plus(Fn, refF_norm)
            r["spacing"] = spacing(Fn)
            r["spread"] = spread(Fn, refF_norm)
            r["n_front"] = int(feas.sum())
            r["odc_lo"], r["odc_hi"] = int(Ff[:, 0].min()), int(Ff[:, 0].max())
        else:
            r["HV"], r["IGD+"], r["spacing"], r["spread"] = 0.0, float("nan"), float("nan"), float("nan")
            r["n_front"], r["odc_lo"], r["odc_hi"] = 0, None, None
        r["n_eval"] = ps.meta.get("n_eval")
        r["time"] = ps.meta.get("elapsed_sec")

    # ---------- agregação por método ----------
    summary = {}
    for mname in methods:
        rs = [r for r in runs if r["method"] == mname]
        agg = {}
        for key in ["HV", "IGD+", "spacing", "spread", "n_front", "n_eval", "time"]:
            agg[key] = _ci95([r[key] for r in rs])
        agg["n_runs"] = len(rs)
        agg["odc_range"] = (min(r["odc_lo"] for r in rs if r["odc_lo"] is not None),
                            max(r["odc_hi"] for r in rs if r["odc_hi"] is not None)) if any(r["odc_lo"] is not None for r in rs) else None
        summary[mname] = agg

    _print_table(summary, methods)

    out_json = os.path.join(OUT_DIR, inst.name, "summary.json")
    with open(out_json, "w") as fh:
        json.dump(_jsonable_summary(summary, ref), fh, indent=2)
    print(f"\nResumo salvo em: {out_json}")
    print(f"Fronteiras salvas em: {os.path.join(OUT_DIR, inst.name)}")
    return runs, ref, summary


def _fmt_ci(mc, nd=4):
    m, c = mc
    if isinstance(m, float) and np.isnan(m):
        return "nan"
    return f"{m:.{nd}f} ± {c:.{nd}f}"


def _print_table(summary, methods):
    print("\n### Tabela comparativa (Natal, modo justo) — média ± IC95% sobre seeds\n")
    print("| Método | HV ↑ | IGD+ ↓ | Spacing ↓ | Spread Δ ↓ | |front| | n_eval | tempo (s) |")
    print("|---|---|---|---|---|---|---|---|")
    for m in methods:
        a = summary[m]
        print(f"| {m} | {_fmt_ci(a['HV'])} | {_fmt_ci(a['IGD+'])} | {_fmt_ci(a['spacing'])} "
              f"| {_fmt_ci(a['spread'])} | {_fmt_ci(a['n_front'],1)} | {_fmt_ci(a['n_eval'],0)} | {_fmt_ci(a['time'],2)} |")
    # vencedor por HV
    best = max(methods, key=lambda m: (summary[m]["HV"][0] if not np.isnan(summary[m]["HV"][0]) else -1))
    print(f"\nMaior HV: **{best}**")


def _jsonable_summary(summary, ref):
    out = {"reference_front": {"n_points": int(len(ref.F)), "ideal": ref.ideal.tolist(),
                               "nadir": ref.nadir.tolist(), "hv_ref_point": HV_REF_POINT.tolist()},
           "methods": {}}
    for m, a in summary.items():
        out["methods"][m] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in a.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=list(FAIR_OPTIMIZERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--budget", type=int, default=N_GEN)
    args = ap.parse_args()
    run(args.methods, args.seeds, budget=args.budget)


if __name__ == "__main__":
    main()
