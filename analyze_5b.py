"""analyze_5b.py — re-seleciona representantes da Fase 5b por DESBALANCEAMENTO medido (não por HV)
e compara as DISTRIBUIÇÕES das populações A vs B. Usa a população salva em report_data_5b.json
(nenhuma re-execução de ReEvo). Sandbox de PROCESSO -> arquivo com guarda __main__."""
from __future__ import annotations
import json, multiprocessing, os, sys


def main():
    REPO = os.path.dirname(os.path.abspath(__file__))
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import numpy as np
    from src.problem.instance import load_instance_sites
    from src.problem.odc_problem import FairODCProblem
    from src.optimizers.llm.heuristic_runtime import HeuristicInstance
    from src.eval.runner_phase5b import evaluate_front_balance, _feasible_summary, _csv

    R = json.load(open(os.path.join(REPO, "results", "phase5b", "report_data_5b.json"), encoding="utf-8"))
    cities = R["cities"]
    insts, hinsts, fairs = {}, {}, {}
    for c in cities:
        inst = load_instance_sites(_csv(c))
        insts[c] = inst
        hinsts[c] = HeuristicInstance.from_instance(inst, 11.0, 1000.0)
        fairs[c] = FairODCProblem(inst, 11.0, 1000.0)

    def imb(code, city):
        recs = evaluate_front_balance(code, insts[city], hinsts[city], fairs[city], 1000.0, 80, 2.5)
        return _feasible_summary(recs)

    out = {}
    for cond in ["A_baseline", "B_balanced"]:
        seen, uniq = set(), []
        for h in R["population"][cond]:
            if h["code"] in seen:
                continue
            seen.add(h["code"]); uniq.append(h)
        rows = []
        for h in uniq:
            sN = imb(h["code"], "Natal")
            rows.append(dict(code=h["code"], agg=h["agg"], origin=h["origin"],
                             imbN=sN["mean_imbalance"], nfeasN=sN["n_feasible"], f2N=sN["mean_f2"]))
        valid = [r for r in rows if r["nfeasN"] > 0 and np.isfinite(r["imbN"])]
        print(f"\n=== {cond}: {len(uniq)} únicas | {len(valid)} viáveis em Natal ===")
        for r in sorted(rows, key=lambda x: (x["imbN"] if np.isfinite(x["imbN"]) else 9e9)):
            print(f"  imbN={r['imbN'] if np.isfinite(r['imbN']) else float('nan'):.3f} f2N={r['f2N']:.3f} "
                  f"nfeas={r['nfeasN']:3d} agg={r['agg']:.3f} origin={r['origin']}")
        if valid:
            best_bal = min(valid, key=lambda r: r["imbN"])
            hv_win = max(rows, key=lambda r: r["agg"])
            imbs = [r["imbN"] for r in valid]
            out[cond] = dict(best_bal=best_bal, hv_win=hv_win,
                             mean_imb=float(np.mean(imbs)), median_imb=float(np.median(imbs)), n_valid=len(valid))
            print(f"  >>> melhor BALANCEADOR: imbN={best_bal['imbN']:.3f} f2N={best_bal['f2N']:.3f} agg={best_bal['agg']:.3f}")
            print(f"  >>> vencedora HV       : imbN={hv_win['imbN'] if np.isfinite(hv_win['imbN']) else float('nan'):.3f} agg={hv_win['agg']:.3f}")
            print(f"  >>> imbalance população: média={out[cond]['mean_imb']:.3f} mediana={out[cond]['median_imb']:.3f}")

    if "A_baseline" in out and "B_balanced" in out:
        A, B = out["A_baseline"], out["B_balanced"]
        print("\n================ COMPARAÇÃO (Natal) ================")
        print(f"melhor balanceador:  A={A['best_bal']['imbN']:.3f}  B={B['best_bal']['imbN']:.3f}  "
              f"-> B {'MENOR (melhor)' if B['best_bal']['imbN'] < A['best_bal']['imbN'] else 'NÃO menor'}")
        print(f"imbalance médio pop: A={A['mean_imb']:.3f}  B={B['mean_imb']:.3f}  "
              f"-> B {'MENOR (melhor)' if B['mean_imb'] < A['mean_imb'] else 'NÃO menor'}")
        # salva o melhor balanceador de B e de A para o relatório
        for cond, o in out.items():
            with open(os.path.join(REPO, "results", "phase5b", f"best_balancer_{cond}.py"), "w", encoding="utf-8") as fh:
                fh.write(f"# melhor BALANCEADOR {cond}: imbN={o['best_bal']['imbN']:.3f} f2N={o['best_bal']['f2N']:.3f} "
                         f"agg={o['best_bal']['agg']:.3f} origin={o['best_bal']['origin']}\n{o['best_bal']['code']}\n")
        json.dump({k: {kk: (vv if kk not in ('best_bal', 'hv_win') else {x: vv[x] for x in ('imbN','f2N','agg','origin')})
                       for kk, vv in v.items()} for k, v in out.items()},
                  open(os.path.join(REPO, "results", "phase5b", "rebalance_analysis.json"), "w", encoding="utf-8"),
                  indent=2, default=float)
        print("\nsalvos: best_balancer_{A,B}.py, rebalance_analysis.json")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
