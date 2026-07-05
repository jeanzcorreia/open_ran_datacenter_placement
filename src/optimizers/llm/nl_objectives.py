"""
src/optimizers/llm/nl_objectives.py — Objetivos adicionais descritos em LINGUAGEM NATURAL (Fase 5b).

DIFERENCIAL DO PROJETO: adicionar um objetivo (aqui, BALANCEAMENTO DE CARGA) descrevendo-o em
PORTUGUÊS no prompt de geração, fazendo o LLM REGENERAR a heurística para considerá-lo — SEM tocar
na função de fitness (`FairODCProblem.evaluate` continua [nº ODCs, distância média]). O efeito é
medido FORA do fitness por um índice de desbalanceamento de carga.

Contraste (o ponto da tese): para os baselines (NSGA-II/NSGA-III/MOEA/D/greedy) adicionar o MESMO
objetivo exige EDITAR código à mão — em `src/problem/odc_problem.py` (`FairODCProblem._evaluate_one`:
computar um 3º objetivo, `n_obj` 2→3, ajustar `to_pymoo`) e/ou no critério guloso em
`src/optimizers/fair.py` (`GreedyFair.solve`). O LLM faz só pela descrição.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------- objetivo NL (PT)
LOAD_BALANCING_OBJECTIVE_PT = (
    "\n\nOBJETIVO ADICIONAL — BALANCEAMENTO DE CARGA:\n"
    "Além de minimizar o número de ODCs e a distância média de fronthaul, DISTRIBUA A CARGA de forma "
    "EQUILIBRADA entre os ODCs ativos: evite que algum ODC fique muito mais carregado que os outros. "
    "A carga de um ODC ativo é a soma de `instance.client_demand` dos clientes atendidos por ele (o "
    "ODC ativo MAIS PRÓXIMO). Calcule-a VETORIZADA, por exemplo:\n"
    "    sub = instance.distances[:, sel]      # (n_clients, len(sel))\n"
    "    nearest = sub.argmin(axis=1)          # ODC mais próximo de cada cliente\n"
    "    load = np.zeros(len(sel)); np.add.at(load, nearest, instance.client_demand)\n"
    "Sua heurística deve PREFERIR seleções em que a carga MÁXIMA por ODC fique próxima da carga MÉDIA "
    "(= demanda_total / nº de ODCs ativos) — minimize `load.max()/load.mean()` (ou o desvio-padrão das "
    "cargas). Por exemplo: escolha sites que dividam melhor a demanda, ou reserve ODCs para regiões de "
    "alta demanda. Mantenha a VIABILIDADE (carga por ODC <= max_capacity, distância <= max_distance) e "
    "continue rápida e VETORIZADA: NÃO use laços sobre clientes NEM sobre sites (nada de "
    "`for s in range(n_sites)`/`for c in range(n_clients)`) — use operações de array do numpy."
)


# Few-shot VÁLIDO que JÁ considera balanceamento (mostra o formato correto — computa carga com
# client_demand e alivia o ODC mais carregado a cada passo). Validado: imbalance menor que uma
# heurística só-distância, ao custo de maior distância média (o trade-off esperado).
_LOAD_FEWSHOT = """```python
import numpy as np
def place_odcs(instance, n_active):
    D = instance.distances                       # (n_clients, n_sites)
    n_sites = instance.n_sites
    n = int(max(1, min(n_active, n_sites)))
    selected = [int(np.argmin(D.min(axis=0)))]   # 1o ODC: melhor cobertura
    while len(selected) < n:                     # laço SÓ sobre n_active (nunca sobre clientes)
        sub = D[:, selected]
        assign = sub.argmin(axis=1)              # ODC ativo mais próximo de cada cliente
        load = np.zeros(len(selected))
        np.add.at(load, assign, instance.client_demand)   # CARGA por ODC ativo
        busy = int(np.argmax(load))              # ODC mais carregado
        clients_busy = np.where(assign == busy)[0]        # clientes do ODC saturado
        d_busy = D[clients_busy, :].mean(axis=0)          # proximidade aos clientes saturados
        d_busy[selected] = np.inf
        selected.append(int(np.argmin(d_busy)))  # adiciona o site que DESCARREGA o ODC saturado
    return selected
```"""


def balanced_system(base_system: str) -> str:
    """Anexa a descrição do objetivo de balanceamento + um few-shot VÁLIDO ao system prompt base
    (condição B). Não muda o fitness — só o prompt visto pelo LLM (geração/crossover/mutação/
    reflexão). O few-shot dá ao 7B um molde correto de balanceamento (computar carga com
    client_demand e aliviar o ODC saturado) — escreva uma DIFERENTE seguindo a direção."""
    return (base_system.rstrip() + LOAD_BALANCING_OBJECTIVE_PT +
            "\n\nEXEMPLO de heurística VÁLIDA que JÁ considera o balanceamento (mostra o formato — "
            "escreva uma DIFERENTE, mas COMPUTE a carga com `instance.client_demand` e priorize "
            "aliviar o ODC mais carregado, como abaixo; NÃO copie):\n" + _LOAD_FEWSHOT)


# --------------------------------------------------------------------- métrica (fora do fitness)
def assign_loads(selection, hinst):
    """Carga por ODC ativo, IGUAL ao `assign_clients` do fitness (cliente -> ODC ativo mais próximo;
    carga = soma de client_demand). Retorna (loads, sel, nearest, client_dist).
    `loads[i]` é a carga do site `sel[i]` (inclui ODCs com carga 0)."""
    sel = np.asarray([int(s) for s in selection], dtype=int)
    sel = np.unique(sel[(sel >= 0) & (sel < hinst.n_sites)])
    if sel.size == 0:
        return np.zeros(0), sel, None, None
    sub = hinst.distances[:, sel]                          # (n_clients, |sel|)
    nearest = sub.argmin(axis=1)
    client_dist = sub[np.arange(sub.shape[0]), nearest]
    loads = np.zeros(sel.size)
    np.add.at(loads, nearest, hinst.client_demand)
    return loads, sel, nearest, client_dist


def load_imbalance(loads) -> dict:
    """Índices de desbalanceamento de carga entre ODCs ativos:
      - max_over_mean : carga máxima / carga média  (1.0 = perfeito; maior = pior)  [HEADLINE]
      - cv            : desvio-padrão / média        (0 = perfeito; maior = pior)
      - jain          : índice de Jain               (1.0 = perfeito; -> 1/n = pior)
    Calculado sobre TODOS os ODCs ativos (inclui carga 0 — penaliza ODC ocioso)."""
    loads = np.asarray(loads, dtype=float)
    n = int(loads.size)
    if n == 0:
        return dict(n_odc=0, max_over_mean=float("nan"), cv=float("nan"), jain=float("nan"),
                    max_load=0.0, mean_load=0.0, std_load=0.0)
    total = float(loads.sum())
    mean = total / n
    mx = float(loads.max())
    sq = float((loads ** 2).sum())
    return dict(
        n_odc=n,
        max_over_mean=(mx / mean if mean > 0 else float("nan")),
        cv=(float(loads.std()) / mean if mean > 0 else float("nan")),
        jain=(total ** 2 / (n * sq) if sq > 0 else 1.0),
        max_load=mx, mean_load=mean, std_load=float(loads.std()),
    )
