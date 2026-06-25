"""
src/optimizers/llm/prompts.py — Prompts do otimizador-LLM (estilo ReEvo) para o MODO JUSTO.

Constrói (system, user) para cada operação: generate, reflect_short, reflect_long,
crossover, mutate. O system (estável → cacheado) descreve o FairODCProblem, a interface da
Instance exposta no sandbox e a assinatura `place_odcs`.
"""

from __future__ import annotations

SYSTEM = """Você é um especialista em otimização que ESCREVE HEURÍSTICAS EM PYTHON para o \
posicionamento de Open RAN Data Centers (ODCs) em redes 5G.

PROBLEMA (multiobjetivo, minimizar os dois):
- Há `n_sites` SITES candidatos (estações únicas da cidade). Uma solução ATIVA um subconjunto deles.
- f1 = nº de ODCs ativos; f2 = distância média de fronthaul (cada cliente vai ao ODC ativo MAIS PRÓXIMO).
- Viável só se: capacidade por ODC <= max_capacity (cores) E distância de cada cliente ao seu ODC <= max_distance (km).
- A fronteira é varrida chamando sua função para n_active = mínimo_viável .. n_sites; score = HYPERVOLUME (penalizado por inviabilidade). Mais HV = melhor.

ASSINATURA EXATA:
    def place_odcs(instance, n_active):
        # retorna a LISTA dos índices (inteiros em [0, instance.n_sites)) dos `n_active` sites a ativar.

INTERFACE de `instance` (somente leitura; JÁ pronta — NÃO recompute distâncias a partir de coordenadas):
    instance.n_sites       : int
    instance.n_clients      : int
    instance.distances      : np.ndarray de shape (n_clients, n_sites)  <-- USE ISTO. distances[c, s] = km do cliente c ao site s (Haversine).
    instance.client_demand  : np.ndarray de shape (n_clients,)          -> demanda de CPU (cores) por cliente
    instance.max_distance   : float (km)   instance.max_capacity : float (cores)
    instance.site_coords    : np.ndarray (n_sites, 2)   instance.client_coords : np.ndarray (n_clients, 2)   instance.demand_total : float

IDIOMAS CORRETOS (copie estes padrões VETORIZADOS — cuidado com os shapes!):
    D = instance.distances                  # (n_clients, n_sites)
    # dado um subconjunto `sel` (lista/array de índices de site):
    sub = D[:, sel]                          # (n_clients, len(sel))
    nearest_idx = sub.argmin(axis=1)         # (n_clients,) -> qual ODC (posição em sel) atende cada cliente
    nearest_dist = sub.min(axis=1)           # (n_clients,) -> distância de cada cliente ao ODC mais próximo
    mean_fronthaul = nearest_dist.mean()     # = f2
    # carga por ODC selecionado:
    load = np.zeros(len(sel)); np.add.at(load, nearest_idx, instance.client_demand)   # (len(sel),)
    feasible = (load.max() <= instance.max_capacity) and (nearest_dist.max() <= instance.max_distance)

REGRAS:
- Python puro + `np` (numpy) já disponível (pode escrever `import numpy as np`). PROIBIDO: outros imports, open, I/O, rede, eval/exec, dunder.
- Use SEMPRE `instance.distances` (não recalcule). Cuidado com shapes: distances é (n_clients, n_sites).
- VETORIZE (sem laços sobre clientes); rápido (< 1 s para n_sites~100). Determinístico (evite np.random; se usar, fixe seed).
- Retorne EXATAMENTE `n_active` índices de site DISTINTOS e válidos. Se sua construção gerar menos, COMPLETE com os demais sites (p.ex. os de menor distância média a algum cliente); se gerar mais, corte para os `n_active` primeiros.
- NUNCA chame argmin/min/argmax/argsort de um array possivelmente vazio; trate o caso de subconjunto vazio. Garanta `1 <= n_active <= instance.n_sites`.
- Responda APENAS com UM bloco ```python contendo a função `place_odcs` (nada fora do bloco)."""


def generate_user(idea_hint: str = "") -> str:
    base = (
        "Escreva uma heurística `place_odcs` NOVA e eficaz. Pense no trade-off nº de ODCs x distância "
        "média e na cobertura para viabilidade. Seja criativo — não copie um greedy trivial."
    )
    if idea_hint:
        base += f"\nDireção sugerida: {idea_hint}"
    return base


def reflect_short_user(better_code, better_score, worse_code, worse_score) -> str:
    return (
        "Compare DUAS heurísticas para este problema. A primeira teve MAIOR hypervolume.\n\n"
        f"[MELHOR | HV={better_score:.4f}]\n```python\n{better_code}\n```\n\n"
        f"[PIOR | HV={worse_score:.4f}]\n```python\n{worse_code}\n```\n\n"
        "Em 2-3 frases, explique de forma ACIONÁVEL por que a melhor supera a pior "
        "(que princípio de design importa). Responda só com o texto da reflexão."
    )


def reflect_long_user(short_reflections: list[str]) -> str:
    joined = "\n- ".join(short_reflections[-10:])
    return (
        "Estas são reflexões de curto prazo acumuladas ao evoluir heurísticas para este problema:\n- "
        f"{joined}\n\nSintetize, em 3-5 itens, os PRINCÍPIOS DE DESIGN mais importantes para escrever "
        "uma heurística de alto hypervolume. Responda só com os itens."
    )


def crossover_user(parent_a, parent_b, long_reflection: str) -> str:
    return (
        "Combine as forças das DUAS heurísticas-pai numa NOVA `place_odcs` melhor, guiado pelos princípios.\n\n"
        f"PRINCÍPIOS:\n{long_reflection}\n\n"
        f"[PAI A]\n```python\n{parent_a}\n```\n\n[PAI B]\n```python\n{parent_b}\n```\n\n"
        "Responda APENAS com o bloco ```python da nova função `place_odcs`."
    )


def mutate_user(parent_code, long_reflection: str) -> str:
    return (
        "MUTE a heurística abaixo para aumentar o hypervolume, guiado pelos princípios. Faça uma "
        "mudança significativa (não cosmética), mantendo-a válida e rápida.\n\n"
        f"PRINCÍPIOS:\n{long_reflection}\n\n[HEURÍSTICA]\n```python\n{parent_code}\n```\n\n"
        "Responda APENAS com o bloco ```python da função `place_odcs` mutada."
    )
