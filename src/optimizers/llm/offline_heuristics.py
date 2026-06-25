"""
src/optimizers/llm/offline_heuristics.py — Biblioteca determinística usada pelo OfflineBackend
(SEM LLM). Cada template é uma `place_odcs(instance, n_active)` VÁLIDA que roda no sandbox.

NÃO é um LLM: é um conjunto curado de heurísticas clássicas de facility-location + combinação
estruturada, para (a) fallback sem chave e (b) piso "sem LLM" / smoke test. As variações
(generate/crossover/mutate) são determinísticas, parametrizadas pelo `context`.
"""

from __future__ import annotations

# --- biblioteca de heurísticas (strings de código) ---------------------------------

GREEDY_DISTANCE = '''
def place_odcs(instance, n_active):
    """Greedy k-median: a cada passo adiciona o site que mais reduz a distância média."""
    D = instance.distances
    n_sites = instance.n_sites
    selected = []
    cur = None
    k = min(int(n_active), n_sites)
    for _ in range(k):
        best, best_val = -1, None
        for s in range(n_sites):
            if s in selected:
                continue
            nd = D[:, s] if cur is None else np.minimum(cur, D[:, s])
            v = float(nd.mean())
            if best_val is None or v < best_val:
                best_val, best = v, s
        selected.append(best)
        cur = D[:, best] if cur is None else np.minimum(cur, D[:, best])
    return selected
'''

DEMAND_WEIGHTED = '''
def place_odcs(instance, n_active):
    """Escolhe os sites com maior demanda de CPU atribuída (clientes mais próximos)."""
    D = instance.distances
    dem = instance.client_demand
    n_sites = instance.n_sites
    nearest = D.argmin(axis=1)
    site_dem = np.zeros(n_sites)
    np.add.at(site_dem, nearest, dem)
    order = list(np.argsort(-site_dem))
    return order[:min(int(n_active), n_sites)]
'''

FARTHEST_FIRST = '''
def place_odcs(instance, n_active):
    """Farthest-first: site central, depois adiciona o mais distante dos já escolhidos."""
    D = instance.distances
    n_sites = instance.n_sites
    selected = [int(np.argmin(D.mean(axis=0)))]
    k = min(int(n_active), n_sites)
    while len(selected) < k:
        best, best_val = -1, -1.0
        for s in range(n_sites):
            if s in selected:
                continue
            d = min(float(np.abs(D[:, s] - D[:, t]).mean()) for t in selected)
            if d > best_val:
                best_val, best = d, s
        selected.append(best)
    return selected
'''

CAPACITY_BALANCED = '''
def place_odcs(instance, n_active):
    """Greedy que minimiza a carga MÁXIMA por ODC (equilíbrio de capacidade)."""
    D = instance.distances
    dem = instance.client_demand
    n_sites = instance.n_sites
    selected = []
    k = min(int(n_active), n_sites)
    for _ in range(k):
        best, best_val = -1, None
        for s in range(n_sites):
            if s in selected:
                continue
            cand = selected + [s]
            sub = D[:, cand]
            nearest = sub.argmin(axis=1)
            loads = np.zeros(len(cand))
            np.add.at(loads, nearest, dem)
            v = float(loads.max())
            if best_val is None or v < best_val:
                best_val, best = v, s
        selected.append(best)
    return selected
'''

# Híbrido (crossover): greedy de distância semeado pela ordem de demanda.
GREEDY_DEMAND_HYBRID = '''
def place_odcs(instance, n_active):
    """Híbrido: semeia com o site de maior demanda, depois greedy de distância."""
    D = instance.distances
    dem = instance.client_demand
    n_sites = instance.n_sites
    nearest0 = D.argmin(axis=1)
    site_dem = np.zeros(n_sites)
    np.add.at(site_dem, nearest0, dem)
    seed = int(np.argmax(site_dem))
    selected = [seed]
    cur = D[:, seed].copy()
    k = min(int(n_active), n_sites)
    while len(selected) < k:
        best, best_val = -1, None
        for s in range(n_sites):
            if s in selected:
                continue
            nd = np.minimum(cur, D[:, s])
            v = float(nd.mean())
            if best_val is None or v < best_val:
                best_val, best = v, s
        selected.append(best)
        cur = np.minimum(cur, D[:, best])
    return selected
'''

# Mutação: greedy de distância com desempate por menor carga local (guarda de capacidade).
GREEDY_DISTANCE_CAPGUARD = '''
def place_odcs(instance, n_active):
    """Greedy de distância; desempata preferindo o site que mantém a carga máxima menor."""
    D = instance.distances
    dem = instance.client_demand
    n_sites = instance.n_sites
    selected = []
    cur = None
    k = min(int(n_active), n_sites)
    for _ in range(k):
        best, best_key = -1, None
        for s in range(n_sites):
            if s in selected:
                continue
            nd = D[:, s] if cur is None else np.minimum(cur, D[:, s])
            cand = selected + [s]
            sub = D[:, cand]
            loads = np.zeros(len(cand))
            np.add.at(loads, sub.argmin(axis=1), dem)
            key = (round(float(nd.mean()), 6), float(loads.max()))
            if best_key is None or key < best_key:
                best_key, best = key, s
        selected.append(best)
        cur = D[:, best] if cur is None else np.minimum(cur, D[:, best])
    return selected
'''

_LIBRARY = [GREEDY_DISTANCE, DEMAND_WEIGHTED, FARTHEST_FIRST, CAPACITY_BALANCED]


def generate(context) -> str:
    idx = int(context.get("index", 0)) % len(_LIBRARY)
    return _LIBRARY[idx].strip() + "\n"


def crossover(context) -> str:
    # combinação determinística: o híbrido demanda+distância
    return GREEDY_DEMAND_HYBRID.strip() + "\n"


def mutate(context) -> str:
    # perturbação determinística: greedy de distância com guarda de capacidade
    return GREEDY_DISTANCE_CAPGUARD.strip() + "\n"


def reflect(op, context) -> str:
    if op == "reflect_short":
        return ("Heurísticas que adicionam ODCs minimizando a distância média (greedy k-median) "
                "dominam as que ordenam por demanda: a distância é o objetivo discriminante e o "
                "greedy aproxima bem o ótimo por k. Capacidade raramente é o gargalo (cargas << 1000).")
    return ("Insight de longo prazo: priorizar redução de distância média por adição incremental; "
            "usar demanda apenas como desempate/semente; manter cobertura para viabilidade de 11 km.")
