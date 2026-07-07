# Heurística vencedora REPRESENTATIVA (ReEvo MULTI-CIDADE, Fase 5a corrigida)
# seed=1 origem=seed robusta=True train_meanHV=0.9067 train_maximin=0.7903
# proveniência por seed: {1: 'seed', 2: 'seed_strong', 3: 'seed_strong', 4: 'seed'}
import numpy as np

def place_odcs(instance, n_active):
    D = instance.distances                       # (n_clients, n_sites)
    n_clients, n_sites = instance.n_clients, instance.n_sites
    
    # Inicialize a lista de ODCs ativos
    selected = []
    
    # Calcular a demanda total dos clientes por site
    demand_per_site = np.zeros(n_sites)
    for c in range(n_clients):
        nearest_idx = D[c].argmin()
        demand_per_site[nearest_idx] += instance.client_demand[c]
    
    # Ordenar sites pela capacidade disponível em ordem crescente
    sorted_indices = np.argsort(demand_per_site)
    
    # Selecionar os n_active sites com menor carga
    for i in range(n_active):
        selected.append(sorted_indices[i])
    
    # Garantir que todos os clientes estejam cobertos e a capacidade não seja excedida
    while len(selected) < n_active:
        cur = D[:, selected].min(axis=1)
        gain = (cur[:, None] - np.minimum(cur[:, None], D)).sum(axis=0)
        gain[selected] = -1.0
        new_site = int(np.argmax(gain))
        if demand_per_site[new_site] + instance.client_demand[cur.argmin()] <= instance.max_capacity:
            selected.append(new_site)
            demand_per_site[new_site] += instance.client_demand[cur.argmin()]
    
    return selected[:n_active]
