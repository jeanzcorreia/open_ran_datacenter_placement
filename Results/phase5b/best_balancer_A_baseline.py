# melhor BALANCEADOR A_baseline: imbN=3.082 f2N=0.294 agg=0.940 origin=seed
import numpy as np

def place_odcs(instance, n_active):
    D = instance.distances  # (n_clients, n_sites)
    n_clients = instance.n_clients
    client_demand = instance.client_demand
    max_distance = instance.max_distance
    max_capacity = instance.max_capacity
    
    # Ordena os clientes pela demanda de CPU em ordem decrescente
    sorted_indices = np.argsort(client_demand)[::-1]
    
    selected = []
    load = np.zeros(instance.n_sites)
    
    for c in sorted_indices:
        if len(selected) >= n_active:
            break
        
        nearest_idx = D[c].argmin()
        
        # Verifica se adicionar este cliente ao ODC atual é viável
        if load[nearest_idx] + client_demand[c] <= max_capacity and D[c, nearest_idx] <= max_distance:
            selected.append(nearest_idx)
            load[nearest_idx] += client_demand[c]
    
    # Se não foi possível ativar n_active ODCs, complete com os de menor distância média
    if len(selected) < n_active:
        remaining = n_active - len(selected)
        remaining_indices = np.argsort(D.min(axis=0))[:remaining]
        selected.extend(remaining_indices)
    
    return sorted(selected[:n_active])
