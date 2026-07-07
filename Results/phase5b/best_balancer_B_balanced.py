# melhor BALANCEADOR B_balanced: imbN=2.185 f2N=0.271 agg=0.465 origin=seed
import numpy as np

def place_odcs(instance, n_active):
    D = instance.distances                       # (n_clients, n_sites)
    n_sites = instance.n_sites
    n = int(max(1, min(n_active, n_sites)))
    
    # Inicialize a lista de sites selecionados com o site mais próximo do cliente mais distante
    nearest_to_furthest_client = np.argmax(D.max(axis=0))
    selected = [nearest_to_furthest_client]
    
    while len(selected) < n:
        sub = D[:, selected]                      # (n_clients, len(selected))
        nearest_idx = sub.argmin(axis=1)          # ODC mais próximo de cada cliente
        load = np.zeros(len(selected))              # Carga por ODC ativo
        np.add.at(load, nearest_idx, instance.client_demand)
        
        # Encontre o ODC mais carregado e os clientes que estão nesse ODC
        max_load_idx = np.argmax(load)
        clients_in_max_load_odc = np.where(nearest_idx == max_load_idx)[0]
        
        # Calcule a distância média dos clientes no ODC mais carregado aos outros sites
        distances_to_other_sites = D[clients_in_max_load_odc, :].mean(axis=0)
        distances_to_other_sites[selected] = np.inf  # Evitar selecionar os sites já selecionados
        
        # Selecione o site que minimiza a distância média para aliviar o ODC mais carregado
        next_site = int(np.argmin(distances_to_other_sites))
        selected.append(next_site)
    
    return selected
