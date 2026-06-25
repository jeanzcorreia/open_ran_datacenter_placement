# Heurística vencedora evoluída pelo ReEvo (origem=seed, HV interno=0.9506, backend=groq(gen)+gemini(reflect))
import numpy as np

def place_odcs(instance, n_active):
    """
    Coloca ODCs em uma rede 5G de forma eficaz, considerando o trade-off entre o número de ODCs e a distância média de fronthaul.
    
    :param instance: Instância do problema, contendo informações sobre os sites, clientes, distâncias, demandas e limites.
    :param n_active: Número de ODCs a serem ativados.
    :return: Lista de índices dos sites a serem ativados.
    """
    
    # Cria uma lista de todos os índices de sites
    all_sites = list(range(instance.n_sites))
    
    # Gulosamente, seleciona os n_active sites com menor distância média a algum cliente
    distances_to_clients = instance.distances.min(axis=0)  # (n_sites,)
    initial_sites = np.argsort(distances_to_clients)[:n_active]  # (n_active,)
    
    # Refina a seleção mediante trocas locais (swap)
    for _ in range(100):  # número de iterações
        selected_sites = initial_sites.copy()
        
        # Avalia a qualidade da solução atual
        distances_to_selected = instance.distances[:, selected_sites]  # (n_clients, n_active)
        nearest_idx = distances_to_selected.argmin(axis=1)  # (n_clients,)
        nearest_dist = distances_to_selected.min(axis=1)  # (n_clients,)
        mean_fronthaul = nearest_dist.mean()  # f2
        load = np.zeros(n_active); np.add.at(load, nearest_idx, instance.client_demand)  # (n_active,)
        feasible = (load.max() <= instance.max_capacity) and (nearest_dist.max() <= instance.max_distance)
        
        # Se a solução atual for viável, tenta melhorá-la mediante swap
        if feasible:
            best_mean_fronthaul = mean_fronthaul
            
            # Tentativa de swap para cada par de sites
            for i in range(n_active):
                for j in all_sites:
                    if j not in selected_sites:
                        new_sites = selected_sites.copy()
                        new_sites[i] = j
                        
                        # Avalia a qualidade da solução após o swap
                        distances_to_new_sites = instance.distances[:, new_sites]  # (n_clients, n_active)
                        new_nearest_idx = distances_to_new_sites.argmin(axis=1)  # (n_clients,)
                        new_nearest_dist = distances_to_new_sites.min(axis=1)  # (n_clients,)
                        new_mean_fronthaul = new_nearest_dist.mean()  # f2
                        new_load = np.zeros(n_active); np.add.at(new_load, new_nearest_idx, instance.client_demand)  # (n_active,)
                        new_feasible = (new_load.max() <= instance.max_capacity) and (new_nearest_dist.max() <= instance.max_distance)
                        
                        # Aceita o swap se a nova solução for melhor e viável
                        if new_feasible and new_mean_fronthaul < best_mean_fronthaul:
                            initial_sites = new_sites
                            best_mean_fronthaul = new_mean_fronthaul
        
        # Se a solução atual não for viável, tenta torná-la viável mediante a troca de um site
        else:
            for i in range(n_active):
                for j in all_sites:
                    if j not in selected_sites:
                        new_sites = selected_sites.copy()
                        new_sites[i] = j
                        
                        # Avalia a qualidade da solução após a troca
                        distances_to_new_sites = instance.distances[:, new_sites]  # (n_clients, n_active)
                        new_nearest_idx = distances_to_new_sites.argmin(axis=1)  # (n_clients,)
                        new_nearest_dist = distances_to_new_sites.min(axis=1)  # (n_clients,)
                        new_load = np.zeros(n_active); np.add.at(new_load, new_nearest_idx, instance.client_demand)  # (n_active,)
                        new_feasible = (new_load.max() <= instance.max_capacity) and (new_nearest_dist.max() <= instance.max_distance)
                        
                        # Aceita a troca se a nova solução for viável
                        if new_feasible:
                            initial_sites = new_sites
                            break
        
        # Verifica se houve melhoria na solução
        if mean_fronthaul == best_mean_fronthaul:
            break
    
    return initial_sites.tolist()
