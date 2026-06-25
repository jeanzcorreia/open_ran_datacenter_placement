# Fase 5 — Parte 1b: sanity de escala (antes da otimização completa)

**Data:** 2026-06-24. **Objetivo:** dimensionar a Fase 5 e decidir cap de candidatos.
**O que foi rodado:** UMA execução reduzida (pop=100, n_gen=20, 1 seed) de `nsga2`, `nsga3`,
`moead`, `greedy` na MENOR (Vitória, 270 sites) e MAIOR (Belo Horizonte, 1111 sites) cidade,
**modo justo** (`FairODCProblem`), multi-operadora. **ReEvo NÃO foi rodado** (Fase 4 ainda
executando, API-bound, ~7% CPU). **As 10 cidades NÃO foram rodadas.**

## 1. Tempos medidos (reduzido, wall-clock)

| método | Vitória (429 cli × 270 sites) | BH (1524 cli × 1111 sites) |
|---|---:|---:|
| nsga2 | 0.34 s | 6.40 s |
| nsga3 | 0.33 s | 6.42 s |
| moead | 0.85 s | 6.78 s |
| greedy | 0.13 s | 8.96 s |

A maior instância (BH) custa **< 9 s por método** na config reduzida. O laço de avaliação
(`assign_clients`, O(n_clients × n_sites_ativos)) domina; sorting/seleção do pymoo é desprezível
nesse porte → o tempo dos EAs escala ~linearmente com o nº de avaliações (pop × n_gen).

## 2. Extrapolação para a config cheia (pop=300, n_gen=60)

Modelo: EAs `t_full(1 seed) = (a + b·n_cli·n_sites)·9` (fator 9 = 18000/2000 avaliações; ajuste
linear de 2 pontos). Greedy `t = a + b·n_sites²·n_cli` (independente de pop/gen/seed).

| cidade | sites | nsga2 | nsga3 | moead | greedy |
|---|---:|---:|---:|---:|---:|
| Belo Horizonte | 1111 | 57.6 s | 57.8 s | 61.0 s | 9.0 s |
| Manaus | 1009 | 40.0 s | 40.1 s | 43.7 s | 5.7 s |
| Curitiba | 975 | 45.0 s | 45.1 s | 48.7 s | 6.2 s |
| Goiânia | 722 | 29.0 s | 29.0 s | 33.0 s | 3.0 s |
| Recife | 709 | 21.6 s | 21.6 s | 25.8 s | 2.2 s |
| Natal | 453 | 8.9 s | 8.8 s | 13.4 s | 0.6 s |
| Campo Grande | 431 | 6.4 s | 6.3 s | 10.9 s | 0.4 s |
| João Pessoa | 414 | 7.9 s | 7.8 s | 12.4 s | 0.5 s |
| Florianópolis | 350 | 4.5 s | 4.4 s | 9.1 s | 0.2 s |
| Vitória | 270 | 3.1 s | 3.0 s | 7.7 s | 0.1 s |

**(valores por 1 seed.)**

## 3. Tempo TOTAL estimado da Fase 5 (10 cidades × 4 baselines)

EAs com **5 seeds**; greedy **1** (determinístico):

| método | total | seeds |
|---|---:|---|
| nsga2 | 18.7 min | ×5 |
| nsga3 | 18.7 min | ×5 |
| moead | 22.1 min | ×5 |
| greedy | 0.5 min | ×1 |
| **TOTAL (1 core, sequencial)** | **~60 min (1.0 h)** | |
| **TOTAL (12 cores, paralelo)** | **~6 min** | |

## 4. Conclusão e cap de candidatos

**A config cheia é VIÁVEL sem cap:** ~1 h num núcleo, ~6 min paralelizando por cidade/seed nos
12 núcleos. A maior tarefa única (BH, 1 método, 1 seed) é ~1 min. **Recomendação: rodar
todos-os-sites (sem cap).** As opções de cap abaixo são **contingência** (ex.: se forem somados
muitos métodos/seeds, ou se a versão LLM/ReEvo — que reavalia muitas heurísticas por geração —
elevar o custo de CPU nas cidades grandes):

- **(a) Capar candidatos via KMeans** (n_var = k fixo) só nas cidades > 700 sites (BH, Manaus,
  Curitiba, Goiânia, Recife), mantendo todos-os-sites nas pequenas:
  - k=200 → **~21 min** (1 core) / ~2 min (12 cores)
  - k=300 → **~26 min** (1 core) / ~3 min (12 cores)
- **(b) Manter todos-os-sites** e reduzir pop/gen/seeds nas grandes (ex.: pop=200, n_gen=40,
  3 seeds nas > 700 sites) → reduz ~2,3× o custo dessas cidades, sem reintroduzir o KMeans.

**A escolha do cap (se houver) é decisão do arquiteto.**
