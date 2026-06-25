# Fase 3 — Modo justo (MO genuíno), métricas de Pareto e baselines

**Fase:** 3. **Data:** 2026-06-23.
**Escopo executado:** (1) `evaluate` do **modo justo** (preservando o modo reprodução da Fase 2); (2) **portão de não-degenerescência** (PASSOU); (3) métricas HV/IGD+/spacing/spread + fronteira de referência; (4) baselines evolutivos (NSGA-III, MOEA/D) + busca aleatória + greedy (opcional); (5) runner com tabela comparativa. **SEM LLM** (Fase 4).

**Resultado em uma frase:** a reformulação restaura um **trade-off de Pareto genuíno** (#ODCs × distância de fronthaul) com fronteira de **53 pontos** abrangendo **3→55 ODCs**; sobre Natal, o **NSGA-II domina em convergência** (maior HV, menor IGD+), com um **greedy de construção** quase empatando a custo ~150× menor, e a **busca aleatória** como piso claro.

---

## 1. Por que o modo justo (motivação da degenerescência)

A formulação da referência (modo reprodução, Fase 2) é **degenerada**: capacidade servida é constante (`F[0]` inerte), `#ODCs` tem peso 0 (não é otimizado) e o objetivo efetivo vira distância pura — o ótimo **ativa TODOS os candidatos** e a fronteira **colapsa em 1 ponto**. Para comparar otimizadores (e, na Fase 4, o LLM) de forma significativa, precisamos de um **MO de verdade**, com objetivos que se opõem e restrições que discriminam.

Os **dois modos coexistem** e são separados em `src/problem/odc_problem.py`:
- `ODCPlacementProblem` — modo **reprodução** (Fase 2), intacto e revalidado (tabela de Natal idêntica após o refactor).
- `FairODCProblem` — modo **justo** (Fase 3), abaixo.

---

## 2. Formulação final do modo justo

| Elemento | Definição |
|---|---|
| **Candidatos a ODC** | **TODOS os sites únicos** da cidade (dedupe por `cell_site_id`, lat/lon do 1º registro). Natal: `n_var = 55`. *(`load_instance_sites`)* |
| **Clientes** | Todas as linhas do CSV, **SEM dedupe** (`oru_id = index+1`), como na Fase 2. |
| **Carga por cliente** | `cpu_cores = ceil((bandwidth_mhz/100)·14)` (idêntico ao parser). Demanda total Natal = **2315 cores**. |
| **Genótipo** | binário `x ∈ [0,1]^n_var`, limiar **`x>0.5`** ⇒ ODC ativo no site *i*. |
| **Atribuição** | cada cliente → ODC **ativo mais próximo** (Haversine pré-computado). |
| **Objetivos (MIN)** | `f1 = nº de ODCs ativos`; `f2 = distância média de fronthaul (km)`. |
| **Restrições (magnitude CONTÍNUA; viável ⇔ ambas = 0)** | `g_cap = Σ_ODC max(0, carga_ODC − 1000)` (sobrecarga total, cores); `g_dist = Σ_cli max(0, dist_atribuída − 11)` (excesso total de distância, km). |
| **Params do cenário** | `max_km=11`, `capacity=1000`, `cpuper100=14`. |

`evaluate(x) -> (F=[f1,f2], feasible, info)` com `info = {n_odc, n_odc_nonempty, mean_fiber_km, total_overload, total_dist_excess, max_load_per_odc, viol}`. Caso degenerado "sem ODC ativo" retorna **valores finitos** (sentinela de distância = `max(distances)`) e é corretamente **inviável** — evita `inf`/`nan` nas métricas. **Não** foi replicada a normalização min-max não-estacionária da variante GPT.

**Diferença-chave vs. modo reprodução:** restrições com **magnitude contínua** (o constrained-domination do NSGA-II precisa do "quão longe" da viabilidade; o original se prejudicava com flags 0/1) e objetivos que se opõem de fato. Todos os métodos (inclusive o baseline NSGA-II) usam este mesmo `evaluate`.

---

## 3. Portão de não-degenerescência — **PASSOU**

NSGA-II (pop 300, terminação `n_gen=60`) rodado em Natal, 5 seeds; união dos não-dominados viáveis:

- **52–53 pontos distintos** na fronteira (≥5 exigido). ✔
- **Faixa de #ODCs ativos: 3 → 55** (do mínimo viável ao máximo), com trade-off **monótono e claro** #ODCs × distância. ✔
- **Mínimo viável = 3 ODCs** (coerente com a cota de capacidade ⌈2315/1000⌉ = 3 + cobertura de distância ≤ 11 km).

Amostra da fronteira (NSGA-II, união de seeds):

| nº ODCs | dist. média (km) | | nº ODCs | dist. média (km) |
|---:|---:|---|---:|---:|
| 3 | 2.788 | | 20 | 0.520 |
| 4 | 1.911 | | 27 | 0.359 |
| 5 | 1.587 | | 35 | 0.219 |
| 8 | 1.116 | | 45 | 0.090 |
| 13 | 0.780 | | 55 | 0.000 |

A fronteira **não** colapsa (contraste direto com o modo reprodução). Gráfico em `results/phase3/Natal_sites/fronts.png`. **Portão satisfeito ⇒ prosseguimos aos baselines.**

---

## 4. Métricas e fronteira de referência

- **HV** (Hypervolume), **IGD+**, **Spacing** (Schott), **Spread Δ** (Deb, 2 objetivos) — em `src/eval/metrics.py`. HV/IGD+ via `pymoo.indicators`; spacing/spread implementados (pymoo 0.6.x não os fornece).
- **Espaço normalizado:** todos os objetivos são normalizados por `(ideal, nadir)` da fronteira de referência antes das métricas (f1 e f2 têm escalas distintas). 
- **Fronteira de referência** = união dos **não-dominados VIÁVEIS** de **todos** os métodos × seeds (`src/eval/reference_front.py`). Para Natal: **53 pontos**, `ideal=[3, 0.0]`, `nadir=[55, 2.746]`.
- **Ponto de referência do HV (fixo e documentado):** `(1.1, 1.1)` no espaço **normalizado** — isto é, nadir + **10% de margem** por objetivo. HV maior é melhor (máximo teórico ≈ 1.21).
- **Inviáveis tratados explicitamente:** excluídos da fronteira de referência e das métricas; runs sem nenhuma solução viável recebem HV = 0. O caso `inf*0=nan` já foi corrigido na Fase 2 e o sentinela finito do modo justo evita `inf` na fronteira.

---

## 5. Tabela comparativa (Natal, modo justo) — média ± IC95% (5 seeds)

Cenário: `max_km=11`, `capacity=1000`, `cpuper100=14`, `pop=300`, `n_gen=60`, orçamento = **18000 avaliações** por run (greedy é determinístico, 55 avaliações).

| Método | HV ↑ | IGD+ ↓ | Spacing ↓ | Spread Δ ↓ | \|fronteira\| | tempo (s) |
|---|---|---|---|---|---|---|
| **nsga2** | **1.0244 ± 0.0025** | **0.0038 ± 0.0007** | 0.0382 ± 0.0175 | 0.5738 ± 0.0818 | 49.2 ± 2.4 | 1.42 |
| nsga3 | 1.0078 ± 0.0062 | 0.0092 ± 0.0015 | 0.0399 ± 0.0194 | 0.6561 ± 0.0606 | 31.2 ± 3.8 | 1.42 |
| moead | 0.9690 ± 0.0092 | 0.0452 ± 0.0074 | 0.0566 ± 0.0162 | 0.7430 ± 0.0357 | 22.6 ± 1.1 | 7.52 |
| random | 0.8677 ± 0.0162 | 0.0539 ± 0.0024 | 0.0278 ± 0.0336 | 0.7357 ± 0.0490 | 25.4 ± 2.1 | 0.44 |
| **greedy** | 1.0228 ± 0.0000 | 0.0056 ± 0.0000 | **0.0193 ± 0.0000** | **0.4527 ± 0.0000** | 52.0 ± 0.0 | 0.01 |

**Leitura:**
- **NSGA-II** vence em **convergência** (maior HV, menor IGD+) e fornece a fronteira mais ampla (~49 pts).
- **Greedy** (construção sobre os sites) **quase empata** o NSGA-II em HV (1.0228 vs 1.0244) e tem o **melhor spacing/spread**, a **~150× menos avaliações** e ~100× menos tempo — baseline determinístico forte porque o subproblema "para cada k, minimizar distância" é essencialmente *k-median*, onde a heurística gulosa é quase ótima. **Não é contaminado** (opera sobre os sites fixos, não gera candidatos como o k-means).
- **NSGA-III** é o segundo EA: boa convergência, fronteira menor.
- **MOEA/D** é o EA mais fraco aqui (menor HV, maior IGD+) e o mais lento — coerente com a limitação de restrições (penalidade estática; ver §6) e com a decomposição escalar sofrendo num espaço com objetivo inteiro (f1).
- **Busca aleatória** é o **piso** inequívoco (menor HV; visivelmente dominada no gráfico).

*(Spacing de `random` é baixo por agrupar poucos pontos perto de ~25 ODCs — baixo spacing com cobertura ruim não é "bom"; ler junto do HV.)*

---

## 6. Ajustes de implementação (decisões registradas)

1. **MOEA/D não suporta restrições no pymoo 0.6.1.3** (`AssertionError`). Solução padrão: a **busca** do MOEA/D usa **penalidade estática** (`FairODCProblem.to_pymoo_penalized`, penalidade somada aos dois objetivos ∝ violação); ao **extrair a fronteira**, as soluções são **reavaliadas no problema VERDADEIRO** (objetivos e viabilidade idênticos aos demais), preservando a comparação justa.
2. **Fronteira = conjunto no espaço de objetivos:** `feasible_nd_front` deduplica por vetor `F` único (várias `X` mapeiam no mesmo `(n_odc, dist)`); duplicatas distorceriam spacing/spread (não afetam HV/IGD+).
3. **Reavaliação uniforme:** *todos* os métodos têm a fronteira recomputada no `FairODCProblem` verdadeiro e reduzida a viável-não-dominado — garante o mesmo espaço de comparação independentemente de como cada algoritmo buscou.
4. **Terminação `n_gen=60` fixa** para todos os EAs (em vez do `DefaultSingleObjectiveTermination` do original, impróprio para MO) ⇒ orçamento idêntico (18000 avaliações) e comparação limpa.
5. **NSGA-III/MOEA-D ref_dirs:** `das-dennis`, 2 objetivos, `n_partitions = pop−1 = 299` ⇒ 300 direções (≈ pop).
6. **Modo reprodução preservado:** `ODCPlacementProblem` intacto; a tabela de Natal da Fase 2 permanece idêntica após o refactor (revalidado).

---

## 7. Arquivos criados/alterados (todos em `src/`, sem tocar no original)

```
src/problem/instance.py     (+ load_instance_sites; refactor _build_clients compartilhado)
src/problem/odc_problem.py   (+ FairODCProblem, assign_clients, to_pymoo_penalized)
src/optimizers/base.py       (+ extract_front_X, feasible_nd_front)
src/optimizers/fair.py       (NSGA2Fair, NSGA3Fair, MOEADFair, RandomSearchFair, GreedyFair)
src/eval/{metrics,reference_front,runner}.py
src/viz/plots.py             (gráfico das fronteiras)
results/phase3/Natal_sites/  (fronteiras .npz por método×seed, summary.json, fronts.png)
```

Comandos:
```bash
python -m src.eval.runner          # roda todos os métodos × seeds e imprime a tabela
python -m src.viz.plots            # gera o gráfico das fronteiras
```

**Validação de código (revisão adversarial):** revisor independente confirmou **zero bugs** na formulação do modo justo + métricas + fronteira de referência (`g_cap`/`g_dist` corretos; sentinela finito no caso sem-ODC; `to_pymoo_penalized` preserva F de soluções viáveis; `load_instance_sites` com candidatos = sites únicos e matriz clientes×sites; normalização, HV, IGD+, spacing de Schott e spread Δ de Deb corretos, com guardas de casos degenerados).

---

## 8. Próximos passos (Fase 4 — fora do escopo agora)

- Otimizador-LLM (ReEvo): laço gerar→avaliar→refletir→evoluir, com `place_odcs(instance) -> x` **sobre os mesmos sites** do modo justo, pontuado pelo **mesmo** `FairODCProblem.evaluate` (HV penalizado por inviabilidade), em **sandbox**; API Anthropic.
- O **greedy forte** é um baseline a bater (não só os EAs): a contribuição do LLM precisa ao menos igualá-lo em HV com boa diversidade.
- Escalar para Manaus/10 cidades (Fase 5) reusando `load_instance_sites`.
