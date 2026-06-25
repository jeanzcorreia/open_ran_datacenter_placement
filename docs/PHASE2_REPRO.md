# Fase 2 — Encapsulamento do problema + baseline NSGA-II + reprodução de Natal

**Fase:** 2 (encapsular `src/` do problema e do baseline; reproduzir Natal vs `Results/`).
**Data:** 2026-06-23.
**Escopo executado:** ambiente fixado; `src/problem` (Instance + ODCPlacementProblem); `src/optimizers` (base + nsga2, com persistência da fronteira); reprodução de Natal nos 4 cenários de candidatos e comparação com `Results/`. **NÃO** foram implementados outros baselines (NSGA-III/MOEA-D/random/greedy) nem nada de LLM (Fases 3–4).

**Resultado em uma frase:** o baseline encapsulado **reproduz `Results/` de Natal exatamente** em nº de ODCs ativos e capacidade média/ODC (erro 0), e a menos de **0,02–0,07 km** na distância média — divergência residual atribuível 100% à não-determinância do KMeans entre versões do scikit-learn (centróides), não à lógica de avaliação.

---

## 0. Ambiente — versão do pymoo descoberta, fixada e documentada

| Pacote | Versão fixada | Observação |
|---|---|---|
| Python | 3.10.12 | ambiente do host |
| **pymoo** | **0.6.1.3** | escolhida por compatibilidade de API (abaixo) |
| numpy | 2.0.0 | já presente; pymoo 0.6.1.3 importa e roda sob numpy 2.0.0 (smoke test OK) |
| pandas | 2.2.3 | leitura de CSV |
| scikit-learn | 1.5.2 | KMeans (gera os candidatos a ODC) |
| matplotlib | 3.9.3 | dependência transitiva do pymoo / plots futuros |

**Por que pymoo 0.6.1.3.** O parser original importa exatamente:
`from pymoo.algorithms.moo.nsga2 import NSGA2`, `from pymoo.core.problem import Problem`,
`from pymoo.optimize import minimize`, `from pymoo.termination.default import DefaultSingleObjectiveTermination`.
O caminho `pymoo.termination.default` é da **série 0.6.x** (a 0.5.x usava `pymoo.util.termination.default`, que aparece **comentado** no parser — `odc_placement_parser.py:20`). A 0.6.1.3 (última 0.6.x) satisfaz toda a API e ainda aceita os kwargs **legados** `n_obj`/`n_constr` do construtor de `Problem` (`n_constr=2` é mapeado para `n_ieq_constr=2`, `n_eq_constr=0` — confirmado em runtime). Smoke test executado: instanciar `Problem(n_var=5,n_obj=3,n_constr=2,xl=0,xu=1)`, rodar `NSGA2(pop_size=20)` + `DefaultSingleObjectiveTermination(...)` + `minimize(...)` sob numpy 2.0.0 → OK.

`requirements.txt` criado em `open_ran_datacenter_placement-main/requirements.txt` com os pins acima. *(O parser ORIGINAL também usa `contextily`, `pyproj`, `imageio`, `tqdm` — só para mapas/GIF; a camada `src/` não depende deles.)*

```bash
pip install -r requirements.txt
python -m src.reproduce_natal            # reproduz os 4 cenários de Natal
```

---

## 1. Arquivos criados

```
open_ran_datacenter_placement-main/
  requirements.txt                      # pins de versão
  src/__init__.py
  src/problem/__init__.py
  src/problem/instance.py               # CSV -> Instance (SEM dedupe; KMeans; Haversine)
  src/problem/odc_problem.py            # ODCPlacementProblem: física EXATA + evaluate()/to_pymoo()
  src/optimizers/__init__.py
  src/optimizers/base.py                # Optimizer (ABC) + ParetoSet (PERSISTE X e F)
  src/optimizers/nsga2.py               # baseline NSGA-II espelhando o original
  src/reproduce_natal.py                # harness de reprodução (Fase 2)
  results/phase2/Natal_k{27,18,13,0}/   # pareto.npz + pareto_meta.json (fronteiras salvas)
```

Nenhum arquivo original foi alterado. Toda a lógica nova vive em `src/`.

**Contrato implementado** (igual a CLAUDE.md §6):
- `ODCPlacementProblem.evaluate(x) -> EvalResult(F, feasible, info)`, com `F` shape `(3,)` (sinal de `F[0]` preservado), `feasible == (G==0)`, `info{n_odc, n_odc_nonempty, total_cap, mean_cap_per_odc, mean_fiber_km, max_fiber_km, mean_fiber_per_odc_km, viol}`. Tolera `float('inf')` em `F[2]`.
- `ODCPlacementProblem.evaluate_population(X) -> (F, G)` reproduz `_evaluate` do original; `.to_pymoo()` devolve o `Problem` do pymoo para o NSGA-II.
- `Optimizer.solve(instance, budget, seed) -> ParetoSet`; `ParetoSet.save/load` (npz + json).

---

## 2. Confirmações pedidas pelo arquiteto

### (a) O que `capacity_total` representa fisicamente

`capacity` é acumulada em `evaluate_trial` por `capacities[closest_odc] += client["cpu_cores"]` e `total_capacity = np.sum(capacities)` (`odc_placement_parser.py:151,160`). Como **todo cliente é atribuído a exatamente um ODC ativo**, a soma sobre ODCs é igual à soma sobre clientes:

> **`total_capacity` = Σ_clientes `cpu_cores` = demanda agregada de CPU SERVIDA** (carga de banda-base em cores), onde `cpu_cores = ceil((bandwidth_mhz/100)·14)`. Para Natal isso é **2315 cores**, constante.

Consequências:
- **É demanda servida (throughput de processamento), NÃO capacidade instalada/ociosa.** Por isso `F[0] = −total_capacity·w0` está codificado como "maximizar" (negativo, pois o pymoo minimiza): servir mais demanda é melhor.
- **Mas é invariante** entre soluções viáveis (qualquer seleção com ≥1 ODC serve todos os 170 clientes ⇒ soma = 2315). Logo **o objetivo de capacidade é inerte** (não discrimina soluções) no modelo atual — só o caso degenerado "zero ODC" daria 0.
- **Sinal correto:** mantemos `F[0] = −capacidade` no **modo reprodução** (fidelidade ao original). Para o **modo justo** (Fase 3), como a capacidade servida é constante, esse objetivo não contribui para a fronteira; se quisermos que ele discrimine, será preciso modelar **capacidade instalada por ODC** (custo) ou demanda potencialmente não-servida — decisão a registrar na Fase 3. Verificado: `total_cap` reportado pelo nosso `evaluate` = 2315 em todos os cenários viáveis.

### (b) Como `k = len(initial_odcs)` sai do cenário RUs/N

`k` é o argumento `--odcs` das **campanhas** (não computado em código; hardcoded no YAML). Em `Campaigns/Placement_Natal_Case_1_2/Placement_Natal_Case_1_2.yaml`:

```yaml
odcs:
    - 0
    - 27
    - 18
    - 13
```

Esses valores são **`floor(RUs/N)` com `RUs` = nº de `cell_site_id` ÚNICOS = 55** (não as 170 linhas):

| Cenário | N | `floor(55/N)` | `--odcs` (k) |
|---|---|---|---|
| RUs/2 | 2 | 27 | **27** |
| RUs/3 | 3 | 18 | **18** |
| RUs/4 | 4 | 13 | **13** |
| "ODCs = O-RUs" | — | — | **0** → `k = n_clients = 170` → refit KMeans ⇒ **55** centróides |

Confirmação cruzada (Manaus): `odcs ∈ {0, 45, 30, 22}` = `90/{2,3,4}` (90 sites únicos). Portanto **a base do `RUs/N` é o nº de sites únicos**, embora o otimizador trate as 170 linhas como clientes (sem dedupe). Para `k=0`, o KMeans pede 170 clusters sobre 55 coordenadas distintas → dispara `ConvergenceWarning` → refit com 55 (replicado em `src/problem/instance.py:generate_initial_odcs`).

> **Nota de rótulo importante:** o nº de ODCs **ativos** na solução ótima é **igual a `k`** (todos os candidatos são ativados), porque o objetivo efetivo é **distância pura** (ver §3/§4). Logo o cenário **RUs/2 produz 27 ODCs ativos** — e **não** "~13" como sugere a referência informal do paper (13 ODCs correspondem a `k=13`, isto é, **RUs/4**).

---

## 3. Tabela de reprodução (Natal) — nosso `src/` vs `Results/` vs paper

Cenário do paper (das campanhas): `cpuper100=14`, `maxdistance=11 km`, `capacity=1000 cores`, pesos `(wcpu,wodc,wd)=(0,0,1)`, `pop=300`, `n_gen=60`. Todas as soluções são **viáveis**. "dist" = distância média cliente↔ODC ativo mais próximo (= objetivo `F[2]` bruto, proxy de fibra de fronthaul). Métricas operacionais médias sobre os 20 JOBs presentes em `Results/`.

| Cenário (k → candidatos) | n_odc ativos (nosso) | n_odc ativos (`Results/`) | cap/ODC (nosso) | cap/ODC (`Results/`) | dist km (nosso) | dist km (`Results/`) | Δdist |
|---|---|---|---|---|---|---|---|
| **RUs/2** (k=27→27) | **27** | 27.0 ± 0.0 | **85.74** | 85.74 | 0.3709 | 0.3536 | 0.017 |
| **RUs/3** (k=18→18) | **18** | 18.0 ± 0.0 | **128.61** | 128.61 | 0.6142 | 0.5929 | 0.021 |
| **RUs/4** (k=13→13) | **13** | 13.0 ± 0.0 | **178.08** | 178.08 | 0.8592 | 0.7909 | 0.068 |
| **O-RUs** (k=0→55) | **55** | 55.0 ± 0.0 | **42.09** | 42.09 | 0.0000 | 0.0000 | 0.000 |

- **nº de ODCs ativos:** casa **exatamente** (erro 0) em todos os cenários.
- **capacidade média/ODC:** casa **exatamente** (= 2315/n_ativos: 2315/27=85.74, /18=128.61, /13=178.08, /55=42.09). Isso é estrutural — independe das posições dos centróides.
- **distância média:** casa com erro **0,017–0,068 km** (≤ ~9%). Origem da diferença em §4.
- **Determinismo (verificado):** `Results/` tem desvio-padrão **0** entre as 20 seeds em todas as métricas; rodando nosso NSGA-II com seeds 1/42/999 obtemos `dist = 0.370858` idêntico. Causa: o objetivo efetivo é distância pura e a capacidade total é invariante ⇒ o ótimo **ativa todos os `k` candidatos** ⇒ a solução não depende da seed do GA (só do KMeans, que tem `random_state=0`). A "fronteira de Pareto" salva colapsa, de fato, num **único ponto operacional** (verificado: 1 linha de `F` única = `[-0, 0, 0.3709]` para k=27).

**Referência do paper (Natal, aproximada, conforme enunciado):** ~13 ODCs, fibra ~3,76 km, ~105 CPUs/ODC — **não reproduzível a partir do repositório** (ver §4.2).

---

## 4. Divergências residuais (documentadas)

### 4.1 Distância média: KMeans entre versões do scikit-learn
A única fonte de diferença é a posição dos centróides candidatos. Verificado: para k=27 a maioria dos centróides bate (mediana da diferença NN = 0 km), mas alguns clusters convergem para posições levemente distintas (máx ~0,75 km), porque o **default de `n_init` do KMeans mudou** (`'auto'`=1 em sklearn 1.5.2 vs `10` em versões antigas) e o algoritmo/ordenação interna evoluiu. Isso **não afeta** `n_odc` nem `cap/ODC` (estruturais, exatos), só a distância média (Δ ≤ 0,07 km).
- **Mitigação:** fixamos `scikit-learn==1.5.2` ⇒ **nossos** resultados são 100% reprodutíveis. Reproduzir a distância **exata dos autores** exigiria a versão de sklearn deles (não documentada no repo). Tentativas com `n_init∈{auto,1,10}` não casaram exatamente — confirma que é dependência de versão, não de parâmetro acessível.

### 4.2 Números do paper (~13 ODCs / 3,76 km / 105 CPU) não casam com o repositório
- "~13 ODCs" corresponde a **k=13 (RUs/4)**, não a RUs/2 (que dá 27). Há **inconsistência de rótulo** na referência informal.
- "~105 CPU/ODC" não bate com nenhum cenário de Natal (cap/ODC ∈ {42, 86, 129, 178}) — 105 cores/ODC implicaria ~22 ODCs (2315/105), valor que **não existe** em Natal (existe em **Manaus**, RUs/4 = 22 ODCs).
- "~3,76 km" não bate com Natal (dist média ∈ {0; 0,35; 0,59; 0,79}; fibra/ODC ∈ {4,3; 5,9; 11,2}) nem com Manaus.
- **Conclusão:** os números citados do paper parecem **aproximados / de uma figura com definição ou agregação diferente** (possivelmente misturando cidades/casos). A **validação concreta e auditável é o `Results/` do repositório**, que reproduzimos exatamente nas métricas estruturais. Registrado como divergência residual; não bloqueia a Fase 3.

### 4.3 Fronteira degenerada
Com pesos `(0,0,1)` e capacidade invariante, o problema roda **efetivamente mono-objetivo** (distância), e a fronteira colapsa num ponto (todos os ODCs ativos). É exatamente o comportamento que o **modo justo** da Fase 3 (pesos `(1,1,1)` + magnitude **contínua** de violação) vai corrigir para explorar a fronteira 3-objetivo de verdade. Em modo reprodução, manter o comportamento original é o correto.

---

## 5. Conformidade com as HARD RULES (CLAUDE.md §9)

- **§9.2** Não reescrevemos o original; encapsulamos em `src/`. Não replicamos a normalização min-max da variante GPT (`odc_problem.py` usa valores brutos ponderados).
- **§9.3** `encoding='latin1'` na leitura; **sem dedupe** (cada linha = cliente, `oru_id=index+1`). *(Nota: o parser original lia em utf-8 default; para `Natal.csv` as colunas usadas são ASCII ⇒ latin1 e utf-8 são byte-idênticos, não afeta a reprodução.)*
- **§9.5** **Fronteira persistida** (`ParetoSet.save` → `pareto.npz` com `X`,`F`,`feasible` + `pareto_meta.json`), reconstruída de `res.opt` (fallback `res.X/F`/histórico).
- **§9.7** pymoo **fixado** (0.6.1.3); seeds fixas; resultados salvos em `results/phase2/` com metadados (pop, gerações, n_eval, pesos, params do cenário, versão do pymoo).
- Métricas **operacionais/de Pareto** (nº ODCs, CPU/ODC, fibra), nunca acurácia/F1.

---

## 5b. Validação do código `src/` (revisão adversarial)

Dois revisores independentes auditaram a camada `src/` em modo somente-leitura:

- **Equivalência semântica vs. o parser original** — veredito: **nenhuma divergência** que altere objetivo, restrição ou atribuição. Confirmados ponto a ponto: limiar estrito `x>0.5`; sinais/ordem de `F` (`F[0]` negativo; `F[1]`=nº de ODCs **selecionados**, não os não-vazios); `G[0]`=capacidade/`G[1]`=distância com convenção 0/1; equivalência de `np.add.at` ao laço de acumulação original; refit do KMeans em `ConvergenceWarning`; `ceil` por cliente; `oru_id=index+1`; ordem de coordenadas `[lat,lon]`. Notas de impacto zero: `cpu_cores` como `float` (valores já são inteiros pós-`ceil` ⇒ somas idênticas) e leitura `latin1` vs utf-8 (colunas ASCII ⇒ idêntico).
- **Caça a bugs / casos de borda no otimizador** — confirmou limpos: robustez de `_extract_front` (`res.opt`→`res.X/F`→histórico, com guardas), recomputo de viabilidade `np.all(G<=0)`, round-trip `ParetoSet.save/load`, alinhamento de índices `sel`/`caps`, e fidelidade do NSGA-II (operadores **default** do pymoo, `pop_size=300`, terminação idêntica, `seed`/`save_history` repassados).

**Correção aplicada (1 bug latente):** em `odc_problem.py:_weighted_F`, uma solução sem ODC ativo tem `avg_dist=inf`; com um peso de objetivo igual a 0 (modos de ablação suportados, p.ex. `(1,0,0)`), `inf*0` produziria **`NaN`**, que envenena silenciosamente a ordenação não-dominada do pymoo e o HV/IGD+ da Fase 3. Agora um peso zero **zera o termo** (`f_i = 0.0 if w_i==0 else ...`). Verificado: nenhum `NaN` para pesos `(0,0,1)/(1,0,0)/(0,1,0)/(1,1,1)`; e a **tabela de reprodução permanece idêntica** (em modo `(0,0,1)` só muda `-0.0`→`0.0`). Também endurecido `_fmt` em `reproduce_natal.py` para formatar escalares numpy (`np.int64` não é subclasse de `int`). Itens informacionais sem ação: tuplas em `meta` recarregam como listas no JSON (irrelevante para metadados).

## 6. Próximos passos sugeridos (para a Fase 3, fora do escopo agora)

1. Implementar `src/eval/metrics.py` (HV, IGD+, spacing, spread) e `reference_front.py`.
2. Implementar o **modo justo**: expor magnitude **contínua** de violação no `evaluate` (além das flags 0/1) e rodar com pesos `(1,1,1)` para todos os métodos, inclusive o baseline.
3. Decidir a modelagem de capacidade (servida vs instalada) para que `F[0]` deixe de ser inerte (ver §2a).
4. Baselines adicionais: NSGA-III, MOEA/D, busca aleatória, greedy opcional (NÃO k-means, que gera os candidatos).
```
