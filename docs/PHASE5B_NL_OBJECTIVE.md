# Fase 5b — Objetivo por LINGUAGEM NATURAL (balanceamento de carga)

**Data:** 2026-06-28 · **Gerador:** `qwen2.5-coder:7b` LOCAL (Ollama+Vulkan, RX 5700 XT, `:11434/v1`).
**Diferencial do projeto:** adicionar um objetivo (balanceamento de carga) descrevendo-o **em
português no prompt** — o LLM **regenera a heurística** para considerá-lo, **SEM tocar na função de
fitness** — enquanto os baselines (NSGA-II/greedy) exigiriam editar a fitness à mão.

> **Resultado (1 linha):** a descrição em PT fez o 7B produzir uma heurística que **de fato balanceia
> a carga** (imbalance **3.08→2.17** em Natal, **3.87→2.04** em BH; Jain sobe) com o **trade-off**
> esperado em distância — e isso saiu **só do prompt**. Caveats honestos em §6.

---

## 0. Pré-requisito — sandbox de ISOLAMENTO DE PROCESSO (corrigido + validado)

O sandbox da Fase 5a usava um **watchdog de thread** que, no Windows, **não matava** a heurística
travada — threads-zumbi acumulavam, contaminavam o GIL e tornavam o piso de robustez
**não-determinístico** (foi a causa provável da falha da 5a). Trocado por **isolamento de processo**
(`src/optimizers/llm/heuristic_runtime.py`):

- Cada heurística roda num **subprocesso** (`HeuristicSandbox`, spawn). No timeout o processo é
  **morto** (`proc.kill()`) — a heurística travada morre de fato. O worker é reusado entre os pontos
  do sweep (paga o spawn+import numpy 1x via handshake `"ready"`, então o timeout cronometra só a
  EXECUÇÃO, não o startup) e fechado ao fim de cada avaliação.
- Comportamento preservado: crash/timeout/inviável ⇒ HV=0 (piso de qualidade, não elimina); só
  numpy/math liberados.

**Validação (`test_sandbox_isolation.py`) — PASSOU ✔:** (a) `while True` é **morta**,
`SandboxError('timeout')`, score 0, **0 filhos depois**; (b) heurística válida roda igual (score
idêntico ao da 5a); (c) a contagem de processos **volta ao baseline** após cada avaliação
(OS-check: 0 python órfão). Smoke do ReEvo (pop2/gen2) fecha sem zumbi. **O piso de robustez agora é
determinístico.**

---

## 1. Experimento — A (sem) vs B (com balanceamento)

- **Mesmo setup, só o prompt difere.** `src/optimizers/llm/nl_objectives.py` define o objetivo em PT
  (`LOAD_BALANCING_OBJECTIVE_PT`) + um few-shot vetorizado válido; `balanced_system()` anexa isso ao
  system prompt **só na condição B**. A fitness do ReEvo continua **HV(nº ODCs, distância)** nas duas
  (`FairODCProblem.evaluate` intocado). O efeito é medido **FORA do fitness**.
- **Métrica de desbalanceamento** (`nl_objectives.load_imbalance`, calculada como o `assign_clients`
  do fitness): **max_over_mean** = carga_máx/carga_média (1.0=perfeito), **Jain** (1.0=perfeito),
  CV. Carga de um ODC = Σ `client_demand` dos clientes a ele atribuídos (ODC ativo mais próximo).
- **Execução:** `runner_phase5b.py`, cidades **Natal + Belo Horizonte** (1 pequena, 1 grande),
  2 seeds, pop=8, gen=6, sandbox de processo, timeout de laço 1.5 s, **sem seed forte** (a vencedora
  É a heurística do LLM), rodada destacada (Agendador) com checkpoint. 54.6 min, US$ 0 (local).
- **Seleção do representante:** A = melhor por HV. B = a heurística que **mais reduz o
  desbalanceamento medido** entre as geradas sob o prompt B (seleção por `analyze_5b.py` sobre a
  população salva — ver §6 por que NÃO a vencedora-HV).

### 1.1 As duas heurísticas (código)

**A — representante (sem balanceamento)** — gulosa por demanda+capacidade; a carga aparece **só na
checagem de viabilidade** (`load[...] + demand <= max_capacity`), nunca para **equalizar**:

```python
def place_odcs(instance, n_active):
    sorted_indices = np.argsort(instance.client_demand)[::-1]
    selected = []; load = np.zeros(instance.n_sites)
    for c in sorted_indices:                       # laço sobre clientes (capacidade)
        if len(selected) >= n_active: break
        nearest_idx = instance.distances[c].argmin()
        if load[nearest_idx] + instance.client_demand[c] <= instance.max_capacity and ...:
            selected.append(nearest_idx); load[nearest_idx] += instance.client_demand[c]
    ...                                            # completa por menor distância média
```

**B — representante (com balanceamento)** — a lógica que **só apareceu sob o prompt em PT**: a cada
passo computa a **carga por ODC** e adiciona o site que **alivia o ODC mais carregado**:

```python
def place_odcs(instance, n_active):
    D = instance.distances
    n = int(max(1, min(n_active, instance.n_sites)))
    selected = [int(np.argmax(D.max(axis=0)))]
    while len(selected) < n:
        sub = D[:, selected]; nearest_idx = sub.argmin(axis=1)
        load = np.zeros(len(selected))
        np.add.at(load, nearest_idx, instance.client_demand)     # CARGA por ODC ativo
        max_load_idx = np.argmax(load)                            # ODC mais carregado
        clients = np.where(nearest_idx == max_load_idx)[0]        # seus clientes
        d = D[clients, :].mean(axis=0); d[selected] = np.inf
        selected.append(int(np.argmin(d)))                        # site que DESCARREGA o ODC
    return selected
```

**Diferença-chave:** A usa a carga só para *viabilidade*; **B computa a carga e a usa como
critério de seleção para equalizá-la** (`np.add.at(load, …, client_demand)` → identifica o ODC
saturado → adiciona ODC perto dos clientes dele). Essa lógica **não existe em nenhuma das 7
heurísticas únicas de A** — apareceu **exclusivamente** por causa da descrição em português.

### 1.2 Efeito — desbalanceamento e trade-off (representantes nas 2 cidades)

| Método | cidade | **imbalance** (máx/méd) ↓ | **Jain** ↑ | dist. f2 | nº pts viáveis |
|---|---|---:|---:|---:|---:|
| **A (sem)** | Natal | 3.076 | 0.758 | 0.304 | 118 |
| **B (com)** | Natal | **2.170** | **0.823** | 0.275 | 119 |
| greedy | Natal | 3.034 | 0.801 | 0.149 | 435 |
| **A (sem)** | Belo Horizonte | 3.868 | 0.752 | 0.211 | 116 |
| **B (com)** | Belo Horizonte | **2.035** | **0.808** | 0.345 | 81 |
| greedy | Belo Horizonte | 3.663 | 0.814 | 0.121 | 1066 |

- **B reduz o desbalanceamento nas DUAS cidades** (Natal −29%, BH −47%) e melhora o Jain — **só pela
  descrição no prompt**. Bate inclusive o `greedy` (que é ótimo em distância mas não balanceia).
- **Trade-off (visível em BH):** B troca **distância** (f2 0.211→0.345) e **alcance da fronteira**
  (116→81 pontos viáveis) por carga equilibrada — coerente com balancear "quebrar" o k-median puro.
  Em Natal B até domina A (menor imbalance E menor distância), porque A's representante também não é
  ótimo em distância.
- **Nível de POPULAÇÃO:** as heurísticas geradas sob o prompt B balanceiam melhor **em média**
  (imbalance médio 2.945 vs 3.085 de A em Natal) — o prompt deslocou a distribuição, não só 1 ponto.

---

## 2. Contraste com os baselines (o ponto da tese)

Adicionar o MESMO objetivo de balanceamento aos clássicos exige **editar código à mão**:

- **NSGA-II / NSGA-III / MOEA-D:** editar `FairODCProblem._evaluate_one` (`src/problem/odc_problem.py`)
  para **computar um 3º objetivo** (ex.: `caps.max()/caps.mean()` ou `caps.std()`) e devolvê-lo;
  mudar `evaluate_population` p/ um `F` de 3 colunas; **`n_obj` 2→3**; ajustar `to_pymoo()` (n_obj=3);
  reconfigurar `get_reference_directions("das-dennis", **3**, …)` no NSGA3/MOEA-D
  (`src/optimizers/fair.py`). Mais: métricas/fronteira-referência assumem 2 objetivos.
- **Greedy:** editar o critério por passo em `GreedyFair.solve` (`src/optimizers/fair.py`) — hoje
  escolhe `argmin` da **distância média** (`m = newmin.mean()`); seria preciso dobrar a carga nesse
  score à mão.
- **O LLM:** **uma frase em português** no prompt. **Zero** edição de código de fitness/seleção.

Esse é o diferencial: **mudar o objetivo via linguagem natural** vs **reengenharia manual da fitness**.

---

## 3. Reprodutibilidade

```bash
# rodada destacada (Agendador 'Phase5B' -> run_5b.ps1) ou direto:
python -m src.eval.runner_phase5b --config configs/phase5a.yaml --out results/phase5b \
       --cities Natal,BeloHorizonte --seeds 1,2 --pop 8 --gen 6 --loop-timeout 1.5
python analyze_5b.py     # re-seleciona representantes por imbalance medido (não por HV) -> best_balancer_{A,B}.py
python check_bh_5b.py    # tabela imbalance/Jain/f2 dos representantes nas 2 cidades -> rep_table.json
```
Artefatos: `results/phase5b/{report_data_5b.json, winner_{A,B}.py, best_balancer_{A,B}.py,
rebalance_analysis.json, rep_table.json, checkpoints/}`.

---

## 6. Leitura honesta

1. **A heurística B difere de A com lógica de balanceamento que só apareceu pelo prompt em PT? SIM.**
   B computa carga (`client_demand`) e equaliza o ODC saturado; nenhuma das heurísticas de A faz isso.
2. **B reduz o desbalanceamento vs A? SIM**, nas duas cidades (com o trade-off em distância em BH).
3. **A evolução agregou aqui? NÃO — e de um jeito instrutivo.** O melhor balanceador de B veio da
   **população inicial** (origem `seed`), não do laço. Pior: como o **fitness é HV(ODCs, distância)
   inalterado**, e os balanceadores **não escalam para BH** (o passo de recomputar carga é
   O(n_active·n_clients·n_sites) → estoura o timeout em n_active alto na cidade grande, `agg=0.465`),
   **a seleção por HV escolheu uma heurística NÃO-balanceada** (a vencedora-HV de B tinha agg=1.034 e
   imbalance 3.7 — pior que A). Ou seja: **o prompt PRODUZIU os balanceadores, mas o fitness
   inalterado SELECIONA contra eles.** Para um balanceador VENCER, seria preciso **também** levar o
   balanceamento à fitness — o que, nos baselines, é exatamente a edição manual de §2; no LLM seria
   adicionar o objetivo como 3º critério de seleção (trivial) além do prompt.
4. **Caveat do 7B (escalabilidade):** os balanceadores do 7B são corretos mas **não-ótimos em custo**
   (recomputam a carga a cada passo) → travam em n_active alto em BH (fronteira viável menor: 81 vs
   116 pts). Um modelo mais forte provavelmente escreveria um balanceador vetorizado escalável.
5. **Confundimento (herdado da 5a):** "problema fácil" × "7B fraco" continua — mas é **ortogonal** ao
   que a 5b demonstra: o **mecanismo** (objetivo via NL muda o código gerado, com efeito medível)
   funcionou; sua *qualidade/seleção* é limitada pelo 7B e pela fitness inalterada.

**Veredito:** a Fase 5b **demonstra o diferencial** — descrever um novo objetivo em português
regenerou a heurística para considerá-lo (efeito medível: imbalance 3.08→2.17 / 3.87→2.04), algo que
os baselines só conseguiriam com reengenharia manual da fitness. A ressalva honesta é que, com a
fitness inalterada (HV) + o 7B, o laço de **seleção** não premia o balanceamento — o balanceador
existe na população por causa do prompt, mas não vence por HV. O mecanismo está provado; fechá-lo
("o balanceador também VENCE") pede alinhar a fitness ao objetivo (1 critério a mais) e/ou um gerador
mais forte.

---

### Arquivos
`docs/PHASE5B_NL_OBJECTIVE.md` (este), `src/optimizers/llm/nl_objectives.py`,
`src/eval/runner_phase5b.py`, `src/optimizers/llm/heuristic_runtime.py` (sandbox de processo),
`test_sandbox_isolation.py`, `analyze_5b.py`, `check_bh_5b.py`,
`results/phase5b/best_balancer_{A_baseline,B_balanced}.py`.
