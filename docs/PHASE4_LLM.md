# Fase 4 — Otimizador guiado por LLM (ReEvo) no modo justo

**Fase:** 4. **Data:** 2026-06-24.
**Escopo:** implementar o otimizador-LLM estilo ReEvo (gerar→avaliar→refletir→evoluir heurísticas de
código em sandbox) e avaliá-lo no MESMO `FairODCProblem` (modo justo da Fase 3) em Natal, com
ablação zero-shot e transferência Natal→Manaus. SEM extensão por linguagem natural e SEM 10 cidades (Fase 5).

**Provedor: Google Gemini + Groq** (não Anthropic) — ver §2. **Leitura honesta (resumo):**
- **Em Natal (in-distribution): SUCESSO.** A heurística evoluída é **competitiva com greedy/NSGA-II** (HV 1.0209 vs 1.0228/1.0244; **melhor IGD+ de todos, 0.0014**; fronteira de 51 pts).
- **A ablação confirma que a busca importa:** zero-shot HV **0.52** ≪ ReEvo **1.02**.
- **O ganho da EVOLUÇÃO foi limitado:** a vencedora foi uma heurística **inicial** (origem=seed); a curva de HV ficou plana (0.9506) nas 6 gerações — o problema é fácil (k-median, greedy≈ótimo) e o orçamento de LLM/reflexão foi restrito por quota.
- **Generalização Natal→Manaus: a vencedora FALHA (HV 0.45)** por um **bug de robustez** (erro em soluções iniciais inviáveis). PORÉM a busca PRODUZIU heurísticas que generalizam bem (uma heurística greedy "limpa" da mesma busca → **82 pts viáveis** em Manaus): a falha é da **seleção** (otimizou só o HV de Natal), não da abordagem.

---

## 1. Formulação e contrato (igual à Fase 3)

- Problema = **modo justo** (`FairODCProblem`): candidatos = todos os sites únicos; F=[nº ODCs, dist. média de fronthaul]; restrições contínuas (capacidade ≤1000 cores/ODC, distância ≤11 km/cliente); cenário cpuper100=14.
- A heurística gerada implementa `place_odcs(instance, n_active) -> índices dos n_active sites a ativar`. A **fronteira é varrida** chamando-a para `n_active = mínimo_viável .. n_sites`; o **score** é o **hypervolume** dessa fronteira (penalizado por inviabilidade). Todos os pontos são reavaliados no `FairODCProblem` VERDADEIRO (mesmo espaço dos baselines).
- **Sandbox** (`heuristic_runtime.py`): valida por AST (sem I/O/rede/eval; `import numpy/math` tolerados, resto rejeitado), executa com builtins restritos + numpy, timeout por chamada; qualquer falha ⇒ score ruim (a heurística é descartada).

## 2. Provedor LLM, modelos e a saga de quota (free tier)

O laço usa o **modelo mais barato** na geração e um **mais forte** na reflexão (split estilo ReEvo). No free tier:
- **gemini-2.5-pro** saiu do free tier (pago) → descartado.
- **gemini-2.5-flash-lite** (geração): o próprio 429 da API revelou **20 requisições/DIA** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit:20`) — inviável para o laço. **Decisão do arquiteto: mover a GERAÇÃO para o Groq.**
- **Geração/crossover/mutação → Groq `llama-3.3-70b-versatile`** (não há `qwen3-coder` no Groq). Limites reais (lidos dos headers `x-ratelimit-*`): **1000 req/dia, 12 000 TPM e 100 000 TOKENS/DIA**. O laço **esgotou os 100k tokens/dia** perto da geração 4 → as offspring seguintes deram 429 (sem efeito, pois a evolução já havia estabilizado).
- **Reflexão → Gemini `gemini-2.5-flash`** (GA estável). `gemini-3-flash-preview` foi tentado mas dava **503 (overloaded)** com frequência → trocado pelo GA + retry em 503.
- Cliente **provider-agnóstico** (`llm_client.py`): `OpenAICompatBackend` (Gemini/Groq via endpoint OpenAI-compatível) + `RoutedClient` (roteia por OP: geração→Groq, reflexão→Gemini). **Rate limiter por TOKENS** (token bucket vs TPM, sync com headers do Groq) + backoff+jitter honrando o retry do servidor; **cache de respostas em disco** (131 hits de cache evitaram re-chamadas); **timeout curto (40s)** para uma chamada travada falhar rápido (corrigiu um *hang* de rede que paralisou uma rodada anterior).

Config completa em `configs/experiment.yaml` (`pop_size=8, generations=6, elite=2`, seed 1).

## 3. Resultados — Natal (modo justo), HV/IGD+/spacing/spread + custo LLM

Fronteira de referência (união dos não-dominados viáveis de TODOS os métodos): 53 pts; `ideal=[3,0]`, `nadir=[55,2.746]`; HV ref point (1.1,1.1) no espaço normalizado (igual à Fase 3).

| Método | HV ↑ | IGD+ ↓ | Spacing ↓ | Spread ↓ | \|front\| | chamadas LLM | tokens (saída) | custo US$ | tempo |
|---|---|---|---|---|---|---|---|---|---|
| **nsga2** | 1.0244 | 0.0044 | 0.038 | 0.574 | 49.2 | — | — | — | 1.4 s |
| **greedy** | 1.0228 | 0.0062 | 0.019 | 0.453 | 52 | — | — | — | 0.01 s |
| **reevo (LLM)** | **1.0209** | **0.0014** | **0.013** | 0.442 | **51** | ~33 (Groq)+~12 (Gemini)¹ | ~26k | ~$0.06¹ | evolução ~min² |
| nsga3 | 1.0078 | 0.0096 | 0.040 | 0.656 | 31.2 | — | — | — | 1.4 s |
| moead | 0.9690 | 0.0456 | 0.057 | 0.743 | 22.6 | — | — | — | 7.5 s |
| **zero_shot (LLM)** | 0.5228 | 0.2923 | 0.011 | 0.579 | 43 | 4 (Groq) | ~0.3k | ~$0 | <1 s |
| random | 0.8677 | 0.0542 | 0.028 | 0.736 | 25.4 | — | — | — | 0.4 s |

¹ Orçamento LLM TOTAL da Fase 4 (todas as rodadas, ver §6): **33 chamadas Groq (~85k tokens) + ~12 chamadas Gemini de reflexão (~13k tokens)**, custo estimado ~US$0,06 (tier pago; no free tier = $0). A maior parte do laço foi servida do **cache** (131 hits).
² Tempo de relógio dominado por rate-limiting/backoff do free tier, não por computação (uma rodada gastou ~100 min quase toda dormindo em backoff após esgotar o TPD do Groq).

**Gráfico:** `results/phase4/fronts_natal.png` — a fronteira da ReEvo **sobrepõe** as de greedy e NSGA-II; a de zero-shot fica muito acima (dominada).

**Leitura.** A **ReEvo é competitiva com os melhores baselines** (HV 1.0209 ≈ greedy 1.0228 ≈ NSGA-II 1.0244) e tem o **melhor IGD+ (0.0014)** — converge para perto da fronteira de referência. A **ablação zero-shot (HV 0.52, IGD+ 0.29)** é muito pior: **sem a busca/seleção, a heurística que o LLM propõe de primeira é medíocre** — o valor está na busca + sandbox + seleção por HV, não numa única chamada.

**Ressalva sobre o "ganho da evolução".** A curva de convergência é **plana (HV interno 0.9506 nas 6 gerações)** e a vencedora tem **origem=seed** (uma das heurísticas iniciais). Ou seja, nenhuma offspring (crossover/mutação) superou a melhor heurística inicial. Causas: (i) o problema é **fácil** — uma boa construção gulosa já aproxima o ótimo por k (consistente com a Fase 3, onde o greedy quase iguala o NSGA-II); (ii) **reflexão e geração foram restritas por quota** (TPD do Groq esgotado; reflexão limitada). Honestamente: **nesta instância, a parte "evolução reflexiva" do ReEvo não agregou sobre uma boa população inicial** — o ganho mensurável veio de *gerar+avaliar+selecionar* várias heurísticas (vs zero-shot).

## 4. Transferência Natal→Manaus (a melhor heurística de Natal, SEM re-evoluir)

| Método | HV ↑ | IGD+ ↓ | \|front\| |
|---|---|---|---|
| nsga2 | 0.9909 | 0.0107 | 68.8 |
| greedy | 0.9758 | 0.0052 | 79 |
| nsga3 | 0.9578 | 0.0155 | 44.8 |
| moead | 0.8831 | 0.0908 | 24 |
| random | 0.7945 | 0.0808 | 24 |
| **reevo_transfer** | **0.4497** | 0.2341 | **28** |

**A vencedora NÃO generaliza** (HV 0.45, só 28 pts viáveis dos 87 do sweep). Diagnóstico: a heurística vencedora é uma **construção gulosa + busca local (swap)** com um **bug de robustez** — `best_mean_fronthaul` só é definido no ramo viável mas é referenciado incondicionalmente, gerando `UnboundLocalError` sempre que a solução inicial é **inviável**. Em Natal isso afeta só 2 de 53 valores de `n_active`; na instância maior/mais difícil de Manaus afeta **59 de 87** ⇒ fronteira esfacelada.

> **Nuance honesta (importante).** A **abordagem** gera heurísticas que generalizam: uma heurística **greedy "limpa" da MESMA busca**, aplicada a Manaus, produz **82 pts viáveis** (vs 28 da vencedora). A falha é da **SELEÇÃO** — escolher a vencedora só pelo HV de Natal premiou uma heurística **overfit e não-robusta**. A reflexão (que poderia ter detectado o bug) ficou limitada por quota. **Correção clara para a Fase 5:** selecionar por robustez/transferência (HV em instância(s) held-out), não só pelo HV de treino, e dar orçamento de reflexão suficiente para o LLM depurar a própria heurística.

## 5. Heurística vencedora (código)

Salva em `results/phase4/winning_heuristic.py` (origem=seed; HV interno 0.9506; competitiva em Natal, HV 1.0209). É uma construção gulosa por distância + refino por trocas locais (swap). **Contém o bug de robustez** descrito em §4 (`UnboundLocalError` em início inviável) — mantida como está para fidelidade ao resultado real. Trecho-chave:

```python
def place_odcs(instance, n_active):
    D = instance.distances
    n_active = max(1, min(n_active, instance.n_sites))
    distances_to_clients = D.min(axis=0)
    initial_sites = np.argsort(distances_to_clients)[:n_active]      # construção gulosa
    for _ in range(100):                                            # refino por swap
        selected = initial_sites.copy()
        sub = D[:, selected]; nearest = sub.min(axis=1)
        load = np.zeros(n_active); np.add.at(load, sub.argmin(axis=1), instance.client_demand)
        feasible = (load.max() <= instance.max_capacity) and (nearest.max() <= instance.max_distance)
        if feasible:
            best_mean_fronthaul = nearest.mean()                    # <-- só definido aqui
            ... (tenta swaps que reduzem a distância média) ...
        else:
            ... (tenta swaps que tornam viável) ...                 # <-- NÃO define best_mean_fronthaul
        if nearest.mean() == best_mean_fronthaul:                   # <-- UnboundLocalError se início inviável
            break
    return initial_sites.tolist()
```

## 6. Orçamento LLM consumido (todas as rodadas da Fase 4)

Agregado dos logs de chamada (`results/phase4/*calls*.jsonl`), apenas chamadas REAIS (não-cache):

| Provedor / modelo | papel | chamadas | tokens in | tokens out |
|---|---|---|---|---|
| Groq `llama-3.3-70b-versatile` | geração/crossover/mutação | 33 | 58 952 | 26 297 |
| Gemini `gemini-2.5-flash` | reflexão (final) | 7 | 11 447 | 1 382 |
| Gemini `gemini-3-flash-preview` | reflexão (descartado, 503) | 5 | 12 672 | 899 |
| Gemini `gemini-2.5-flash-lite` | geração (descartado, 20/dia) | 3 | 7 974 | 5 790 |
| **TOTAL real** | | **~48 úteis** | **~91k** | **~34k** |

Cache: **131 hits** (re-runs e replay determinístico do laço foram servidos do disco). Custo estimado ~US$0,06 (tier pago; **free tier = $0**). Limites do free tier que moldaram o experimento: Gemini flash-lite **20 req/dia**; Groq **1000 req/dia + 12k TPM + 100k tokens/dia**.

## 7. Arquivos criados (Fase 4)

```
src/optimizers/llm/{__init__,llm_client,heuristic_runtime,prompts,reevo,zero_shot,offline_heuristics}.py
src/eval/runner_llm.py
configs/experiment.yaml
results/phase4/{winning_heuristic.py, summary.json, Natal/{reevo,zero_shot}/, Manaus/reevo_transfer/, *calls*.jsonl, llm_cache/}
requirements.txt (+ pyyaml, openai)
```

## 8. Conformidade e honestidade

- Heurística gerada **sempre em sandbox** (§1); contrato `evaluate` da Fase 3 reutilizado sem alteração.
- Resultados reportados **sem maquiagem**: ReEvo empata os melhores baselines em Natal e tem o melhor IGD+, mas **(a)** a evolução reflexiva não superou a melhor heurística inicial nesta instância fácil, e **(b)** a heurística selecionada **não generaliza** (bug de robustez) — embora a abordagem produza heurísticas que generalizam, indicando que o critério de seleção deve mirar transferência.
- Caveats operacionais (free tier, troca de provedor de geração, hang de rede corrigido, reflexão 503→GA) documentados para reprodutibilidade.

## 9. Próximos passos (Fase 5)

1. **Selecionar por transferência/robustez**, não só HV de treino (held-out city no fitness) — endereça §4 diretamente.
2. Dar **orçamento de reflexão suficiente** (sem estrangulamento de quota) para o LLM depurar/robustecer as heurísticas.
3. Objetivos/restrições em **linguagem natural** (contribuição central da tese) sobre este mesmo arcabouço.
4. Escalar para **10 cidades** (Fase 5 Parte 1 — dados — já concluída) com split treino/teste para medir generalização de forma sistemática.
