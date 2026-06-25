# Fase 5a — ReEvo MULTI-CIDADE + benchmark 10 cidades + generalização (treino vs held-out)

**Fase:** 5a. **Data:** 2026-06-24.
**Escopo:** treinar UMA heurística-código (`place_odcs`) estilo ReEvo **só nas 6 cidades de treino**,
com fitness = **HV médio multi-cidade** + **penalidade de robustez** (crash/inviável em qualquer
cidade de treino ⇒ não pode vencer); aplicar a vencedora (sem re-evoluir) às **10 cidades**;
benchmarkar contra os baselines por cidade; medir **generalização nas 4 cidades held-out**. Modo
justo, sem cap. **SEM** balanceamento de carga (isso é a Fase 5b). Código original não tocado.

## TL;DR honesto (leia isto)

1. **Objetivo de segurança ATINGIDO.** A vencedora multi-cidade roda **viável e SEM crash nas 10
   cidades** (`n_crash=0` em todas) — o **bug da Fase 4 está corrigido** (a transferência
   Natal→Manaus quebrava com `UnboundLocalError`; aqui nada quebra).
2. **Generalização SEM desabar: SIM.** Nas 4 cidades held-out o ReEvo **não desaba** (HV de teste
   **0.330 ≥** HV de treino **0.310**; gap treino−teste = **−0.021**, comparável ao do greedy
   −0.017). A queda da Fase 4 (Manaus HV 0.45) **não se repete**.
3. **Competitivo? NÃO nesta rodada.** A vencedora é a **mais fraca** em HV (≈0.31), **abaixo até do
   random** (≈0.68); o **greedy domina** (≈1.08–1.10). `reevo == zero_shot` (idênticos) ⇒ **a
   evolução/reflexão não agregou NADA** — curva de HV plana, vencedora de origem=`seed`.
4. **Por quê (3 causas honestas, com evidência):**
   (a) o **filtro de robustez é só-crash, não de qualidade**: das 8 heurísticas iniciais, só **1**
   passou (as outras 7 estouravam timeout/inviabilidade em alguma cidade grande), então a única
   sobrevivente venceu **sem competição**;
   (b) essa sobrevivente é **silenciosamente degenerada** — o "greedy" dela é **código morto** (o
   backtrack de viabilidade esvazia a lista a cada passo), colapsando para "os `n_active` sites de
   **menor distância MÉDIA** aos clientes" (= sites centrais agrupados = construção fraca);
   (c) a **evolução não pôde consertá-la** porque **as duas quotas DIÁRIAS gratuitas de LLM já
   estavam esgotadas hoje** (Groq 100k tokens/dia **e** Gemini 2.5-flash **20 req/dia**, ambas
   consumidas pela Fase 4 no mesmo dia) ⇒ gerações 2–6 não geraram **nenhuma** offspring nova.
5. **A abordagem PODE ser competitiva — a SELEÇÃO desta rodada é que não foi.** Uma construção
   **limpa por menor distância MÍNIMA** (cobertura, vetorizada, rápida, robusta) — o tipo de
   heurística que a busca deveria ter selecionado — dá **HV 0.95–0.99** (vs greedy ~1.09 e EAs
   ~0.82–1.00). Ou seja: existe uma heurística rápida+robusta+competitiva ao alcance da busca; a
   boa construção, porém, vinha **acoplada ao bug da Fase 4** e foi (corretamente) filtrada,
   deixando a alternativa robusta-porém-fraca. Isto é **exatamente a lição da Fase 4** se
   materializando (lá: forte-mas-bugada; aqui: robusta-mas-fraca).

> **Veredito:** Fase 5a cumpriu o critério **"generaliza sem desabar" + "vencedora não quebra em
> lugar nenhum"**. **Não** cumpriu **"competitivo"** — por esgotamento de quota (mesmo dia da
> Fase 4) somado a um filtro de robustez que pune crash mas não degenerescência. Recomendações de
> correção em §10.

---

## 1. Split treino/teste e INTEGRIDADE (trava dura §10)

| Grupo | Cidades | sites (candidatos) | clientes |
|---|---|---|---|
| **TREINO (6)** | Manaus, Natal, Belo Horizonte, Goiânia, João Pessoa, Campo Grande | 1009 / 453 / 1111 / 722 / 414 / 431 | 1172 / 627 / 1524 / 1198 / 615 / 490 |
| **TESTE held-out (4)** | Curitiba, Recife, Florianópolis, Vitória | 975 / 709 / 350 / 270 | 1362 / 920 / 448 / 429 |

**🔒 Integridade (CLAUDE.md §10):** as 4 cidades de teste **nunca** entraram no fitness/seleção.
Garantias em camadas: (i) `solve_multi` recebe **só** as 6 instâncias de treino; (ii) as instâncias
de teste só são **carregadas** na fase de benchmark (depois da seleção); (iii) asserção
anti-vazamento em `reevo_multicity.py` (`TEST_CITY_TOKENS`) aborta se um nome de teste chegar ao
treino; (iv) uma **revisão adversarial multi-agente** dedicada a vazamento varreu todos os caminhos
de dados e **não achou nenhum** (0 findings na dimensão integridade). A fronteira-referência por
cidade e as métricas de teste são avaliação FINAL, não sinal de treino.

## 2. ReEvo multi-cidade — desenho

- **Fitness** = **HV MÉDIO** da heurística sobre as **6 cidades de treino** (`instance_hv`
  normalizado por cidade, sweep coarse de 12 pontos no laço). Reporta também o **maximin** (HV da
  pior cidade de treino).
- **Penalidade de robustez:** se a heurística **lança exceção/timeout** (`n_crash>0`) **OU** produz
  **fronteira inviável** (nenhum ponto viável ⇒ score 0) em **qualquer** cidade de treino, recebe
  score **negativo** (`agg = −n_falhas + 1e-3·meanHV`) e **não pode vencer** uma robusta
  (`agg = meanHV ≥ 0`). Endereça o bug da Fase 4 e **descarta heurísticas que não escalam** (laços
  python estouram o timeout nas cidades grandes BH/Manaus).
- **INVARIANTE de custo (confirmada):** os operadores LLM (`_gen/_crossover/_mutate/_reflect_*`)
  atuam só sobre **código**; a avaliação multi-cidade é **100% CPU** (sandbox, 0 API). Logo o nº de
  chamadas de API **não cresce** com o nº de cidades — depende só de `pop/gen` (~68/seed nominal:
  8 init + 6×(4 cross + 4 mut) + 6×2 reflexão). Verificado: trocar 1→6 cidades não muda a contagem.
- **Roteamento/cache/budget:** inalterados da Fase 4 (Groq geração / Gemini reflexão; cache de disco
  compartilhado `results/phase4/llm_cache`).

## 3. Heurística vencedora (código) e o diagnóstico da degenerescência

`origem=seed` (heurística inicial; a evolução não a superou), **robusta** (n_crash=0 nas 6),
`train_meanHV=0.4271`, `train_maximin=0.0525`. Salva em
`results/phase5a/winning_heuristic_multicity.py`. Núcleo:

```python
def place_odcs(instance, n_active):
    active_sites = []
    while len(active_sites) < n_active:
        if not active_sites:
            mean_distances = np.mean(instance.distances, axis=0)   # distância MÉDIA a TODOS os clientes
            new_site = np.argmin(mean_distances)                   # site mais "central"
            active_sites.append(new_site)
        else:
            # greedy k-median (reduz a distância média) ... O(n_sites) por passo
            ...
            active_sites.append(np.argmax(reductions))
        # checagem de viabilidade APÓS cada adição:
        load = ...; nearest_dist = ...
        if load.max() > instance.max_capacity or nearest_dist.max() > instance.max_distance:
            active_sites.pop()                 # <-- com 1 ODC a carga é SEMPRE > 1000 -> remove
            if not active_sites: break         # <-- esvazia e SAI no 1º passo, SEMPRE
    if len(active_sites) < n_active:           # completa com os de menor distância MÉDIA
        mean_distances = np.mean(instance.distances, axis=0)
        remaining = sorted([i for i in range(instance.n_sites) if i not in active_sites],
                           key=lambda x: mean_distances[x])
        active_sites += remaining[:n_active - len(active_sites)]
    return active_sites
```

**Degenerescência (verificada):** em TODAS as 10 cidades a demanda total (4 482–17 525 cores) faz
**1 ODC** sempre exceder a capacidade de 1 000 ⇒ a checagem de viabilidade falha logo após o
**primeiro** site, `pop()` esvazia a lista e `break` sai. **O greedy k-median (o `else`) nunca
executa** — é **código morto**. A heurística colapsa para o fallback: **"os `n_active` sites de
menor distância MÉDIA aos clientes"** = sites centrais agrupados. Rápida e robusta (sem crash), mas
**construção fraca** (centralidade ≠ cobertura).

**Evidência (HV de benchmark, mesma cidade, 3 construções):**

| cidade | vencedora (dist. **média**, selecionada) | construção limpa (dist. **mínima**) | greedy |
|---|---:|---:|---:|
| Natal | **0.461** | **0.982** | 1.092 |
| Belo Horizonte | **0.343** | **0.989** | 1.093 |
| Vitória | **0.380** | **0.954** | 1.097 |

A construção **limpa por menor distância MÍNIMA** (`argsort(distances.min(axis=0))[:n_active]` —
cobre quem está perto de ALGUM cliente, vetorizada, robusta) seria **competitiva** (0.95–0.99). A
busca tinha essa boa construção ao alcance, mas ela vinha **acoplada ao bug de robustez da Fase 4**
(swap com `UnboundLocalError`) e foi filtrada; sobrou a variante por distância **média**, fraca.

## 4. HV de treino (interno, sweep coarse de 12 pts) por cidade

| cidade de treino | HV interno | n_valid | crash |
|---|---:|---:|---:|
| João Pessoa | 0.857 | 12 | 0 |
| Natal | 0.840 | 13 | 0 |
| Belo Horizonte | 0.424 | 13 | 0 |
| Goiânia | 0.335 | 13 | 0 |
| Manaus | 0.055 | 13 | 0 |
| Campo Grande | 0.052 | 13 | 0 |
| **média (fitness)** | **0.427** | | |
| **maximin (pior)** | **0.053** (Campo Grande) | | |

(HV interno baixo é em parte artefato do sweep coarse de 12 pts + fronteiras **comprimidas por
capacidade** nas cidades grandes — Manaus/Campo Grande precisam de muitos ODCs para serem viáveis,
encolhendo a região viável do Pareto.)

## 5. Benchmark por cidade — TREINO (6) (modo justo, sem cap; HV/IGD+ ; |front|)

EAs = média sobre 5 seeds; greedy/reevo/zero_shot determinísticos (1). Fronteira-referência por
cidade = união dos viáveis não-dominados de **todos** os métodos naquela cidade.

| Cidade (sites, ref) | método | HV ↑ | IGD+ ↓ | \|front\| |
|---|---|---:|---:|---:|
| **Manaus** (1009, 942) | greedy | **1.069** | 0.000 | 942 |
| | nsga2 | 0.814 | 0.095 | 72 |
| | nsga3 / moead / random | 0.770 / 0.762 / 0.656 | 0.114 / 0.155 / 0.163 | 33 / 15 / 20 |
| | **reevo = zero_shot** | **0.104** | 0.508 | 4 |
| **Natal** (453, 435) | greedy | **1.092** | 0.000 | 435 |
| | nsga2 | 0.916 | 0.053 | 71 |
| | nsga3 / moead / random | 0.863 / 0.862 / 0.708 | 0.069 / 0.103 / 0.126 | 39 / 17 / 27 |
| | **reevo = zero_shot** | **0.461** | 0.334 | 111 |
| **Belo Horizonte** (1111, 1066) | greedy | **1.093** | 0.000 | 1066 |
| | nsga2 | 0.816 | 0.087 | 70 |
| | nsga3 / moead / random | 0.778 / 0.794 / 0.670 | 0.104 / 0.126 / 0.148 | 33 / 12 / 26 |
| | **reevo = zero_shot** | **0.343** | 0.411 | 71 |
| **Goiânia** (722, 679) | greedy | **1.075** | 0.000 | 679 |
| | nsga2 | 0.840 | 0.082 | 70 |
| | nsga3 / moead / random | 0.805 / 0.793 / 0.670 | 0.097 / 0.141 / 0.153 | 34 / 17 / 22 |
| | **reevo = zero_shot** | **0.334** | 0.403 | 60 |
| **João Pessoa** (414, 399) | greedy | **1.099** | 0.000 | 399 |
| | nsga2 | 0.929 | 0.047 | 77 |
| | nsga3 / moead / random | 0.880 / 0.880 / 0.724 | 0.062 / 0.095 / 0.118 | 34 / 18 / 26 |
| | **reevo = zero_shot** | **0.537** | 0.329 | 179 |
| **Campo Grande** (431, 391) | greedy | **1.041** | 0.000 | 391 |
| | nsga2 | 0.889 | 0.068 | 76 |
| | nsga3 / moead / random | 0.831 / 0.769 / 0.675 | 0.090 / 0.172 / 0.160 | 37 / 21 / 23 |
| | **reevo = zero_shot** | **0.079** | 0.528 | 2 |

## 6. Benchmark por cidade — TESTE held-out (4)

| Cidade (sites, ref) | método | HV ↑ | IGD+ ↓ | \|front\| |
|---|---|---:|---:|---:|
| **Curitiba** (975, 896) | greedy | **1.077** | 0.000 | 896 |
| | nsga2 | 0.830 | 0.098 | 71 |
| | nsga3 / moead / random | 0.788 / 0.756 / 0.668 | 0.114 / 0.165 / 0.163 | 31 / 12 / 24 |
| | **reevo = zero_shot** | **0.306** | 0.443 | 66 |
| **Recife** (709, 677) | greedy | **1.098** | 0.000 | 677 |
| | nsga2 | 0.865 | 0.070 | 75 |
| | nsga3 / moead / random | 0.817 / 0.827 / 0.693 | 0.085 / 0.117 / 0.134 | 33 / 12 / 22 |
| | **reevo = zero_shot** | **0.411** | 0.385 | 92 |
| **Florianópolis** (350, 333) | greedy | **1.108** | 0.000 | 333 |
| | nsga2 | 0.982 | 0.037 | 74 |
| | nsga3 / moead / random | 0.918 / 0.906 / 0.744 | 0.053 / 0.100 / 0.113 | 31 / 23 / 21 |
| | **reevo = zero_shot** | **0.225** | 0.434 | 29 |
| **Vitória** (270, 260) | greedy | **1.097** | 0.000 | 260 |
| | nsga2 | 1.003 | 0.030 | 71 |
| | nsga3 / moead / random | 0.928 / 0.913 / 0.752 | 0.046 / 0.103 / 0.109 | 35 / 18 / 20 |
| | **reevo = zero_shot** | **0.380** | 0.395 | 166 |

## 7. Generalização — resumo (média de HV por grupo) e a pergunta central

| método | HV treino (média) | HV treino (maximin) | HV teste (média) | HV teste (maximin) | gap (treino−teste) |
|---|---:|---:|---:|---:|---:|
| greedy | 1.078 | 1.041 | 1.095 | 1.077 | −0.017 |
| nsga2 | 0.867 | 0.814 | 0.920 | 0.830 | −0.053 |
| nsga3 | 0.821 | 0.770 | 0.863 | 0.788 | −0.041 |
| moead | 0.810 | 0.762 | 0.850 | 0.756 | −0.040 |
| random | 0.684 | 0.656 | 0.714 | 0.668 | −0.030 |
| **reevo = zero_shot** | **0.310** | **0.079** | **0.330** | **0.225** | **−0.021** |

**Pergunta central — "nas 4 held-out o ReEvo fica COMPETITIVO (não desaba como Manaus na Fase 4)?"**
- **Não desaba: SIM.** HV de teste (0.330) **≥** treino (0.310); o **gap (−0.021)** é o **2º menor**
  (só perde p/ greedy −0.017) e o **maximin de teste (0.225) é MAIOR que o de treino (0.079)**. Ou
  seja, a vencedora multi-cidade é **estável** entre treino e teste — **sem overfitting e sem
  colapso** (contraste direto com a Fase 4, onde a vencedora de Natal desabava para HV 0.45 em
  Manaus). **Esse era o objetivo da Fase 5a, e foi cumprido.**
- **Competitiva: NÃO.** Em valor absoluto o ReEvo é o **pior** método (HV ~0.31), abaixo do random.
  Generaliza de forma **consistente**, porém **consistentemente fraca**.

**IGD+ por grupo:** reevo 0.43 (treino) / 0.42 (teste); greedy 0.00/0.00; nsga2 0.075/0.060. O IGD+
alto do ReEvo confirma fronteiras longe da referência (dominada por greedy).

## 8. Verificação ANTI-BUG — a vencedora roda viável e SEM crash nas 10? **SIM ✔**

| Cidade | grupo | n_crash | \|viável\| | OK? |
|---|---|---:|---:|:--:|
| Manaus | treino | 0 | 4 | ✔ |
| Natal | treino | 0 | 111 | ✔ |
| Belo Horizonte | treino | 0 | 71 | ✔ |
| Goiânia | treino | 0 | 60 | ✔ |
| João Pessoa | treino | 0 | 179 | ✔ |
| Campo Grande | treino | 0 | 2 | ✔ |
| Curitiba | teste | 0 | 66 | ✔ |
| Recife | teste | 0 | 92 | ✔ |
| Florianópolis | teste | 0 | 29 | ✔ |
| Vitória | teste | 0 | 166 | ✔ |

**`n_crash=0` em TODAS as 10 — o bug de robustez da Fase 4 está corrigido.** (Manaus=4 e Campo
Grande=2 pontos viáveis: poucos, por compressão de capacidade — essas cidades precisam de muitos
ODCs para viabilizar, e a construção central da vencedora cobre mal; mas ≥1 viável e zero crash.)

## 9. Orçamento LLM e a saga de quota (honesto)

- **Esta rodada: 0 chamadas de REDE, US$ 0.00.** TODAS as chamadas necessárias foram **acertos de
  cache** (a população inicial — mesmos `IDEA_HINTS` e prompt da Fase 4 — veio do cache de disco
  `results/phase4/llm_cache`). `offspring avaliados = 8` (todos da geração 1, de cache); gerações
  2–6 produziram **0 offspring novas**.
- **Por que 0 offspring novas:** **as duas quotas DIÁRIAS gratuitas estavam esgotadas hoje**
  (mesmo dia da Fase 4): **Groq `llama-3.3-70b` = 100 000 tokens/DIA** (já em ~98k) e **Gemini
  `2.5-flash` = 20 requisições/DIA** (a mesma trava que matou o flash-lite na Fase 4). Toda chamada
  **nova** de geração/crossover/mutação batia em **429 TPD** e toda reflexão batia em **429
  RESOURCE_EXHAUSTED**. Como o reset diário é em ~2h, configurei `max_retries: 0` p/ ambos
  (falha-rápido em quota morta) — o laço tolera (segue com cache / menos offspring), conforme
  projetado.
- **Custo nominal do MÉTODO (se rodado com quota fresca):** ~68 chamadas/seed (8 init + 48
  cross/mut + 12 reflexão), **independente do nº de cidades** (avaliação = CPU). Na Fase 4 isso
  custou ~US$0.06 (tier pago) / US$0 (free).
- **Invariante confirmada:** custo de API **não** cresceu com 6 cidades vs 1.

## 10. Leitura honesta e recomendações

**O que FUNCIONOU (objetivos da Fase 5a):**
1. **Robustez/sem-crash:** a vencedora roda viável em todas as 10 cidades — **o bug da Fase 4 sumiu**.
   A penalidade de robustez + treino multi-cidade fizeram o que prometiam contra *crashes*.
2. **Generalização sem desabar:** treino≈teste (gap −0.021, maximin de teste > treino) — **sem
   overfitting, sem colapso**. O modo de falha da Fase 4 (transferência que esfacela) **não ocorre**.
3. **Engenharia:** integridade treino/teste à prova de vazamento (revisão adversarial limpa);
   invariância de API; benchmark completo das 10 cidades; bug crítico interno corrigido antes de
   rodar (o shim de operadores sem `.origin` que tornaria a evolução **inerte de forma silenciosa**
   — pego por revisão adversarial e consertado).

**O que NÃO funcionou (e por quê, sem maquiagem):**
4. **Vencedora fraca (pior HV, abaixo do random).** Três causas, todas evidenciadas:
   - **Filtro de robustez é só-crash, não de qualidade.** Das 8 heurísticas iniciais, **só 1**
     sobreviveu (as outras 7 estouravam timeout/inviabilidade em cidade grande). A única robusta
     venceu **sem competição** — e era a fraca.
   - **Degenerescência silenciosa.** A vencedora tem o **greedy como código morto** (backtrack
     esvazia a lista) e colapsa para "sites de menor distância **média**" (centralidade), construção
     fraca. A penalidade pega **crash**, não **degenerescência**.
   - **Evolução quota-bloqueada.** Com as quotas diárias de Groq **e** Gemini esgotadas no mesmo dia
     da Fase 4, gerações 2–6 não geraram nada novo — a evolução **não pôde** mutar/consertar a
     heurística (curva de HV plana `0.4271`, origem sempre `seed`).
5. **Reflexão morta.** `reevo == zero_shot` exatamente. Confirma (ablação limpa, mesmo init) que **a
   evolução reflexiva não agregou** — combinação de problema fácil (Fase 4 já mostrava) **+** quota
   esgotada. Honestamente: nesta rodada o ReEvo é, na prática, "melhor-do-init", e o init só tinha 1
   robusta (fraca).

**A APROXIMAÇÃO pode ser competitiva — a SELEÇÃO desta rodada não foi.** A construção **limpa por
distância mínima** dá **HV 0.95–0.99** (§3), rápida e robusta. Ela existe no espaço da busca; só
não foi selecionada porque (a) a boa construção vinha acoplada ao bug da Fase 4 e foi filtrada, e
(b) a evolução que a "limparia" estava bloqueada por quota.

**Recomendações concretas (para uma re-execução / Fase 5b):**
- **Rodar com quota fresca** (dia diferente da Fase 4, ou tier pago) para que as gerações 2–6
  efetivamente gerem/selecionem heurísticas — provável vencedora robusta **e** competitiva.
- **Tornar a robustez ciente de QUALIDADE, não só de crash:** descartar também heurísticas cujo HV
  médio de treino fique **abaixo de um piso** (p.ex. do random), evitando "robusta porém degenerada".
- **Semear o init com uma construção vetorizada forte conhecida** (distância mínima) para garantir
  ≥1 sobrevivente robusta **e** competitiva, dando à evolução um ponto de partida bom.
- **Afrouxar levemente o timeout do laço OU detectar código morto** (a vencedora "tem" um greedy que
  nunca roda) — uma checagem de que o ramo principal executa ao menos uma vez evitaria a
  degenerescência silenciosa.

**Consistência com a tese (CLAUDE.md §1/§8c):** o problema 2-objetivo é **fácil para construção**
(greedy k-median ~1.08 domina; EAs ~0.82–1.00; random ~0.68) — coerente com as Fases 3–4. A
contribuição "projeto automático de heurística robusta que **generaliza**" foi **parcialmente**
demonstrada: a robustez/generalização-sem-colapso **sim**; a competitividade **não nesta rodada**
(quota + degenerescência). O diferencial central (objetivo por linguagem natural) é a **Fase 5b**.

## 11. Reprodutibilidade

```bash
python3 -m src.eval.runner_phase5 --config configs/phase5a.yaml --out results/phase5a
# smoke (sem API): python3 -m src.eval.runner_phase5 --smoke --backend offline --out results/phase5a_smoke
```

- **Novos artefatos:** `src/optimizers/llm/reevo_multicity.py` (MultiCityReEvoOptimizer +
  penalidade de robustez + MultiCityZeroShot + `apply_heuristic_to_city`); `src/eval/runner_phase5.py`
  (treino 6, benchmark 10 paralelo, fronteira-ref por cidade, tabelas treino×teste); `configs/phase5a.yaml`.
- **Resultados:** `results/phase5a/{report_data.json, winning_heuristic_multicity.py, full_run.log,
  <Cidade>/<método>/...}` (fronteiras persistidas).
- **Config desta rodada:** `pop_size=8, generations=6, elite=2, seed=1`; baselines `pop=300,
  n_gen=60, 5 seeds`; sweep de aplicação 200 pts; timeout do laço 0.6 s, 12 pts; `max_retries=0`
  (quota diária morta). pymoo 0.6.1.3, numpy 2.0.0. Heurística sempre em sandbox.
