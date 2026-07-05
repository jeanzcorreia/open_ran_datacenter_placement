# Fase 5a — ReEvo multi-cidade + generalização (treino 6 / teste held-out 4)

**Versão:** v2 (rodada LOCAL) · **Data da rodada:** 2026-06-26 (17:20→21:30, ~4.2 h, 4 seeds)
**Gerador LLM:** `qwen2.5-coder:7b` (Q4_K_M) **local**, endpoint OpenAI-compat `http://localhost:11434/v1`
(Ollama + Vulkan na AMD RX 5700 XT, 100% GPU). Geração e reflexão no mesmo modelo. Sem Groq/Gemini.

> Esta é a **rodada local** (4 seeds, sem quota). Substitui a v1 (2026-06-24, Gemini/Groq, 1 seed,
> bloqueada por quota). É um resultado **honesto e majoritariamente negativo** sobre o *laço evolutivo*
> com um 7B local: o pipeline e a generalização estão OK, mas a evolução não agregou e a seleção
> escolheu uma vencedora não-escalável. As seções 4–7 são as conclusões que importam.

---

## 1. Setup

- **Split (trava dura §10):** treino = {Manaus, Natal, Belo Horizonte, Goiânia, João Pessoa, Campo
  Grande}; teste held-out = {Curitiba, Recife, Florianópolis, Vitória}. As 4 de teste **nunca** entram
  no fitness/seleção — verificado nesta sessão: `solve_multi` recebe só as 6 de treino, e a trava
  anti-vazamento **lança** se um nome de teste chegar ao treino (`RuntimeError: VAZAMENTO DE TESTE`).
- **ReEvo:** pop=8, gerações=6, elite=2, 4 seeds [1,2,3,4]; 1 seed forte injetada (construção
  vetorizada por distância **mínima** — cobertura). Fitness = HV médio sobre as 6 cidades de treino +
  piso de robustez (crash/inviável numa cidade ⇒ HV daquela cidade = 0).
- **Custo LLM:** 128 chamadas de rede (resto = cache), in≈195.7k / out≈42.6k tokens, **US$ 0** (local).
  offspring avaliados por seed = [48, 48, 48, 48] — o laço evolutivo rodou por inteiro (não foi inerte).
- **Otimizações desta rodada:** few-shot vetorizado + temperatura 0.4 na geração; sandbox com
  **early-exit** no 1º crash (corta desperdício); timeout de chamada ao Ollama = 60 s; checkpoint por
  seed (resumível); sandbox cross-platform (watchdog de thread no Windows — ver §6).

| Cidade | grupo | sites | clientes |
|---|---|---:|---:|
| Manaus | treino | 1009 | 1172 |
| Natal | treino | 453 | 627 |
| Belo Horizonte | treino | 1111 | 1524 |
| Goiânia | treino | 722 | 1198 |
| João Pessoa | treino | 414 | 615 |
| Campo Grande | treino | 431 | 490 |
| Curitiba | teste | 975 | 1362 |
| Recife | teste | 709 | 920 |
| Florianópolis | teste | 350 | 448 |
| Vitória | teste | 270 | 429 |

---

## 2. Generalização — HV por grupo (treino 6 vs teste held-out 4)

HV normalizado por cidade (fronteira-referência = união dos viáveis não-dominados de todos os métodos
naquela cidade). **maximin = pior cidade do grupo.** Maior é melhor. `gap = treino − teste`.

| Método | HV treino (média) | HV treino (maximin) | HV teste (média) | HV teste (maximin) | gap |
|---|---:|---:|---:|---:|---:|
| **greedy** | **1.078** | **1.041** | **1.107** | **1.097** | −0.029 |
| zero_shot | 0.891 | 0.589 | 0.971 | 0.934 | −0.079 |
| **reevo** | 0.879 | 0.589 | 0.937 | 0.858 | −0.058 |
| nsga2 | 0.867 | 0.814 | 0.929 | 0.865 | −0.062 |
| nsga3 | 0.821 | 0.770 | 0.872 | 0.817 | −0.051 |
| moead | 0.810 | 0.762 | 0.872 | 0.828 | −0.061 |
| random | 0.684 | 0.656 | 0.723 | 0.693 | −0.039 |

**Gap negativo em todos os métodos** (teste > treino): as 4 cidades de teste são, em média, menores/
"mais fáceis" que as 6 de treino — **não há overfitting** do ReEvo (que seria gap positivo grande). O
gap do reevo (−0.058) é da ordem dos baselines ⇒ a heurística treinada **generaliza tão bem quanto os
EAs**. Mas isso vale pouco quando o laço não agrega (§5).

### IGD+ por grupo (média; menor é melhor)

| Método | IGD+ treino | IGD+ teste |
|---|---:|---:|
| greedy | 0.000 | 0.000 |
| nsga2 | 0.072 | 0.051 |
| zero_shot | 0.097 | 0.083 |
| nsga3 | 0.089 | 0.066 |
| **reevo** | 0.105 | 0.097 |
| moead | 0.132 | 0.105 |
| random | 0.145 | 0.120 |

> Nuance: em **HV** o reevo bate nsga2/nsga3/moead; em **IGD+** ele **perde para o nsga2** (0.105 vs
> 0.072 no treino) — a fronteira da heurística cobre hipervolume mas fica mais distante da referência
> que a do NSGA-II. `greedy` domina ambas as métricas (fronteira densa completa) e é o teto prático.

---

## 3. HV por cidade (10 cidades, média sobre 4 seeds)

| Cidade | grupo | reevo | zero_shot | greedy | nsga2 | nsga3 | moead | random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Manaus | treino | 0.936 | 0.935 | 1.069 | 0.814 | 0.770 | 0.762 | 0.656 |
| Natal | treino | 0.990 | 0.990 | 1.092 | 0.915 | 0.863 | 0.862 | 0.708 |
| Belo Horizonte | treino | 0.794 | 0.869 | 1.093 | 0.816 | 0.778 | 0.794 | 0.670 |
| Goiânia | treino | 0.948 | 0.948 | 1.075 | 0.840 | 0.805 | 0.793 | 0.670 |
| João Pessoa | treino | 1.017 | 1.017 | 1.099 | 0.929 | 0.880 | 0.880 | 0.724 |
| Campo Grande | treino | **0.589** | 0.589 | 1.041 | 0.889 | 0.831 | 0.769 | 0.675 |
| Curitiba | teste | 0.858 | 0.993 | 1.126 | 0.866 | 0.824 | 0.840 | 0.703 |
| Recife | teste | 0.987 | 0.987 | 1.098 | 0.865 | 0.817 | 0.827 | 0.693 |
| Florianópolis | teste | 0.967 | 0.967 | 1.108 | 0.982 | 0.918 | 0.906 | 0.744 |
| Vitória | teste | 0.934 | 0.934 | 1.097 | 1.003 | 0.928 | 0.913 | 0.752 |

- **Campo Grande (0.589)** é a pior cidade do reevo e fixa o maximin de treino — **não por crash**
  (lá houve 0 crash), mas porque a vencedora produz fronteira fraca nessa cidade.
- Em **Belo Horizonte (0.794)** e **Curitiba (0.858)** o reevo cai **abaixo do zero_shot** (0.869 /
  0.993) — efeito direto da vencedora-representativa que **trava** nessas cidades (§4).
- Em **Vitória** o nsga2 (1.003) supera o reevo (0.934).

---

## 4. A vencedora roda viável nas 10 cidades? **NÃO** ❌

`all_winner_feasible_no_crash = False`. Aplicando as 4 vencedoras-por-seed às 10 cidades (sweep denso
de 200 pts, timeout 2.5 s):

| Cidade | grupo | Σ n_crash (4 seeds) | min \|viável\| | status | culpada |
|---|---|---:|---:|---|---|
| Manaus | treino | 103 | 89 | **FALHA** | seed1 (origem=LLM) |
| Belo Horizonte | treino | 197 | 3 | **FALHA** | seed1 |
| Goiânia | treino | 78 | 117 | **FALHA** | seed1 |
| Curitiba | teste | 196 | 4 | **FALHA** | seed1 |
| Natal · João Pessoa · Campo Grande · Recife · Florianópolis · Vitória | — | 0 | ≥76 | OK | — |

**Só a vencedora do seed 1 trava — e exatamente nas 4 maiores cidades** (≥975 sites / ≥1172 clientes).
As vencedoras dos seeds 2, 3, 4 rodam viáveis e sem crash nas 10. O problema é o código da
vencedora-**representativa** (seed 1, `best_seed`, maior HV de treino), que tem um **laço Python sobre
clientes**:

```python
def place_odcs(instance, n_active):
    ...
    demand_per_site = np.zeros(n_sites)
    for c in range(n_clients):            # <-- NÃO-VETORIZADO: O(n_clients) em Python puro
        nearest_idx = D[c].argmin()
        demand_per_site[nearest_idx] += instance.client_demand[c]
    ...
```

**Por que passou no treino e travou na aplicação?** No treino (sweep coarse de 8 pts, timeout 0.4 s)
foi marcada `robusta=True` (0 crashes nas 6 cidades). Na aplicação (200 pts) estoura o timeout **só nas
cidades grandes**. O padrão (trava apenas onde `n_clients` é grande) é consistente com a **contenção de
GIL por threads-zumbi do sandbox** (§6): ao longo dos 4 seeds, heurísticas que travaram deixaram
threads-daemon vivas (o watchdog de thread no Windows **não as mata**), e na fase de aplicação a thread
principal já estava muito mais lenta — o laço-Python "borderline" da vencedora passou a estourar 2.5 s
nas cidades grandes. Isso também explica a rodada ter levado **4.2 h** em vez das ~2.5 h estimadas
(lentidão progressiva).

---

## 5. Proveniência por seed — a evolução agregou sobre a seed forte? **NÃO**

| seed | origem da vencedora | HV treino (média) | maximin | robusta (treino) | offspring | curva agg (6 gerações) |
|---|---|---:|---:|---|---:|---|
| 1 | **seed** (init LLM) | 0.907 | 0.790 | True | 48 | **plana** 0.907 → 0.907 |
| 2 | seed_strong | 0.834 | 0.354 | True | 48 | **plana** 0.834 → 0.834 |
| 3 | seed_strong | 0.834 | 0.354 | True | 48 | **plana** 0.834 → 0.834 |
| 4 | **seed** (init LLM) | 0.834 | 0.354 | True | 48 | **plana** 0.834 → 0.834 |

Proveniência agregada: **{seed: 2, seed_strong: 2}**. Duas conclusões duras:

1. **O laço evolutivo (crossover/mutação/reflexão) não melhorou nada.** Em **todos** os 4 seeds a
   `agg_curve` é **plana nas 6 gerações** e o `origin_curve` é constante — o melhor da população inicial
   permaneceu o melhor do início ao fim. **Nenhuma vencedora veio de `crossover` ou `mutate`.** Os 48
   offspring/seed foram avaliados, mas **nenhum superou a inicial**.
2. **O valor do LLM, quando existiu, veio da população INICIAL**, não da evolução: em 2 de 4 seeds uma
   heurística *gerada na init* (origem=`seed`) superou a seed forte injetada; nos outros 2, a seed forte
   venceu. O 7B às vezes produz uma boa heurística "de primeira", mas seus operadores de edição
   (crossover/mutate guiados por reflexão) **não destilam melhoria**.

Coerente com **zero_shot ≳ reevo** (§2): o zero-shot é, por definição, "o melhor da mesma população
inicial sem evoluir". Se o laço não agrega, o zero-shot iguala (ou supera — sua vencedora-representativa
nesta rodada não era a que trava).

---

## 6. A reflexão reparou algum bug? Sem evidência. + problema técnico de fundo

A reflexão **rodou** (48 chamadas `reflect_short`/`reflect_long`, texto coerente). Mas como as curvas de
convergência são planas em todos os seeds, **não há nenhuma evidência de que a reflexão (ou o crossover/
mutate guiados por ela) tenha consertado um bug ou melhorado uma heurística** nesta rodada. O bug de
escalabilidade da vencedora do seed 1 (laço Python) **não foi reparado** — sobreviveu intacto da init
até o fim e só foi exposto no benchmark denso.

### Problema técnico acionável: vazamento de threads-zumbi no sandbox (Windows)
O sandbox usa **watchdog de thread** (Windows não tem `SIGALRM`). Em timeout, a thread da heurística
**não é morta** — vira daemon e segue consumindo CPU/GIL. Heurísticas com laço infinito/lento deixam
threads-zumbi que **degradam progressivamente** toda a rodada (medido nesta sessão: 10 zumbis
`while True` levaram uma heurística válida de ~ms para **97 s**). O early-exit reduz a taxa de vazamento
~9×, mas **não elimina**. Isso (a) torna o piso de robustez **não-determinístico** (acoplado à carga de
CPU do momento) e (b) é a causa provável do §4. **Recomendação para uma rodada limpa: trocar o watchdog
de thread por isolamento de PROCESSO** (subprocesso morto no timeout) — heurísticas travadas são de fato
encerradas e os zumbis não acumulam.

---

## 7. CAVEAT central — confundimento "problema fácil" × "modelo fraco"

**Não dá para separar, nesta rodada, duas hipóteses:**

- **(A) O problema é fácil / o `greedy` é quase-ótimo.** O greedy domina HV e IGD+ em todas as 10
  cidades (fronteira densa completa); a seed forte vetorizada já é competitiva. O espaço de melhoria
  sobre uma boa construção gulosa pode ser pequeno — nesse caso *nenhum* gerador agregaria muito.
- **(B) O `qwen2.5-coder:7b` é fraco demais.** Empiricamente: parsing 100% OK, mas o 7B (i) produz
  muita heurística que quebra/escala mal (laços Python) e (ii) seus operadores de evolução não destilam
  melhoria (curvas planas). Um modelo mais forte (a Fase 5a foi desenhada p/ `gemini-2.5-flash`) poderia
  gerar heurísticas vetorizadas válidas **e** evoluí-las.

Os dados desta rodada são **consistentes com ambas** e **não as distinguem**. O que se afirma com
segurança: **com o 7B local, o laço evolutivo do ReEvo não superou a melhor heurística inicial em nenhum
seed**, e a seleção por HV-médio-de-treino chegou a **escolher uma vencedora não-escalável** (seed 1) que
falha no benchmark — enquanto a seed forte (seeds 2,3) seria a escolha robusta.

> **Para desconfundir:** repetir (i) com **isolamento de processo** no sandbox (remove o ruído de
> zumbis e o piso de robustez não-determinístico) e (ii) trocando **só o gerador** por um modelo forte
> (`gemini-2.5-flash`, ou local ≥14B se a VRAM permitir). Se com modelo forte o laço passar a agregar,
> era (B); se continuar plano, era (A) — o problema é fácil.

---

## 8. Conclusões

1. **Pipeline e integridade: OK.** Split 6/4 respeitado, trava anti-vazamento ativa, 4 seeds com
   checkpoint, generalização sem overfitting (gap ~ baselines).
2. **HV:** `greedy` > `zero_shot` ≈ `reevo` > `nsga2` > `nsga3` ≈ `moead` > `random`. Em **IGD+**,
   `nsga2` supera o reevo. O reevo (7B) **não bate o greedy** e **empata com sua própria ablação
   zero-shot** ⇒ **o laço evolutivo não agregou**.
3. **Robustez da vencedora: FALHA** — a representativa (seed 1, LLM) trava nas 4 maiores cidades; a
   escolha robusta seria a seed forte (seeds 2,3).
4. **Causa técnica:** vazamento de threads-zumbi no sandbox (Windows) degrada a rodada e mascara o piso
   de robustez ⇒ **trocar por isolamento de processo** antes de qualquer conclusão de qualidade.
5. **Confundimento não resolvido** (problema fácil × 7B fraco): documentado; desconfunde-se trocando o
   gerador por um modelo forte **e** isolando o sandbox.

**Próximos passos recomendados (antes de qualquer conclusão de qualidade):**
- Trocar o sandbox para isolamento de processo (mata heurística travada; remove o não-determinismo).
- Re-rodar trocando só o gerador por um modelo forte; comparar curva de convergência (plana vs agrega).
- Endurecer a seleção: além de robusta-no-treino, exigir robustez no **sweep denso** (não só coarse) —
  evitaria eleger uma vencedora que passa no treino coarse e trava no benchmark.

**Arquivos:** `results/phase5a/report_data.json` (dados completos),
`results/phase5a/winning_heuristic_multicity.py` (vencedora representativa — **não-escalável; não usar
em produção**), `results/phase5a/checkpoints/seed{1..4}.json`, `results/phase5a/winner_seed{1..4}.py`.

## 9. Reprodutibilidade

```bash
# rodada local destacada (resumível por checkpoint), via wrapper run_5a.ps1 (Agendador) ou direto:
python -m src.eval.runner_phase5 --config configs/phase5a.yaml --out results/phase5a --seeds 1,2,3,4
# smoke do pipeline: python -m src.eval.runner_phase5 --smoke --out results/phase5a_runner_smoke
```

- **Backend LLM:** Ollama (Vulkan) `http://localhost:11434/v1`, `qwen2.5-coder:7b`; `OLLAMA_MODELS=
  C:\ollama_models` (path ASCII), `OLLAMA_VULKAN=1`, keep-alive 24 h. Chave dummy `configs/local.key`.
- **Config:** `pop_size=8, generations=6, elite=2, seeds=[1,2,3,4]`; baselines `pop=300, n_gen=60,
  5 seeds`; sweep aplicação 200 pts; laço: timeout 0.4 s, sweep 8 pts, early-exit; `max_tokens=1500`,
  `request_timeout=60`, `max_retries=1`. pymoo 0.6.1.3, numpy 2.0.0 (wheels cp310 win_amd64).
- **Determinismo:** heurísticas sempre em sandbox; variância entre seeds por cache-namespace + temp 0.4.
  Ressalva: o piso de robustez é **não-determinístico** enquanto o sandbox usar watchdog de thread (§6).
