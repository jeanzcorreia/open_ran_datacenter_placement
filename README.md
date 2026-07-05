# LLM-Guided Heuristic Design for Open RAN Data Center Placement

Substituição do NSGA-II por um projetista de heurísticas guiado por LLM (estilo ReEvo) para o problema de posicionamento de Open RAN Data Centers (ODCs), avaliado sobre dados reais da Anatel (10 cidades brasileiras).

Trabalho da disciplina Aprendizado de Máquina Supervisionado (Prof. Dr. Alexandre da Silva Simões), Programa de Pós-Graduação em Engenharia Elétrica — UNESP.

## Resumo

O posicionamento de ODCs em redes Open RAN envolve um trade-off multiobjetivo entre custo (número de data centers) e latência (distância média de fronthaul), sujeito a restrições de capacidade e distância. Partindo de um trabalho que resolve o problema com o algoritmo genético NSGA-II, este projeto substitui os operadores evolutivos fixos por um modelo de linguagem que projeta heurísticas automaticamente: o LLM gera código de heurística, mede o resultado, reflete sobre ele e itera.

Principais resultados (honestos):

* A heurística projetada pelo LLM é competitiva em hypervolume e generaliza para cidades não vistas, sem overfitting.
* Ela não supera uma construção gulosa (greedy k-median), que é o teto prático neste problema.
* O laço reflexivo não agregou sobre a melhor heurística inicial (com um modelo local de 7B) — documentamos o confound "problema fácil × modelo fraco".
* Diferencial: um objetivo novo (balanceamento de carga) pode ser adicionado descrevendo-o em linguagem natural no prompt, sem reescrever a função de fitness — reduzindo o desbalanceamento em até 47%. Métodos clássicos exigiriam reengenharia manual.

O artigo completo (formato IEEE) e os relatórios detalhados de cada fase estão em `docs/`.

## Estrutura do repositório

```
src/                      Código-fonte
  problem/                Formulação do problema (FairODCProblem, objetivos, restrições)
  optimizers/
    fair.py               Baselines: greedy (k-median), random
    llm/                  Projetista guiado por LLM (ReEvo), sandbox, objetivos em linguagem natural
  eval/                   Runners das fases (treino multi-cidade, generalização, 5b)
  data_prep/              Adaptador do dump SMP da Anatel -> instâncias por cidade
configs/                  Configurações de experimento (phase5a.yaml, experiment.yaml)
data/
  processed/              10 instâncias por cidade (CSV) — versionadas
  raw/                    Dump bruto da Anatel — NÃO versionado (ver "Dados")
docs/                     Artigo (IEEE) e relatórios por fase
  PHASE5A_GENERALIZATION.md   Generalização treino(6)/teste(4), métricas, análise
  PHASE5B_NL_OBJECTIVE.md     Objetivo por linguagem natural (o diferencial)
Results/                  Saídas das rodadas (fronteiras .npz, métricas, checkpoints)
```

## Método (visão geral)

1. Formulação justa. Minimizar (f1) número de ODCs e (f2) distância média de fronthaul, sob restrições de capacidade e distância máxima. A saída é uma fronteira de Pareto.
2. Heurística como código. Cada candidato é uma função `place_odcs(instance, n_active)`; varrer `n_active` traça a fronteira. O fitness é o hypervolume médio sobre as cidades de treino.
3. Laço reflexivo (ReEvo). O LLM gera, avalia, reflete (verbaliza por que uma heurística é melhor) e produz novas heurísticas por crossover/mutação, ao longo de 6 gerações.
4. Sandbox de processo. Cada heurística roda em subprocesso com timeout (morto se travar) → piso de robustez determinístico.
5. Treino/teste com trava anti-vazamento. Treino em 6 cidades, teste em 4 held-out; o código lança erro se uma cidade de teste alcançar o treino.
6. Objetivo em linguagem natural (Fase 5b). Um novo objetivo é descrito em português no prompt; o LLM regenera a heurística para considerá-lo, sem alterar a função de fitness.

Baselines: NSGA-II, NSGA-III, MOEA/D (via [pymoo](https://pymoo.org/)), greedy (k-median) e random. Métricas: hypervolume, IGD+, spacing, spread, e desbalanceamento de carga (máx/média, Jain).

## Como rodar

Requisitos: Python 3.10+, `pip install -r requirements.txt`.

O gerador de heurísticas usa um LLM local via endpoint compatível com a API OpenAI (ex.: [Ollama](https://ollama.com/) com `qwen2.5-coder:7b`), configurado em `configs/phase5a.yaml`. A chave é lida de arquivo (`api_key_file`) ou de variável de ambiente (`api_key_env`) — nunca embutida no código. Para um endpoint local, uma chave dummy basta.

```bash
# Rodada principal (treino em 6 cidades, teste em 4 held-out, 4 seeds)
python -m src.eval.runner_phase5 --config configs/phase5a.yaml --out Results/phase5a --seeds 1,2,3,4

# Fase 5b — objetivo de balanceamento por linguagem natural
python -m src.eval.runner_phase5b --config configs/phase5a.yaml --out Results/phase5b \
       --cities Natal,BeloHorizonte --seeds 1,2 --pop 8 --gen 6
```

## Dados

As instâncias derivam do cadastro público de Estações do Serviço Móvel Pessoal (SMP) da Anatel, filtrando portadoras 5G (NR). O dump bruto não é versionado (`data/raw/`, ignorado); as 10 instâncias já processadas estão em `data/processed/`. As cidades cobrem cinco macrorregiões, de 270 a 1111 sites por cidade.

## Autoria

Jean Zanella Correia — orientação: Prof. Dr. Alexandre da Silva Simões — UNESP, 2026.
