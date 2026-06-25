# Fase 5 — Parte 1 (DADOS): expansão para 10 cidades a partir do dump nacional SMP

**Fase:** 5 (Parte 1 — apenas DADOS). **Data:** 2026-06-24.
**Escopo executado:** investigar o dump nacional "Estações do SMP" (Anatel), mapear para o
schema processado do projeto, filtrar 5G/NR nos 10 municípios-alvo, gerar os CSVs processados
por cidade (`data/processed/`), contar sites únicos e registrar em `data/cities.yaml`. **NÃO**
foi rodada otimização; **NÃO** foi tocado o código original; as âncoras single-operator
`CityData/Natal.csv` e `CityData/Manaus.csv` (Fases 2–4) permanecem **intactas**.

**Resultado em uma frase:** o dump foi mapeado 1:1 para o schema de `CityData/<Cidade>.csv`
(cabeçalho **byte-idêntico**), gerando **10 instâncias multi-operator 5G NR** (270–1111 sites
únicos, **nenhuma** trivial) que **carregam sem erro** por `src/problem/instance.py` nos dois
modos (KMeans e sites); a validação Natal/Manaus confirma schema e semântica, com contagens
**maiores** que as âncoras de 2024, como esperado.

---

## 0. Decisão de arquitetura desta parte — escopo de operadora

O dump nacional é **multi-operadora**; as âncoras das Fases 2–4 eram **single-operator**
(o `CityData/preprocess_anatel_city_csv.py` filtra `operator == 'TELEFONICA BRASIL S.A.'`).
A tarefa especifica apenas "filtrar 5G/NR + município", sendo silenciosa quanto à operadora.
**Decisão (arquiteto):** Fase 5 = **TODAS as operadoras** — cenário **Open RAN / O-Cloud
neutral-host** (infraestrutura de edge compartilhada servindo todas as RUs do município).
Justificativa: placement de ODC compartilhado é a premissa correta de Open RAN; single-operator
contradiz isso. As âncoras single-operator (Natal=55 / Manaus=90) seguem servindo a **reprodução**
das Fases 2–4; a Fase 5 é um **problema novo** (multi-operator), no qual se compara **LLM vs
baselines dentro de cada instância**, não contra o paper.

---

## 1. Dump: arquivo, encoding, delimitador

| Item | Valor |
|---|---|
| Arquivo (zip no repo) | `estacoes_smp.zip` (88 MB) → `data/raw/Estacoes_SMP.csv` (964 MB) |
| Dataset | "Estações do SMP" (Anatel), snapshot **2026-06-22** |
| Linhas | **3.226.356** (nacional; lido em chunks de 300k via pandas) |
| **Encoding** | **UTF-8 com BOM** → `encoding='utf-8-sig'`  *(NÃO latin-1; difere dos brutos antigos)* |
| **Delimitador** | **`;`** |
| Colunas | **41** (lista no §2) |

`data/raw/` foi adicionado ao `.gitignore` (arquivo grande, não versionar); o zip e
`Estacoes_SMP.csv` também. Os CSVs processados (`data/processed/`, 40–140 KB cada) **são**
versionados.

### Duas correções ao brief da tarefa (verificadas no código)
1. **Demanda/CPU vem de `Designação Emissão`** (designação ITU, ex. `100MG7W`), **não de
   `tx_frequency`**. `src/problem/instance.py:152` chama `extract_bandwidth(emission_designation)`;
   `tx_frequency`/`rx_frequency` são carregadas por linha mas **não entram na fitness**. A regra
   "linhas em nível de portadora, sem dedupe" continua válida — apenas a coluna que precisa estar
   limpa é `emission_designation`.
2. **Encoding é UTF-8 (com BOM)**, não latin-1.

---

## 2. Colunas do dump (41) e mapeamento para o schema processado

Colunas do dump (ordem original):
`Número Fistel; Número Estação; NumCnpjCpf; NumServico; Frequência (MHz); Banda_MHZ;
Frequência Inicial; Frequência Final; FreqTxMHz; FreqRxMHz; Designação Emissão; Número Ato;
Data Validade; Entidade; NumSetor; Tecnologia; Tipo de Tecnologia 5G; Latitude; Longitude;
Latitude decimal; Longitude decimal; EnderecoEstacao; EndBairro; EndNumero; EndComplemento;
Cep; ClassInfraFisica; Data Primeiro Licenciamento; Data Licenciamento; AnoMesLic; Situacao;
Caráter; Empresa Estação; Faixa Estação; Subfaixa Estação; Geração; Código Nacional;
Código IBGE; Município-UF; UF; Nome da UF`

**Mapeamento dump → schema processado** (`cell_site_id,…,operator`, idêntico a `Natal.csv`):

| coluna processada | coluna do dump | obrigatória? | observação |
|---|---|:--:|---|
| `cell_site_id` | **Número Estação** | ✅ | id da estação/site (único por operadora — ver §5) |
| `emission_designation` | **Designação Emissão** | ✅ | **driver de demanda/CPU** (largura ITU) |
| `latitude` | **Latitude decimal** | ✅ | decimal com ponto |
| `longitude` | **Longitude decimal** | ✅ | decimal com ponto |
| `technology` | Tecnologia | — | chave do filtro 5G (`NR`) |
| `tx_frequency` | FreqTxMHz | — | preservada por linha (não usada na fitness) |
| `rx_frequency` | FreqRxMHz | — | idem |
| `operator` | Entidade | — | canonizada p/ MAIÚSCULAS (funde dupla grafia da Telefónica) |
| `cell_carrier_id` | *sintetizada* | — | id único por linha `f"{ibge}{seq:06d}"` (opcional no loader) |
| `azimuth, antenna_gain, back_front_relation, hpa, mechanical_elevation, polarization, antenna_height, tx_power` | **ausentes no dump** → `NaN` | — | `instance.py` não as consome |

Apenas as **4 primeiras** são de fato obrigatórias (são as únicas lidas por
`_build_clients` em `src/problem/instance.py`). Nenhum campo obrigatório faltou.

**Adaptador:** `src/data_prep/smp_adapter.py` (novo; não reescreve o `preprocess` original).
Reusa `extract_bandwidth` do projeto para validar a designação ITU. Executável:
```bash
python3 -m src.data_prep.smp_adapter            # gera data/processed/*.csv + data/cities.yaml
```

---

## 3. Regra de filtro 5G

**`Tecnologia == "NR"`.** É exato e não-ambíguo: coincide **linha a linha** com `Geração == "5G"`
(ambos **232.365** linhas no nacional). Não foi necessária heurística de faixa (3300–3800 MHz).

Distribuição nacional de `Tecnologia`: LTE 1.600.167 · GSM 721.851 · WCDMA 662.613 ·
**NR 232.365** · CDMA 1.539 · EDGE 12 · (vazio) 7.809.

Linhas com `Designação Emissão` não-parseável por `extract_bandwidth`, coordenada ausente ou
`cell_site_id` vazio seriam descartadas (com log) — **na prática 0 descartes** nas 10 cidades.

---

## 4. Municípios-alvo: casamento e contagens

Casamento por **nome normalizado (sem acento) + UF**; o **IBGE** foi confirmado/reportado (todos
os 10 casaram com IBGE único e esperado).

| Cidade | UF | IBGE | linhas 5G | **sites únicos** | nº operadoras | arquivo |
|---|---|---|---:|---:|---:|---|
| Belo Horizonte | MG | 3106200 | 1524 | **1111** | 3 | `data/processed/BeloHorizonte.csv` |
| Manaus | AM | 1302603 | 1172 | **1009** | 3 | `data/processed/Manaus.csv` |
| Curitiba | PR | 4106902 | 1362 | **975** | 3 | `data/processed/Curitiba.csv` |
| Goiânia | GO | 5208707 | 1198 | **722** | 3 | `data/processed/Goiania.csv` |
| Recife | PE | 2611606 | 920 | **709** | 4 | `data/processed/Recife.csv` |
| Natal | RN | 2408102 | 627 | **453** | 4 | `data/processed/Natal.csv` |
| Campo Grande | MS | 5002704 | 490 | **431** | 3 | `data/processed/CampoGrande.csv` |
| João Pessoa | PB | 2507507 | 615 | **414** | 4 | `data/processed/JoaoPessoa.csv` |
| Florianópolis | SC | 4205407 | 448 | **350** | 4 | `data/processed/Florianopolis.csv` |
| Vitória | ES | 3205309 | 429 | **270** | 3 | `data/processed/Vitoria.csv` |

**Cidades sinalizadas (< ~25 sites): nenhuma.** A menor instância (Vitória, 270) está muito acima
do limiar de trivialidade.

> ⚠️ **Risco de runtime (baselines da Fase 5):** a maior instância é **Belo Horizonte (1111 sites
> candidatos no modo justo)**, seguida de Manaus (1009) e Curitiba (975). Se a rodada completa da
> Fase 5 ficar inviável em tempo, opções: capar candidatos (k do KMeans no modo reprodução), ou
> remover BH do conjunto de treino. **Sinalizado para o arquiteto.**

---

## 5. Integridade da chave de site (`cell_site_id`)

Caveat do arquiteto: confirmar que `cell_site_id` não **colide entre operadoras** (senão usar
coordenada+operadora como chave). **Verificado e resolvido:**

- O `site_integrity` do adaptador, após canonizar a operadora, reporta **0 sites multi-operadora**
  em **todas** as 10 cidades. As aparentes colisões iniciais (Goiânia 66, Vitória 62, BH 30, …)
  eram **100% artefato da dupla grafia da Telefónica** no dump (`TELEFONICA BRASIL S.A.` vs
  `Telefonica Brasil S.a.` = mesma entidade). Resolvido canonizando `operator` para MAIÚSCULAS
  (que também alinha com o formato das âncoras). **Conclusão: `Número Estação` é único por
  operadora e NÃO colide entre operadoras — a chave de site é segura; o fallback coord+operadora
  não foi necessário.**
- **Multi-coordenada:** poucos sites (0–17 por cidade) têm o mesmo `cell_site_id` com >1
  coordenada distinta, quase todos < 200 m de distância (re-survey da mesma estação, mesma
  operadora); máximos isolados em Goiânia (768 m) e Florianópolis (204 m). Inofensivo: clientes
  guardam lat/lon por linha (sem dedupe) e o modo "sites" deduplica por `cell_site_id` mantendo o
  primeiro registro.

Semântica: como cada operadora licencia sua própria estação, um mesmo poste físico com 3
operadoras gera **3 `cell_site_id` distintos** = 3 sites candidatos co-localizados. Isso é coerente
com o cenário neutral-host (a O-Cloud serve cada RU de cada operadora).

---

## 6. Validação Natal/Manaus (dump vs âncoras 2024)

O dump (22/06/2026) é mais recente que os arquivos antigos (~2024); espera-se schema/semântica
**iguais** e contagens **semelhantes ou maiores**. Comparação apples-to-apples = subconjunto
**Telefónica** do dump vs a âncora single-operator:

| | âncora 2024 (Telefónica) | dump 2026 Telefónica (apples-to-apples) | dump 2026 todas operadoras (Fase 5) |
|---|---:|---:|---:|
| **Natal** — linhas / sites | 170 / **55** | 169 / **129** | 627 / **453** |
| **Manaus** — linhas / sites | 288 / **90** | 477 / **397** | 1172 / **1009** |

- **Schema:** cabeçalho processado **byte-idêntico** ao de `CityData/Natal.csv` (17 colunas, mesma
  ordem/nome). Ambas as cidades **carregam por `instance.py`** (modos KMeans e sites) sem erro.
- **Contagens:** sites únicos cresceram (Natal 55→129, Manaus 90→397 no recorte Telefónica) —
  **maiores**, como esperado para a expansão do 5G em ~2 anos. ✔
- **Diferença semântica documentada (granularidade de linha):** o export de *licenciamento* de 2024
  trazia linhas por **portadora × azimute/setor** (≈3,09 linhas/site; até 3 azimutes/site); o dump
  de *Estações SMP* de 2026 traz linhas por **portadora/licença** (azimute ausente → ≈1,31
  linhas/site; setores aparecem como linhas de portadora duplicadas, mantidas sem dedupe). Por isso
  as **linhas** de Natal ficaram ~iguais (170 vs 169) enquanto os **sites** mais que dobraram. As
  **colunas têm o mesmo significado** e o modelo de demanda (designação ITU → largura → CPU) é
  idêntico; a granularidade por linha é **consistente entre as 10 instâncias** da Fase 5.

**Decisão (confirmada):** a Fase 5 usa as 10 cidades derivadas do **dump** (snapshot único e
consistente). As âncoras `CityData/Natal.csv` e `CityData/Manaus.csv` permanecem intactas para as
Fases 2–4.

---

## 7. Artefatos desta parte

| Artefato | Descrição |
|---|---|
| `src/data_prep/smp_adapter.py` | adaptador dump → CSVs processados + `cities.yaml` (novo) |
| `data/processed/<Cidade>.csv` | 10 CSVs processados, schema de `CityData/<Cidade>.csv` |
| `data/cities.yaml` | registro das 10 cidades (nome, UF, IBGE, linhas, sites, operadoras, csv, fonte) — **gerado** pelo adaptador |
| `data/raw/Estacoes_SMP.csv` | dump nacional extraído (**gitignored**) |
| `.gitignore` | + `data/raw/`, `*.zip`, `Estacoes_SMP.csv` |
| `docs/PHASE5_DATA.md` | este documento |

**Reprodução:** `python3 -m src.data_prep.smp_adapter` (requer `data/raw/Estacoes_SMP.csv`).
