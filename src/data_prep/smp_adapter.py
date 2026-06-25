"""
src/data_prep/smp_adapter.py — Fase 5 (DADOS): adaptador do dump nacional
"Estações do SMP" (Anatel) para o schema processado usado por src/problem/instance.py.

NÃO reescreve CityData/preprocess_anatel_city_csv.py (aquele opera sobre os exports de
licenciamento *single-operator* das Fases 2-4). Este adaptador lê o dump NACIONAL e
*multi-operator*, filtra 5G NR nos municípios-alvo da Fase 5, e gera um CSV por cidade em
data/processed/ no MESMO schema (mesmas colunas/ordem) de CityData/<Cidade>.csv.

FASE 5 É MULTI-OPERATOR (todas as prestadoras) — cenário Open RAN / O-Cloud neutral-host
(infra de edge compartilhada servindo todas as RUs do município). Difere das âncoras
single-operator (Telefónica) das Fases 2-4, que permanecem INTACTAS em CityData/Natal.csv e
CityData/Manaus.csv.

Fatos verificados do dump:
  - Encoding UTF-8 COM BOM  -> ler com encoding='utf-8-sig'.
  - Delimitador ';'.
  - ~3,23 milhões de linhas nacionais  -> leitura em chunks.
  - Filtro 5G: coluna 'Tecnologia' == 'NR' (idêntico a 'Geração' == '5G'; 232.365 linhas).
  - Demanda/CPU vem de 'Designação Emissão' (designação ITU, ex. '100MG7W'), NÃO de
    tx_frequency. Linhas com designação não-parseável por extract_bandwidth são DESCARTADAS
    (senão instance.py quebra ao carregar).

Uso:
    python3 -m src.data_prep.smp_adapter \
        --dump data/raw/Estacoes_SMP.csv --outdir data/processed
"""

from __future__ import annotations

import argparse
import os
import unicodedata
from typing import Optional

import numpy as np
import pandas as pd

# Reutiliza a MESMA física do projeto (mesma extração de banda da designação ITU).
from src.problem.instance import extract_bandwidth

# --------------------------------------------------------------------------- #
# Schema de saída (idêntico a CityData/<Cidade>.csv — ver preprocess original).
# --------------------------------------------------------------------------- #
PROC_COLS = [
    "cell_site_id", "emission_designation", "technology", "tx_frequency",
    "rx_frequency", "azimuth", "antenna_gain", "back_front_relation", "hpa",
    "mechanical_elevation", "polarization", "antenna_height", "tx_power",
    "latitude", "longitude", "cell_carrier_id", "operator",
]
# Colunas "nice-to-have" ausentes no dump -> NaN (instance.py não as consome).
NAN_COLS = [
    "azimuth", "antenna_gain", "back_front_relation", "hpa",
    "mechanical_elevation", "polarization", "antenna_height", "tx_power",
]

# Colunas lidas do dump (usecols mantém a leitura enxuta).
DUMP_COLS = [
    "Número Estação", "Designação Emissão", "Tecnologia", "FreqTxMHz",
    "FreqRxMHz", "Latitude decimal", "Longitude decimal", "Entidade",
    "Código IBGE", "Município-UF", "UF",
]

# --------------------------------------------------------------------------- #
# Municípios-alvo da Fase 5: (nome normalizado, UF) -> (rótulo, IBGE esperado, arquivo).
# O CASAMENTO é por nome normalizado (sem acento) + UF; o IBGE é confirmado/reportado.
# --------------------------------------------------------------------------- #
TARGETS = {
    ("manaus", "AM"):         {"label": "Manaus",         "ibge": 1302603, "file": "Manaus.csv"},
    ("natal", "RN"):          {"label": "Natal",          "ibge": 2408102, "file": "Natal.csv"},
    ("belo horizonte", "MG"): {"label": "Belo Horizonte", "ibge": 3106200, "file": "BeloHorizonte.csv"},
    ("curitiba", "PR"):       {"label": "Curitiba",       "ibge": 4106902, "file": "Curitiba.csv"},
    ("recife", "PE"):         {"label": "Recife",         "ibge": 2611606, "file": "Recife.csv"},
    ("goiania", "GO"):        {"label": "Goiânia",        "ibge": 5208707, "file": "Goiania.csv"},
    ("florianopolis", "SC"):  {"label": "Florianópolis",  "ibge": 4205407, "file": "Florianopolis.csv"},
    ("campo grande", "MS"):   {"label": "Campo Grande",   "ibge": 5002704, "file": "CampoGrande.csv"},
    ("joao pessoa", "PB"):    {"label": "João Pessoa",    "ibge": 2507507, "file": "JoaoPessoa.csv"},
    ("vitoria", "ES"):        {"label": "Vitória",        "ibge": 3205309, "file": "Vitoria.csv"},
}

TECH_5G = "NR"  # == Geração '5G'

# Snapshot do dump (data interna do arquivo / informada pelo arquiteto).
SNAPSHOT = "2026-06-22"


def normalize_name(s: str) -> str:
    """minúsculas, sem acento, espaços colapsados (para casar 'Goiânia' == 'goiania')."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.lower().split())


def _municipio_name(municipio_uf: str, uf: str) -> str:
    """Extrai o nome do município de 'Belo Horizonte - MG' removendo o sufixo ' - <UF>'."""
    txt = str(municipio_uf)
    suffix = f" - {uf}"
    if txt.endswith(suffix):
        txt = txt[: -len(suffix)]
    else:  # fallback robusto: corta no último ' - '
        txt = txt.rsplit(" - ", 1)[0]
    return txt


def _coerce_float(series: pd.Series) -> pd.Series:
    """Converte para float tolerando vírgula decimal (defensivo; o dump usa ponto)."""
    s = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _valid_designation(designation: str) -> bool:
    try:
        bw = extract_bandwidth(str(designation))
        return bw > 0
    except Exception:
        return False


def collect_target_rows(dump_path: str, chunksize: int = 300_000) -> pd.DataFrame:
    """Lê o dump em chunks, mantém só NR nas cidades-alvo, devolve um DataFrame único
    com as colunas do dump + chaves de casamento ('_key', '_ibge', '_muni')."""
    keep = []
    target_keys = set(TARGETS.keys())
    reader = pd.read_csv(
        dump_path, sep=";", encoding="utf-8-sig", usecols=DUMP_COLS,
        dtype=str, chunksize=chunksize, low_memory=False,
    )
    for chunk in reader:
        nr = chunk[chunk["Tecnologia"] == TECH_5G]
        if nr.empty:
            continue
        uf = nr["UF"].fillna("").str.strip()
        muni = [
            _municipio_name(mu, u) for mu, u in zip(nr["Município-UF"], uf)
        ]
        key = [(normalize_name(m), u) for m, u in zip(muni, uf)]
        mask = [k in target_keys for k in key]
        if not any(mask):
            continue
        sub = nr.loc[mask].copy()
        sub["_key"] = [k for k, m in zip(key, mask) if m]
        sub["_muni"] = [mu for mu, m in zip(muni, mask) if m]
        sub["_ibge"] = pd.to_numeric(sub["Código IBGE"], errors="coerce")
        keep.append(sub)
    if not keep:
        return pd.DataFrame(columns=DUMP_COLS + ["_key", "_muni", "_ibge"])
    return pd.concat(keep, ignore_index=True)


def to_processed(df_city: pd.DataFrame, ibge: int) -> tuple[pd.DataFrame, dict]:
    """Mapeia as linhas (1 por portadora, SEM dedupe) de uma cidade para o schema
    processado. Descarta linhas inválidas (designação não-parseável / coords ausentes)
    e devolve (df_saída, stats_descarte)."""
    n_in = len(df_city)

    lat = _coerce_float(df_city["Latitude decimal"])
    lon = _coerce_float(df_city["Longitude decimal"])
    site = df_city["Número Estação"].astype(str).str.strip()
    desig = df_city["Designação Emissão"].astype(str).str.strip()

    # Validação da designação ITU uma vez por valor único (barato).
    uniq = {d: _valid_designation(d) for d in desig.unique()}
    valid_desig = desig.map(uniq)

    valid = valid_desig & lat.notna() & lon.notna() & site.ne("") & site.ne("nan")
    n_drop_desig = int((~valid_desig).sum())
    n_drop_coord = int((valid_desig & ~(lat.notna() & lon.notna())).sum())
    n_drop_site = int((valid_desig & lat.notna() & lon.notna() & (site.eq("") | site.eq("nan"))).sum())

    d = df_city.loc[valid].copy()
    out = pd.DataFrame(index=d.index)
    out["cell_site_id"] = site.loc[valid].values
    out["emission_designation"] = desig.loc[valid].values
    out["technology"] = d["Tecnologia"].values
    out["tx_frequency"] = _coerce_float(d["FreqTxMHz"]).values
    out["rx_frequency"] = _coerce_float(d["FreqRxMHz"]).values
    for c in NAN_COLS:
        out[c] = np.nan
    out["latitude"] = lat.loc[valid].values
    out["longitude"] = lon.loc[valid].values
    # cell_carrier_id: id único por linha (portadora). Opcional no loader; sintetizado.
    out["cell_carrier_id"] = [f"{ibge}{i:06d}" for i in range(len(out))]
    # operator: canoniza p/ MAIÚSCULAS + espaços colapsados. Funde a dupla grafia da
    # Telefónica do dump ('TELEFONICA BRASIL S.A.' vs 'Telefonica Brasil S.a.' = mesma
    # entidade) e alinha com o formato das âncoras antigas. Operadoras distintas têm nomes
    # distintos independentemente de caixa -> não há fusão indevida.
    out["operator"] = (
        d["Entidade"].astype(str).str.upper().str.split().str.join(" ").values
    )

    out = out[PROC_COLS]
    stats = {
        "n_in": n_in,
        "n_out": len(out),
        "drop_designation": n_drop_desig,
        "drop_coord": n_drop_coord,
        "drop_site": n_drop_site,
    }
    return out, stats


def site_integrity(out: pd.DataFrame) -> dict:
    """Verifica se cell_site_id NÃO colide entre operadoras/coordenadas (caveat do arquiteto).
    Retorna nº de sites com >1 operadora ou >1 coordenada distinta."""
    g = out.groupby("cell_site_id")
    ops_per_site = g["operator"].nunique()
    coord = out.assign(
        _c=out["latitude"].round(5).astype(str) + "," + out["longitude"].round(5).astype(str)
    )
    coords_per_site = coord.groupby("cell_site_id")["_c"].nunique()
    return {
        "n_sites": int(out["cell_site_id"].nunique()),
        "sites_multi_operator": int((ops_per_site > 1).sum()),
        "sites_multi_coord": int((coords_per_site > 1).sum()),
    }


def build_all(dump_path: str, outdir: str, min_sites_warn: int = 25) -> list[dict]:
    os.makedirs(outdir, exist_ok=True)
    print(f"[smp_adapter] lendo dump: {dump_path}")
    allrows = collect_target_rows(dump_path)
    print(f"[smp_adapter] linhas NR nas cidades-alvo: {len(allrows):,}")

    results = []
    for (norm, uf), meta in TARGETS.items():
        sub = allrows[allrows["_key"].apply(lambda k: k == (norm, uf))]
        # IBGE(s) que casaram com este (nome, UF) — deve ser exatamente um.
        ibge_matched = sorted(set(int(x) for x in sub["_ibge"].dropna().unique()))
        muni_seen = sorted(set(sub["_muni"]))
        expected = meta["ibge"]
        if len(ibge_matched) != 1 or ibge_matched[0] != expected:
            print(f"  [AVISO] {meta['label']}/{uf}: IBGE casado={ibge_matched} "
                  f"esperado={expected} | municípios={muni_seen}")
        ibge = ibge_matched[0] if ibge_matched else expected

        out, st = to_processed(sub, ibge)
        integ = site_integrity(out)
        out_path = os.path.join(outdir, meta["file"])
        out.to_csv(out_path, index=False)

        rec = {
            "label": meta["label"], "uf": uf, "ibge": ibge,
            "muni_seen": muni_seen, "n_lines": st["n_out"],
            "n_sites": integ["n_sites"],
            "drop_designation": st["drop_designation"],
            "drop_coord": st["drop_coord"], "drop_site": st["drop_site"],
            "sites_multi_operator": integ["sites_multi_operator"],
            "sites_multi_coord": integ["sites_multi_coord"],
            "n_operators": int(out["operator"].nunique()),
            "csv_path": os.path.relpath(out_path),
            "trivial": integ["n_sites"] < min_sites_warn,
        }
        results.append(rec)
        flag = "  <<< TRIVIAL (<%d sites)" % min_sites_warn if rec["trivial"] else ""
        print(f"  {meta['label']:16s} IBGE={ibge} linhas={st['n_out']:>5} "
              f"sites={integ['n_sites']:>5} ops={rec['n_operators']:>2} "
              f"drop(desig/coord/site)={st['drop_designation']}/{st['drop_coord']}/{st['drop_site']} "
              f"colis(op/coord)={integ['sites_multi_operator']}/{integ['sites_multi_coord']}"
              f"{flag}")
    return results


def write_cities_yaml(results: list[dict], path: str, raw_file: str) -> None:
    """Emite data/cities.yaml a partir dos resultados (fonte única, sem números na mão)."""
    lines = [
        "# data/cities.yaml — Fase 5 (DADOS). GERADO por src/data_prep/smp_adapter.py.",
        "# Fonte: dump nacional \"Estações do SMP\" (Anatel).",
        "# Filtro 5G: Tecnologia == \"NR\" (idêntico a Geração == \"5G\").",
        "# Escopo: TODAS as operadoras (cenário Open RAN / O-Cloud neutral-host, multi-operator).",
        "# Linhas em nível de PORTADORA, SEM dedupe. cell_site_id = Número Estação (único por",
        "# operadora; não colide entre operadoras — verificado).",
        "# As âncoras single-operator CityData/{Natal,Manaus}.csv (Fases 2-4) permanecem intactas",
        "# e NÃO fazem parte deste registro.",
        "source:",
        f'  dataset: "Estações do SMP (Anatel)"',
        f'  snapshot: "{SNAPSHOT}"',
        f"  raw_file: {raw_file}",
        "  encoding: utf-8-sig",
        '  delimiter: ";"',
        '  filter_5g: '"'"'Tecnologia == "NR"'"'"'',
        "  operator_scope: all          # multi-operator",
        "  dedupe: false",
        "cities:",
    ]
    for r in results:
        flag = "    trivial: true   # < limiar de sites — instância possivelmente trivial" \
            if r["trivial"] else None
        lines += [
            f'  - name: {r["label"]}',
            f"    uf: {r['uf']}",
            f"    ibge: {r['ibge']}",
            f"    n_lines_5g: {r['n_lines']}",
            f"    n_sites: {r['n_sites']}",
            f"    n_operators: {r['n_operators']}",
            f"    csv: {r['csv_path']}",
            f'    source: "dump nacional SMP, {SNAPSHOT}"',
        ]
        if flag:
            lines.append(flag)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[smp_adapter] cities.yaml escrito: {path}")


def main(argv: Optional[list] = None):
    ap = argparse.ArgumentParser(description="Adaptador dump SMP -> CSVs processados (Fase 5).")
    ap.add_argument("--dump", default="data/raw/Estacoes_SMP.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--cities-yaml", default="data/cities.yaml")
    ap.add_argument("--min-sites-warn", type=int, default=25)
    args = ap.parse_args(argv)
    results = build_all(args.dump, args.outdir, args.min_sites_warn)
    if args.cities_yaml:
        write_cities_yaml(results, args.cities_yaml, args.dump)


if __name__ == "__main__":
    main()
