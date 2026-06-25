"""
src/problem/instance.py — Carrega uma cidade (CSV processado da Anatel) numa `Instance`.

Dois construtores de candidatos a ODC:
  - `load_instance`        : candidatos = centróides KMeans(random_state=0) com k clusters
                             (MODO REPRODUÇÃO da Fase 2 — espelha o parser original).
  - `load_instance_sites`  : candidatos = TODOS os sites únicos da cidade (dedupe por
                             cell_site_id, preservando lat/lon) — MODO JUSTO da Fase 3.

Em AMBOS, os CLIENTES são todas as linhas do CSV (SEM dedupe; oru_id = index+1) e a carga
de CPU por cliente é idêntica ao parser: cpu_cores = ceil((bandwidth_mhz/100)·cpu_per_100mhz),
com bandwidth_mhz da designação de emissão ITU. A diferença entre os modos é APENAS o
conjunto de candidatos (e, no modo justo, a formulação de objetivos/restrições em
odc_problem.py).

Encoding: lê com encoding='latin1'. Nos processados as colunas usadas são ASCII ⇒ latin1 e
utf-8 são byte-idênticos (ver docs/PHASE2_REPRO.md).
"""

from __future__ import annotations

import locale
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

locale.setlocale(locale.LC_NUMERIC, "C")

_ITU_UNIT_MHZ = {"H": 1e-6, "K": 1e-3, "M": 1.0, "G": 1e3}


def extract_bandwidth(designation: str) -> float:
    """Largura de banda em MHz da designação ITU (ex.: '100MG7W' -> 100.0). Igual ao original."""
    numeric_part = None
    unit = None
    for i, char in enumerate(designation):
        if char in _ITU_UNIT_MHZ:
            numeric_part = designation[:i]
            unit = char
            break
    if unit is None:
        raise ValueError(f"Designação de emissão sem unidade ITU reconhecível: {designation!r}")
    return float(numeric_part) * _ITU_UNIT_MHZ[unit]


def calculate_cpu_cores(bandwidth_mhz: float, cpu_per_100mhz: float) -> float:
    """(bandwidth_mhz / 100) * cpu_per_100mhz — igual ao original (sem ceil aqui)."""
    return (bandwidth_mhz / 100.0) * cpu_per_100mhz


def haversine_np(lat1, lon1, lat2, lon2):
    """Distância Haversine em km (R = 6371). Igual a `haversine_np` do original."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def generate_initial_odcs(client_coords: np.ndarray, num_initial_odcs: int) -> list[tuple[float, float]]:
    """Centróides KMeans(random_state=0) como candidatos (modo reprodução). Replica
    `generate_initial_odcs`, incluindo o refit em ConvergenceWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        kmeans = KMeans(n_clusters=num_initial_odcs, random_state=0).fit(client_coords)
        initial_odcs = kmeans.cluster_centers_
        if w and issubclass(w[-1].category, ConvergenceWarning):
            numbers = re.findall(r"\d+", str(w[-1].message))
            if len(numbers) >= 2:
                distinct_clusters = int(numbers[0])
                kmeans = KMeans(n_clusters=distinct_clusters, random_state=0).fit(client_coords)
                initial_odcs = kmeans.cluster_centers_
    return [(float(lat), float(lon)) for lat, lon in initial_odcs]


def precompute_distances(client_coords: np.ndarray, initial_odcs: list[tuple[float, float]]) -> np.ndarray:
    """Matriz Haversine (n_clients, n_odcs). Replica `precompute_distances`."""
    odc_coords = np.asarray(initial_odcs, dtype=float)
    n_clients = client_coords.shape[0]
    n_odcs = odc_coords.shape[0]
    distances = np.zeros((n_clients, n_odcs))
    for i in range(n_clients):
        distances[i, :] = haversine_np(
            np.full(n_odcs, client_coords[i, 0]),
            np.full(n_odcs, client_coords[i, 1]),
            odc_coords[:, 0],
            odc_coords[:, 1],
        )
    return distances


@dataclass
class Instance:
    """Uma instância (uma cidade + um conjunto de candidatos a ODC).

    `initial_odcs` é o conjunto de candidatos (centróides KMeans no modo reprodução, ou
    sites únicos no modo justo); `n_var = len(initial_odcs)`. Clientes são sempre todas as
    linhas (sem dedupe)."""

    name: str
    clients: list[dict]
    client_coords: np.ndarray
    cpu_cores: np.ndarray
    initial_odcs: list[tuple[float, float]]
    distances: np.ndarray
    cpu_per_100mhz: float
    requested_k: int
    csv_path: str = ""
    n_unique_sites: int = field(default=0)
    candidate_kind: str = "kmeans"  # "kmeans" (repro) | "sites" (fair)

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    @property
    def n_var(self) -> int:
        return len(self.initial_odcs)

    @property
    def total_cpu_demand(self) -> int:
        return int(self.cpu_cores.sum())

    def summary(self) -> dict:
        return {
            "name": self.name,
            "n_clients": self.n_clients,
            "n_unique_sites": self.n_unique_sites,
            "candidate_kind": self.candidate_kind,
            "n_odc_candidates": self.n_var,
            "cpu_per_100mhz": self.cpu_per_100mhz,
            "total_cpu_demand": self.total_cpu_demand,
        }


def _build_clients(df: pd.DataFrame, cpu_per_100mhz: float):
    """Constrói a lista de clientes (1 por linha, sem dedupe) + arrays de coords e cpu_cores.
    Idêntico ao laço de `read_clients` do original."""
    clients: list[dict] = []
    cpu_list: list[int] = []
    for index, row in df.iterrows():
        bandwidth_mhz = extract_bandwidth(row["emission_designation"])
        cpu_cores = math.ceil(calculate_cpu_cores(bandwidth_mhz, cpu_per_100mhz))
        clients.append(
            {
                "cell_site_id": row["cell_site_id"],
                "emission_designation": row["emission_designation"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "cell_carrier_id": row.get("cell_carrier_id"),
                "bandwidth_mhz": bandwidth_mhz,
                "cpu_cores": cpu_cores,
                "oru_id": index + 1,
            }
        )
        cpu_list.append(cpu_cores)
    client_coords = np.array([[c["latitude"], c["longitude"]] for c in clients], dtype=float)
    cpu_cores_arr = np.array(cpu_list, dtype=float)
    return clients, client_coords, cpu_cores_arr


def _read_csv(csv_path: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(
        csv_path,
        encoding=encoding,
        converters={"latitude": locale.atof, "longitude": locale.atof},
    )


def load_instance(
    csv_path: str,
    k: int,
    cpu_per_100mhz: float = 14.0,
    encoding: str = "latin1",
    name: Optional[str] = None,
) -> Instance:
    """MODO REPRODUÇÃO: candidatos = centróides KMeans. `k` == --odcs (0 => k = n_clients)."""
    df = _read_csv(csv_path, encoding)
    clients, client_coords, cpu_cores_arr = _build_clients(df, cpu_per_100mhz)

    num_initial_odcs = len(clients) if k == 0 else k
    initial_odcs = generate_initial_odcs(client_coords, num_initial_odcs)
    distances = precompute_distances(client_coords, initial_odcs)
    n_unique_sites = int(pd.Series([c["cell_site_id"] for c in clients]).nunique())

    if name is None:
        name = f"{os.path.splitext(os.path.basename(csv_path))[0]}_k{k}"

    return Instance(
        name=name,
        clients=clients,
        client_coords=client_coords,
        cpu_cores=cpu_cores_arr,
        initial_odcs=initial_odcs,
        distances=distances,
        cpu_per_100mhz=cpu_per_100mhz,
        requested_k=k,
        csv_path=csv_path,
        n_unique_sites=n_unique_sites,
        candidate_kind="kmeans",
    )


def load_instance_sites(
    csv_path: str,
    cpu_per_100mhz: float = 14.0,
    encoding: str = "latin1",
    name: Optional[str] = None,
) -> Instance:
    """MODO JUSTO: candidatos = TODOS os sites únicos (dedupe por cell_site_id, lat/lon do
    primeiro registro). `n_var = nº de sites únicos` (Natal = 55). Clientes continuam sem
    dedupe (todas as linhas)."""
    df = _read_csv(csv_path, encoding)
    clients, client_coords, cpu_cores_arr = _build_clients(df, cpu_per_100mhz)

    sites_df = df.drop_duplicates(subset="cell_site_id", keep="first")
    initial_odcs = [
        (float(r["latitude"]), float(r["longitude"])) for _, r in sites_df.iterrows()
    ]
    distances = precompute_distances(client_coords, initial_odcs)
    n_unique_sites = len(initial_odcs)

    if name is None:
        name = f"{os.path.splitext(os.path.basename(csv_path))[0]}_sites"

    return Instance(
        name=name,
        clients=clients,
        client_coords=client_coords,
        cpu_cores=cpu_cores_arr,
        initial_odcs=initial_odcs,
        distances=distances,
        cpu_per_100mhz=cpu_per_100mhz,
        requested_k=n_unique_sites,
        csv_path=csv_path,
        n_unique_sites=n_unique_sites,
        candidate_kind="sites",
    )
