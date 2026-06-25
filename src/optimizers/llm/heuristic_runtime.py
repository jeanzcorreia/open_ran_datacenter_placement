"""
src/optimizers/llm/heuristic_runtime.py — Sandbox para a heurística gerada pelo LLM.

Contrato da heurística (string de código Python):
    def place_odcs(instance, n_active):
        # retorna os índices dos `n_active` sites a ativar (lista/array/set),
        # ou uma máscara booleana de tamanho instance.n_sites.

Segurança (CLAUDE.md §9.6): validação por AST (sem import / open / dunder / eval/exec/...),
execução com __builtins__ restrito + numpy, timeout (SIGALRM), e QUALQUER exceção/violação
=> SandboxError (o chamador atribui score ruim). Sem I/O, sem rede.
"""

from __future__ import annotations

import ast
import math
import signal
from dataclasses import dataclass

import numpy as np

# Builtins seguros disponíveis dentro da heurística.
_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in [
        "len", "range", "enumerate", "sorted", "min", "max", "abs", "sum", "list",
        "set", "dict", "tuple", "int", "float", "bool", "zip", "map", "filter",
        "round", "any", "all", "reversed", "True", "False", "None", "print",
    ]
}

_FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "exit", "quit", "help", "memoryview",
    "breakpoint",
}
_FORBIDDEN_MODULES_HINT = ("os", "sys", "subprocess", "socket", "shutil", "pathlib",
                           "requests", "urllib", "importlib", "pickle", "ctypes")


class SandboxError(Exception):
    pass


# Apenas numpy/math podem ser "importados" (a heurística costuma escrever `import numpy as np`);
# devolvemos os módulos já disponíveis. Qualquer outro import falha.
_ALLOWED_IMPORTS = {"numpy": np, "math": math}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root in _ALLOWED_IMPORTS:
        return _ALLOWED_IMPORTS[root]
    raise SandboxError(f"import proibido: {name}")


# Permite `import numpy as np` / `import math` / `from math import sqrt` dentro da heurística,
# resolvendo para os módulos já disponíveis (sem dar acesso a outros módulos).
_SAFE_BUILTINS["__import__"] = _safe_import


@dataclass
class HeuristicInstance:
    """Dados read-only expostos à heurística (derivados de uma Instance do modo justo)."""

    n_sites: int
    n_clients: int
    site_coords: np.ndarray          # (n_sites, 2) [lat, lon]
    client_coords: np.ndarray        # (n_clients, 2)
    client_demand: np.ndarray        # (n_clients,) cpu cores
    distances: np.ndarray            # (n_clients, n_sites) km (Haversine)
    max_distance: float
    max_capacity: float

    @property
    def demand_total(self) -> float:
        return float(self.client_demand.sum())

    @classmethod
    def from_instance(cls, inst, max_distance, max_capacity):
        return cls(
            n_sites=inst.n_var,
            n_clients=inst.n_clients,
            site_coords=np.asarray(inst.initial_odcs, dtype=float),
            client_coords=np.asarray(inst.client_coords, dtype=float),
            client_demand=np.asarray(inst.cpu_cores, dtype=float),
            distances=np.asarray(inst.distances, dtype=float),
            max_distance=float(max_distance),
            max_capacity=float(max_capacity),
        )


def _validate_ast(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"SyntaxError: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] not in _ALLOWED_IMPORTS:
                    raise SandboxError(f"import proibido: {n.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _ALLOWED_IMPORTS:
                raise SandboxError(f"import proibido: {node.module}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(f"acesso a atributo dunder proibido: {node.attr}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise SandboxError(f"nome proibido: {node.id}")
    if "place_odcs" not in {getattr(n, "name", None) for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef)}:
        raise SandboxError("função place_odcs não definida")


class _Timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def __enter__(self):
        def handler(signum, frame):
            raise SandboxError("timeout")
        self._old = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, *a):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old)


def run_heuristic(code: str, hinst: HeuristicInstance, n_active: int, timeout: float = 2.0) -> np.ndarray:
    """Executa place_odcs e retorna um array de índices de sites únicos e válidos.

    Lança SandboxError em qualquer falha (sintaxe, segurança, timeout, exceção, saída inválida).
    """
    _validate_ast(code)
    ns = {"np": np, "numpy": np, "math": math, "__builtins__": _SAFE_BUILTINS}
    try:
        with _Timeout(timeout):
            exec(compile(code, "<heuristic>", "exec"), ns)
            fn = ns.get("place_odcs")
            if not callable(fn):
                raise SandboxError("place_odcs não é chamável")
            out = fn(hinst, int(n_active))
    except SandboxError:
        raise
    except Exception as e:  # qualquer erro de runtime da heurística
        raise SandboxError(f"{type(e).__name__}: {e}")

    return _normalize_selection(out, hinst.n_sites)


def _normalize_selection(out, n_sites) -> np.ndarray:
    """Converte a saída (índices ou máscara) em índices únicos válidos [0, n_sites)."""
    arr = np.asarray(list(out)) if isinstance(out, (set, frozenset)) else np.asarray(out)
    arr = arr.ravel()
    if arr.dtype == bool and arr.size == n_sites:
        idx = np.where(arr)[0]                                   # máscara booleana
    elif (arr.size == n_sites and np.issubdtype(arr.dtype, np.integer)
          and set(np.unique(arr).tolist()) <= {0, 1}):
        idx = np.where(arr != 0)[0]                              # máscara 0/1
    else:
        idx = np.asarray(arr, dtype=int)                        # lista de índices
    idx = idx[(idx >= 0) & (idx < n_sites)]
    idx = np.unique(idx)
    if idx.size == 0:
        raise SandboxError("seleção vazia")
    return idx
