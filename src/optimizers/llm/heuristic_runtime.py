"""
src/optimizers/llm/heuristic_runtime.py — Sandbox para a heurística gerada pelo LLM.

Contrato da heurística (string de código Python):
    def place_odcs(instance, n_active):
        # retorna os índices dos `n_active` sites a ativar (lista/array/set),
        # ou uma máscara booleana de tamanho instance.n_sites.

Segurança + ROBUSTEZ (Fase 5b — ISOLAMENTO DE PROCESSO):
  • Validação por AST (sem import/open/dunder/eval/exec/...), execução com __builtins__ restrito
    + numpy/math, SEM I/O nem rede.
  • A heurística roda num SUBPROCESSO ISOLADO com timeout REAL. No timeout o processo é MORTO
    (kill). Uma heurística travada (laço infinito/lento) morre de fato: **sem threads-zumbi, sem
    acúmulo de GIL — o piso de robustez é DETERMINÍSTICO** (não depende da carga de CPU do momento).
  • Qualquer exceção/violação/timeout/saída inválida => SandboxError (o chamador atribui HV=0; é o
    piso de qualidade — penaliza, não elimina).

Uso eficiente no LAÇO de avaliação: crie UM `HeuristicSandbox(hinst)` por (heurística × cidade) e
reuse-o entre os pontos do sweep — o subprocesso paga o spawn + import do numpy UMA vez (handshake
"ready"), e cada chamada cronometra só a EXECUÇÃO da heurística, não o startup. Feche-o (context
manager) ao fim da avaliação: a contagem de processos volta ao baseline. `run_heuristic` é a versão
one-shot (um subprocesso por chamada) — conveniente para testes/caminhos frios.
"""

from __future__ import annotations

import ast
import math
import multiprocessing as _mp
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


def validate_code(code: str) -> None:
    """API pública: valida o código por AST (rejeita inseguro / sem place_odcs). Lança SandboxError.
    Deve ser chamada UMA vez por heurística (no processo-pai) antes de gastar subprocesso."""
    _validate_ast(code)


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


# ------------------------------------------------------------------ ISOLAMENTO DE PROCESSO
# spawn em todas as plataformas: determinístico e seguro (evita fork com threads do numpy/BLAS).
_MP = _mp.get_context("spawn")
# tempo máximo para o worker importar numpy + sinalizar "ready" (NÃO conta no timeout da heurística).
_WORKER_STARTUP_TIMEOUT = 30.0


def _sandbox_loop(conn, hinst):
    """Roda no SUBPROCESSO: sinaliza "ready", depois atende (code, n_active) até o pai fechar/matar.
    Cada heurística é exec'd com __builtins__ restrito + numpy; a saída é normalizada para índices.
    Devolve ("ok", idx) ou ("err", msg). Prints da heurística são silenciados."""
    try:
        import os as _os
        import sys as _sys
        _sys.stdout = open(_os.devnull, "w")
        _sys.stderr = open(_os.devnull, "w")
    except Exception:
        pass
    try:
        conn.send("ready")
    except Exception:
        return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return  # o pai fechou o pipe
        try:
            code, n_active = msg
            ns = {"np": np, "numpy": np, "math": math, "__builtins__": _SAFE_BUILTINS}
            exec(compile(code, "<heuristic>", "exec"), ns)
            fn = ns.get("place_odcs")
            if not callable(fn):
                conn.send(("err", "place_odcs não é chamável"))
                continue
            out = fn(hinst, int(n_active))
            idx = _normalize_selection(out, hinst.n_sites)
            conn.send(("ok", idx))
        except SandboxError as e:
            conn.send(("err", str(e)))
        except BaseException as e:   # qualquer erro de runtime da heurística -> erro reportado
            conn.send(("err", f"{type(e).__name__}: {e}"))


class HeuristicSandbox:
    """Subprocesso isolado que detém um `hinst` e executa heurísticas com timeout REAL por chamada.

    Inicia o processo UMA vez (spawn + import numpy) e o reusa entre os pontos do sweep; no timeout
    MATA o processo (a heurística travada morre — sem zumbi) e respawna na próxima chamada. Use como
    context manager: ao fechar, o processo é encerrado e a contagem volta ao baseline.

        with HeuristicSandbox(hinst) as sb:
            for n in sweep:
                try: idx = sb.run(code, n, timeout)
                except SandboxError: ...   # crash/timeout/inviável -> HV=0
    """

    def __init__(self, hinst: HeuristicInstance):
        self._hinst = hinst
        self._proc = None
        self._conn = None

    def _ensure(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._close_proc()
        parent_conn, child_conn = _MP.Pipe()
        self._proc = _MP.Process(target=_sandbox_loop, args=(child_conn, self._hinst), daemon=True)
        self._proc.start()
        child_conn.close()
        self._conn = parent_conn
        # espera o "ready" — o startup do worker NÃO conta no timeout da heurística.
        if not self._conn.poll(_WORKER_STARTUP_TIMEOUT):
            self._close_proc()
            raise SandboxError("sandbox: worker não inicializou (startup timeout)")
        try:
            sig = self._conn.recv()
        except (EOFError, OSError):
            self._close_proc()
            raise SandboxError("sandbox: worker morreu no startup")
        if sig != "ready":
            self._close_proc()
            raise SandboxError("sandbox: handshake inválido")

    def run(self, code: str, n_active: int, timeout: float) -> np.ndarray:
        """Executa place_odcs(n_active) no worker com timeout REAL (só a execução). Retorna os
        índices ou lança SandboxError (timeout=processo MORTO, exceção, saída inválida)."""
        self._ensure()
        try:
            self._conn.send((code, int(n_active)))
        except (BrokenPipeError, OSError):
            self._close_proc()
            raise SandboxError("sandbox: pipe quebrado")
        if self._conn.poll(timeout):
            try:
                status, payload = self._conn.recv()
            except (EOFError, OSError):
                self._close_proc()
                raise SandboxError("sandbox: worker morreu durante a chamada")
            if status == "ok":
                return payload
            raise SandboxError(payload)
        # TIMEOUT real -> MATA o processo (a heurística travada morre de fato); respawn na próxima.
        self._close_proc()
        raise SandboxError("timeout")

    def _close_proc(self):
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.kill()      # SIGKILL / TerminateProcess — incondicional
            except Exception:
                pass
            try:
                self._proc.join(timeout=2)
            except Exception:
                pass
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._proc = None
        self._conn = None

    def close(self):
        self._close_proc()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self._close_proc()


def run_heuristic(code: str, hinst: HeuristicInstance, n_active: int, timeout: float = 2.0) -> np.ndarray:
    """Conveniência ONE-SHOT (um subprocesso por chamada): valida + executa place_odcs isolado e
    retorna os índices. Para o LAÇO de avaliação, prefira `HeuristicSandbox` (reusa o processo no
    sweep — paga o spawn 1x). Lança SandboxError em qualquer falha."""
    _validate_ast(code)
    with HeuristicSandbox(hinst) as sb:
        return sb.run(code, n_active, timeout)
