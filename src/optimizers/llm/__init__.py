"""Otimizador guiado por LLM (Fase 4): cliente, sandbox, prompts, ReEvo, zero-shot."""

from .llm_client import LLMClient, build_llm_client, LLMUsage
from .heuristic_runtime import run_heuristic, HeuristicInstance, SandboxError

__all__ = [
    "LLMClient",
    "build_llm_client",
    "LLMUsage",
    "run_heuristic",
    "HeuristicInstance",
    "SandboxError",
]
