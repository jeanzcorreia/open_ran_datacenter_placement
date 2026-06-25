"""
src/optimizers/llm/llm_client.py — Cliente LLM provider-agnóstico (Fase 4).

Backends:
  • OpenAICompatBackend — qualquer endpoint OpenAI-compatível (Gemini, Groq, ...). Faz
    pacing por RPM (min_interval) E por TPM (token bucket de janela móvel + sync via headers
    x-ratelimit-* quando o provedor os expõe, p.ex. Groq), com backoff+jitter honrando o
    retry do servidor. Cache de respostas em disco. Loga chamadas/tokens/custo e respeita o budget.
  • AnthropicBackend — API Anthropic (opcional; não usado na config atual).
  • OfflineBackend — determinístico, sem rede (fallback / piso sem LLM).
  • RoutedClient — roteia por OP: generate/crossover/mutate -> backend de GERAÇÃO (Groq);
    reflect_* -> backend de REFLEXÃO (Gemini). Usage agregado.

Contrato: complete(op, system, user, context=None, model=None) -> LLMResponse(text, usage).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

# Preço ESTIMADO por 1M tokens (US$) (input, output) — só para logging. Free tier => custo real 0.
_PRICING = {
    # Anthropic
    "claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0), "claude-fable-5": (10.0, 50.0),
    # Google Gemini (estimativa tier pago; free = 0)
    "gemini-2.5-flash-lite": (0.10, 0.40), "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.0),
    # Groq (estimativa tier pago; free = 0)
    "llama-3.3-70b-versatile": (0.59, 0.79), "qwen/qwen3-32b": (0.29, 0.59),
    "qwen/qwen3.6-27b": (0.29, 0.59), "openai/gpt-oss-120b": (0.15, 0.75),
}


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model, in_tok, out_tok, cache_read=0):
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cache_read_tokens += cache_read
        pin, pout = _PRICING.get(model, (0.0, 0.0))
        cost = (in_tok / 1e6) * pin + (out_tok / 1e6) * pout
        self.cost_usd += cost
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
        m["calls"] += 1; m["in"] += in_tok; m["out"] += out_tok; m["cost"] += cost
        return cost

    def merge(self, other: "LLMUsage"):
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_usd += other.cost_usd
        for k, v in other.by_model.items():
            m = self.by_model.setdefault(k, {"calls": 0, "in": 0, "out": 0, "cost": 0.0})
            for kk in ("calls", "in", "out", "cost"):
                m[kk] += v[kk]

    def to_dict(self):
        d = asdict(self)
        d["cost_usd"] = round(self.cost_usd, 4)
        return d


@dataclass
class LLMResponse:
    text: str
    usage: dict = field(default_factory=dict)
    cached: bool = False
    backend: str = ""


class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    """Base: usage acumulado, budget, cache de respostas em disco e log de chamadas."""

    name = "base"

    def __init__(self, model, reflection_model=None, max_tokens=8000, budget=None,
                 cache=True, cache_dir="results/phase4/llm_cache", log_path=None, name=None):
        self.model = model
        self.reflection_model = reflection_model or model
        self.max_tokens = max_tokens
        self.budget = budget or {}
        self.cache = cache
        self.cache_dir = cache_dir
        self.usage = LLMUsage()
        self.log_path = log_path
        # Namespace de cache por "salt" (p.ex. seed): chamadas idênticas de seeds DIFERENTES não
        # colapsam no mesmo cache -> variância real entre seeds (com temperatura > 0).
        self.cache_salt = ""
        if name:
            self.name = name
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def set_cache_salt(self, salt):
        self.cache_salt = str(salt or "")

    def _cache_key(self, op, system, user, model):
        h = hashlib.sha256()
        h.update(f"{self.name}|{self.cache_salt}|{op}|{model}|{system}|{user}".encode("utf-8", "ignore"))
        return h.hexdigest()[:24]

    def _cache_get(self, key):
        if not self.cache:
            return None
        p = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)
        return None

    def _cache_put(self, key, payload):
        if self.cache:
            with open(os.path.join(self.cache_dir, key + ".json"), "w") as fh:
                json.dump(payload, fh)

    def _log(self, record):
        if self.log_path:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")

    def _check_budget(self):
        b = self.budget
        if b.get("max_calls") and self.usage.calls >= b["max_calls"]:
            raise BudgetExceeded(f"max_calls {b['max_calls']} atingido")
        if b.get("max_output_tokens") and self.usage.output_tokens >= b["max_output_tokens"]:
            raise BudgetExceeded(f"max_output_tokens {b['max_output_tokens']} atingido")
        if b.get("usd_limit") and self.usage.cost_usd >= b["usd_limit"]:
            raise BudgetExceeded(f"usd_limit ${b['usd_limit']} atingido")

    def complete(self, op, system, user, context=None, model=None) -> LLMResponse:
        model = model or self.model
        key = self._cache_key(op, system, user, model)
        cached = self._cache_get(key)
        if cached is not None:
            self._log({"op": op, "model": model, "cached": True, "key": key})
            return LLMResponse(cached["text"], cached.get("usage", {}), True, self.name)
        self._check_budget()
        t0 = time.time()
        text, u = self._call(op, system, user, context, model)
        self.usage.add(model, u.get("input_tokens", 0), u.get("output_tokens", 0),
                       u.get("cache_read_tokens", 0))
        self._cache_put(key, {"text": text, "usage": u})
        self._log({"op": op, "model": model, "cached": False, "key": key,
                   "elapsed": round(time.time() - t0, 2), **u})
        return LLMResponse(text, u, False, self.name)

    def _call(self, op, system, user, context, model):
        raise NotImplementedError


# ------------------------------------------------------- pacer por RPM + TPM (token bucket)
class _TokenPacer:
    """Janela móvel de `window` s: limita requests/min (rpm) e tokens/min (tpm). `min_interval`
    força espaçamento mínimo. Sincroniza com headers do servidor quando disponíveis."""

    def __init__(self, tpm=None, rpm=None, min_interval=0.0, window=60.0):
        self.tpm = tpm
        self.rpm = rpm
        self.min_interval = min_interval
        self.window = window
        self.events = []   # (t, tokens)
        self.last = 0.0
        self._server_reset_tokens = 0.0   # segundos até reset de tokens (header)

    def _trim(self, now):
        self.events = [(t, k) for t, k in self.events if now - t < self.window]

    def before(self, est_tokens):
        if self.tpm:
            est_tokens = min(est_tokens, int(self.tpm * 0.9))   # nunca exceder o teto sozinho (evita loop infinito)
        now = time.time()
        if self.min_interval > 0:
            w = self.min_interval - (now - self.last)
            if w > 0:
                time.sleep(w)
        if self.tpm or self.rpm:
            t_start = time.time()
            while time.time() - t_start < self.window * 1.5:    # teto total de espera (~90s)
                now = time.time(); self._trim(now)
                used = sum(k for _, k in self.events)
                cnt = len(self.events)
                tpm_ok = self.tpm is None or used + est_tokens <= self.tpm
                rpm_ok = self.rpm is None or cnt + 1 <= self.rpm
                if tpm_ok and rpm_ok:
                    break
                s = (self.window - (now - self.events[0][0]) + 0.15) if self.events else 0.5
                time.sleep(max(0.2, min(s, self.window)))
        self.last = time.time()
        self.events.append((self.last, est_tokens))
        return self.last

    def after(self, reserve_t, actual_tokens):
        for i, (t, k) in enumerate(self.events):
            if t == reserve_t:
                self.events[i] = (t, actual_tokens)
                return


def _parse_reset(s):
    """'1m26.4s' / '459ms' / '23s' -> segundos."""
    if not s:
        return 0.0
    s = str(s)
    if s.endswith("ms"):
        try:
            return float(s[:-2]) / 1000.0
        except Exception:
            return 0.0
    total = 0.0
    for val, unit in re.findall(r"([\d.]+)(ms|m|s)", s):
        total += float(val) * {"ms": 0.001, "s": 1.0, "m": 60.0}[unit]
    return total


# ----------------------------------------------------- backend OpenAI-compatível (Gemini/Groq)
class OpenAICompatBackend(LLMClient):
    """Endpoint OpenAI-compatível genérico. read_headers=True usa os x-ratelimit-* (Groq)."""

    def __init__(self, *args, api_key=None, base_url=None, min_interval_sec=0.0,
                 tpm=None, rpm=None, est_output_tokens=1400, read_headers=False,
                 max_retries=4, backoff_base=5.0, backoff_max=25.0, backoff_jitter=3.0,
                 request_timeout=40.0, temperature=None, reasoning_effort=None, **kwargs):
        super().__init__(*args, **kwargs)
        from openai import OpenAI

        # timeout curto: uma chamada travada falha rápido e re-tenta (limitado), em vez de pendurar.
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=float(request_timeout))
        self._pacer = _TokenPacer(tpm=tpm, rpm=rpm, min_interval=float(min_interval_sec or 0.0))
        self.est_output_tokens = int(est_output_tokens)
        self.read_headers = read_headers
        self.max_retries = int(max_retries)
        self.backoff_base, self.backoff_max, self.backoff_jitter = backoff_base, backoff_max, backoff_jitter
        # temperatura explícita (variância entre seeds na geração). None = default do provedor.
        self.temperature = None if temperature is None else float(temperature)
        # reasoning_effort: 'none' DESLIGA o "thinking" do gemini-2.5-flash (50s->~9s/chamada,
        # mantendo o modelo forte). None = default do provedor.
        self.reasoning_effort = reasoning_effort

    def _est_tokens(self, system, user):
        return (len(system) + len(user)) // 4 + self.est_output_tokens

    @staticmethod
    def _retry_hint(err):
        s = str(err)
        m = re.search(r"retry in ([\d.]+)s", s) or re.search(r"'retryDelay': '(\d+)s'", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return 0.0
        m = re.search(r"try again in ([\d.]+)s", s)
        return float(m.group(1)) if m else 0.0

    def _call(self, op, system, user, context, model):
        import random

        est = self._est_tokens(system, user)
        last_err = None
        for attempt in range(self.max_retries + 1):
            reserve_t = self._pacer.before(est)
            try:
                kw = {} if self.temperature is None else {"temperature": self.temperature}
                if self.reasoning_effort is not None:
                    kw["reasoning_effort"] = self.reasoning_effort
                raw = self._client.chat.completions.with_raw_response.create(
                    model=model, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}], **kw)
                resp = raw.parse()
                msg = resp.choices[0].message
                text = msg.content or ""
                usage = getattr(resp, "usage", None)
                in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                self._pacer.after(reserve_t, in_tok + out_tok)
                if self.read_headers:
                    self._sync_headers(dict(raw.headers))
                return text, {"input_tokens": in_tok, "output_tokens": out_tok, "cache_read_tokens": 0}
            except Exception as e:
                last_err = e
                es, et = str(e), type(e).__name__
                # retryable: rate limit (429) E indisponibilidade transitória (503/overload/5xx/conexão)
                retryable = ("429" in es or "RateLimit" in et or "RESOURCE_EXHAUSTED" in es
                             or "503" in es or "UNAVAILABLE" in es or "overloaded" in es.lower()
                             or "500" in es or "502" in es or "APIConnection" in et or "Timeout" in et)
                if attempt >= self.max_retries or not retryable:
                    raise
                delay = max(self._retry_hint(e), min(self.backoff_base * (2 ** attempt), self.backoff_max))
                delay += random.uniform(0, self.backoff_jitter)
                time.sleep(delay)
        raise last_err

    def _sync_headers(self, h):
        h = {k.lower(): v for k, v in h.items()}
        rem = h.get("x-ratelimit-remaining-tokens")
        reset = h.get("x-ratelimit-reset-tokens")
        if rem is not None and reset is not None:
            try:
                # se os tokens restantes estão baixos, pré-carrega a janela para forçar espera
                if float(rem) < self.est_output_tokens * 1.5:
                    self._pacer._server_reset_tokens = _parse_reset(reset)
                    time.sleep(min(self._pacer._server_reset_tokens + 0.2, 30.0))
            except Exception:
                pass


# ----------------------------------------------------------------- Anthropic (opcional)
class AnthropicBackend(LLMClient):
    name = "anthropic"

    def __init__(self, *args, api_key=None, **kwargs):
        super().__init__(*args, **kwargs)
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _call(self, op, system, user, context, model):
        resp = self._client.messages.create(
            model=model, max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in resp.content if b.type == "text")
        return text, {"input_tokens": int(resp.usage.input_tokens),
                      "output_tokens": int(resp.usage.output_tokens),
                      "cache_read_tokens": int(getattr(resp.usage, "cache_read_input_tokens", 0) or 0)}


# ----------------------------------------------------------------- Offline (determinístico)
class OfflineBackend(LLMClient):
    name = "offline"

    def _call(self, op, system, user, context, model):
        from . import offline_heuristics as oh

        context = context or {}
        if op == "generate":
            text = oh.generate(context)
        elif op == "crossover":
            text = oh.crossover(context)
        elif op == "mutate":
            text = oh.mutate(context)
        elif op in ("reflect_short", "reflect_long"):
            text = oh.reflect(op, context)
        else:
            text = oh.generate(context)
        return text, {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}


# ----------------------------------------------------------------- Roteador (gen vs reflexão)
class RoutedClient:
    """Roteia por OP: generate/crossover/mutate -> gen_client; reflect_* -> refl_client."""

    GEN_OPS = {"generate", "crossover", "mutate"}

    def __init__(self, gen_client: LLMClient, refl_client: LLMClient):
        self.gen = gen_client
        self.refl = refl_client
        self.name = f"{gen_client.name}(gen)+{refl_client.name}(reflect)"
        self.model = gen_client.model
        self.reflection_model = refl_client.model

    def set_cache_salt(self, salt):
        self.gen.set_cache_salt(salt)
        self.refl.set_cache_salt(salt)

    def complete(self, op, system, user, context=None, model=None) -> LLMResponse:
        client = self.gen if op in self.GEN_OPS else self.refl
        return client.complete(op, system, user, context=context)

    @property
    def usage(self) -> LLMUsage:
        m = LLMUsage()
        m.merge(self.gen.usage)
        m.merge(self.refl.usage)
        return m


# ------------------------------------------------------------------------ factory
def _read_key_file(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return None


def _build_openai_compat(sub, *, max_tokens, budget, cache, cache_dir, log_path) -> OpenAICompatBackend:
    key = _read_key_file(sub.get("api_key_file")) or os.environ.get(sub.get("api_key_env", "") or "_")
    return OpenAICompatBackend(
        model=sub["model"], reflection_model=sub.get("model"), max_tokens=max_tokens,
        budget=budget, cache=cache, cache_dir=cache_dir, log_path=log_path, name=sub.get("provider", "openai"),
        api_key=key, base_url=sub["base_url"],
        min_interval_sec=sub.get("min_interval_sec", 0.0),
        tpm=sub.get("tpm"), rpm=sub.get("rpm"),
        est_output_tokens=sub.get("est_output_tokens", 1400),
        read_headers=sub.get("read_headers", False),
        max_retries=sub.get("max_retries", 4),
        request_timeout=sub.get("request_timeout", 40.0),
        temperature=sub.get("temperature"),
        reasoning_effort=sub.get("reasoning_effort"),
    )


def build_llm_client(cfg, log_path=None) -> LLMClient | RoutedClient:
    """backend ∈ {routed, gemini, groq, anthropic, offline}."""
    llm = cfg.get("llm", {})
    budget = cfg.get("budget", {})
    backend = llm.get("backend", "offline")
    max_tokens = llm.get("max_tokens", 8000)
    cache = llm.get("cache", True)
    cache_dir = llm.get("cache_dir", "results/phase4/llm_cache")

    if backend == "routed":
        base = os.path.splitext(log_path)[0] if log_path else None
        gen = _build_openai_compat(llm["generation"], max_tokens=max_tokens, budget=budget,
                                   cache=cache, cache_dir=cache_dir,
                                   log_path=(base + ".gen.jsonl") if base else None)
        refl = _build_openai_compat(llm["reflection"], max_tokens=max_tokens, budget=budget,
                                    cache=cache, cache_dir=cache_dir,
                                    log_path=(base + ".refl.jsonl") if base else None)
        return RoutedClient(gen, refl)

    common = dict(model=llm.get("model", "gemini-2.5-flash-lite"),
                  reflection_model=llm.get("reflection_model"), max_tokens=max_tokens,
                  budget=budget, cache=cache, cache_dir=cache_dir, log_path=log_path)
    if backend == "offline":
        return OfflineBackend(**common)
    if backend in ("gemini", "groq"):
        key = _read_key_file(llm.get("api_key_file"))
        return OpenAICompatBackend(
            **common, name=backend, api_key=key, base_url=llm["base_url"],
            min_interval_sec=llm.get("min_interval_sec", 0.0), tpm=llm.get("tpm"),
            rpm=llm.get("rpm"), read_headers=llm.get("read_headers", backend == "groq"),
            max_retries=llm.get("max_retries", 6))
    if backend == "anthropic":
        return AnthropicBackend(**common, api_key=os.environ.get("ANTHROPIC_API_KEY")
                                or _read_key_file(llm.get("api_key_file")))
    return OfflineBackend(**common)
