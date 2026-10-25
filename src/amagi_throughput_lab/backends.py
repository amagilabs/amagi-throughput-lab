"""Request formats for the two server families.

llama.cpp's llama-server takes POST /completion and reports its own timings.
vLLM (and other OpenAI-compatible servers) take POST /v1/completions and report
token counts in `usage`. Response parsing started from upstream's
extract_token_count in tests/llama_server_test_utils.py.
"""

import json
import time
import urllib.error
import urllib.request

BACKENDS = ("llamacpp", "vllm")


class HTTPFailure(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def http_json(url, payload=None, timeout=120.0):
    """GET (payload None) or POST JSON; return the decoded reply or raise HTTPFailure."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Connection": "close"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        with exc:
            body = exc.read().decode("utf-8", errors="replace")
        raise HTTPFailure(exc.code, body) from exc


RETRYABLE = (500, 502, 503, 504)


def post_with_retry(url, payload, timeout, attempts=4, base_sleep_s=0.5):
    """POST, retrying only server-busy/loading codes, like upstream's post_json_with_retry."""
    for attempt in range(attempts):
        try:
            return http_json(url, payload, timeout=timeout)
        except HTTPFailure as exc:
            if exc.status in RETRYABLE and attempt < attempts - 1:
                time.sleep(base_sleep_s * (attempt + 1))
                continue
            raise


class LlamaCpp:
    name = "llamacpp"
    path = "/completion"

    def __init__(self, model=None, ignore_eos=True, cache_prompt=False, seed=None):
        self.ignore_eos = ignore_eos
        self.cache_prompt = cache_prompt
        self.seed = seed

    def payload(self, prompt, n_predict, temperature):
        body = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "stream": False,
            "cache_prompt": self.cache_prompt,
        }
        if self.ignore_eos:
            body["ignore_eos"] = True
        if self.seed is not None:
            body["seed"] = self.seed
        return body

    @staticmethod
    def parse(reply):
        """Return (generated_tokens, prompt_tokens, server_decode_tps or None)."""
        timings = reply.get("timings") or {}
        gen = timings.get("predicted_n")
        if gen is None:
            gen = reply.get("tokens_predicted")
        if gen is None:
            gen = (reply.get("usage") or {}).get("completion_tokens", 0)
        prompt = timings.get("prompt_n")
        if prompt is None:
            prompt = reply.get("tokens_evaluated")
        if prompt is None:
            prompt = (reply.get("usage") or {}).get("prompt_tokens", 0)
        tps = timings.get("predicted_per_second")
        if tps is None and timings.get("predicted_n") and timings.get("predicted_ms"):
            tps = timings["predicted_n"] / (timings["predicted_ms"] / 1000.0)
        return int(gen or 0), int(prompt or 0), (float(tps) if tps else None)


class OpenAICompletions:
    """vLLM and any server that speaks /v1/completions with a `usage` block."""

    name = "vllm"
    path = "/v1/completions"

    def __init__(self, model=None, ignore_eos=True, cache_prompt=False, seed=None):
        self.model = model
        self.ignore_eos = ignore_eos
        self.seed = seed

    def payload(self, prompt, n_predict, temperature):
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": n_predict,
            "temperature": temperature,
            "stream": False,
        }
        if self.ignore_eos:
            body["ignore_eos"] = True  # vLLM extension; others ignore unknown keys or 400
        if self.seed is not None:
            body["seed"] = self.seed
        return body

    @staticmethod
    def parse(reply):
        usage = reply.get("usage") or {}
        return int(usage.get("completion_tokens") or 0), int(usage.get("prompt_tokens") or 0), None


def served_model_name(base_url, timeout=10.0):
    """First model id from GET /v1/models (vLLM needs it in every request)."""
    reply = http_json(base_url.rstrip("/") + "/v1/models", timeout=timeout)
    data = reply.get("data") or []
    if not data or "id" not in data[0]:
        raise RuntimeError(f"{base_url}/v1/models listed no models")
    return data[0]["id"]


def make_backend(name, **kwargs):
    if name == "llamacpp":
        return LlamaCpp(**kwargs)
    if name == "vllm":
        return OpenAICompletions(**kwargs)
    raise ValueError(f"backend must be one of {BACKENDS}, not {name!r}")
