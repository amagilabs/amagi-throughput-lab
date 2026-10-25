"""Fire N requests at C concurrent workers and summarize what came back.

Upstream's run_batch (scripts/full_sweep.py) measured aggregate tokens per second.
This version also keeps per-request latency and per-request speed, because a team
cares how long each person waits, not only the total. With several backends and
no proxy, requests go round-robin from the client side.
"""

import itertools
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .backends import post_with_retry


def percentile(values, q):
    """Nearest-rank-with-interpolation percentile (q in 0..100); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


class RoundRobin:
    def __init__(self, urls):
        if not urls:
            raise ValueError("need at least one server URL")
        self._cycle = itertools.cycle([u.rstrip("/") for u in urls])
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            return next(self._cycle)


def run_cell(urls, backend, prompts, n_predict, concurrency, temperature=0.3,
             request_timeout=300.0, retry_attempts=4, clock=time.time):
    """Send len(prompts) requests with `concurrency` in flight. Returns raw + summary."""
    rr = RoundRobin(urls)

    def one(prompt):
        url = rr.next() + backend.path
        started = clock()
        try:
            reply = post_with_retry(url, backend.payload(prompt, n_predict, temperature),
                                    timeout=request_timeout, attempts=retry_attempts)
        except Exception as exc:  # recorded, not raised: one bad request shouldn't end a cell
            return {"ok": False, "error": str(exc)[:300], "start": started, "end": clock()}
        ended = clock()
        gen, prompt_tokens, server_tps = backend.parse(reply)
        return {"ok": True, "start": started, "end": ended, "latency_s": ended - started,
                "gen_tokens": gen, "prompt_tokens": prompt_tokens, "server_tps": server_tps}

    t0 = clock()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(one, prompts))
    t1 = clock()
    return summarize(results, t0, t1, n_predict)


def summarize(results, t0, t1, n_predict=None):
    ok = [r for r in results if r["ok"]]
    errors = [r for r in results if not r["ok"]]
    wall = max(t1 - t0, 1e-9)
    gen = sum(r["gen_tokens"] for r in ok)
    prompt = sum(r["prompt_tokens"] for r in ok)
    lat = [r["latency_s"] for r in ok]
    # What one person sees: their tokens over their own wait (queueing + prefill + decode).
    user_tps = [r["gen_tokens"] / r["latency_s"] for r in ok if r["latency_s"] > 0]
    server_tps = [r["server_tps"] for r in ok if r.get("server_tps")]
    short = sum(1 for r in ok if n_predict and r["gen_tokens"] < n_predict)
    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(errors),
        "first_error": errors[0]["error"] if errors else "",
        "t0": t0,
        "t1": t1,
        "wall_s": wall,
        "gen_tokens": gen,
        "prompt_tokens": prompt,
        "throughput_tps": gen / wall,
        "user_tps_p50": percentile(user_tps, 50),
        "user_tps_p10": percentile(user_tps, 10),
        "server_decode_tps_p50": percentile(server_tps, 50),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "short_replies": short,
    }
