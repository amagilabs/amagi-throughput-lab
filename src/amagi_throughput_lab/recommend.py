"""Turn sweep rows into one recommended server layout and the commands to run it.

The question is "N people share this box; each should see at least X tokens/s".
We look at the cells whose concurrency is the smallest tested value >= N (all N
people sending at once, the busy-moment case), keep the layouts where the slowest
10% of requests still reached X tok/s with no errors, and pick the one with the
most total throughput (or the most tokens per joule with optimize="energy").
Near-ties (within 3%) go to fewer instances, then fewer slots, which use less VRAM.
"""

import shlex

from .servers import OVERRIDDEN_FLAGS, flag_value, split_args, strip_flags

TIE = 0.03


def layout_key(row):
    return (row.get("instances"), row.get("parallel"), row.get("batch"), row.get("ubatch"))


def _ok(row):
    return row.get("status") == "ok" and not row.get("errors") and row.get("throughput_tps")


def target_concurrency(rows, users):
    tested = sorted({r["concurrency"] for r in rows if r.get("concurrency")})
    if not tested:
        return None, False
    above = [c for c in tested if c >= users]
    return (above[0], True) if above else (tested[-1], False)


def meets(row, min_user_tps, max_p95_s=None):
    if not _ok(row):
        return False
    p10 = row.get("user_tps_p10")
    if p10 is None or p10 < min_user_tps:
        return False
    if max_p95_s is not None and (row.get("latency_p95_s") or 0) > max_p95_s:
        return False
    return True


def recommend(rows, users=10, min_user_tps=10.0, optimize="throughput", max_p95_s=None):
    if optimize not in ("throughput", "energy"):
        raise ValueError("optimize must be throughput or energy")
    target, covered = target_concurrency(rows, users)
    result = {"users": users, "min_user_tps": min_user_tps, "optimize": optimize,
              "target_concurrency": target, "covered": covered, "choice": None,
              "fallback": None, "alternatives": [], "headroom": None}
    if target is None:
        return result
    cells = [r for r in rows if r.get("concurrency") == target]
    good = [r for r in cells if meets(r, min_user_tps, max_p95_s)]

    def score(r):
        if optimize == "energy":
            return r.get("tok_per_j") or 0.0
        return r.get("throughput_tps") or 0.0

    if not good:
        ranked = sorted([r for r in cells if _ok(r)], key=lambda r: r.get("user_tps_p10") or 0,
                        reverse=True)
        result["fallback"] = ranked[0] if ranked else None
        return result
    best = max(score(r) for r in good)
    near = [r for r in good if score(r) >= best * (1 - TIE)]
    near.sort(key=lambda r: (r.get("instances") or 0, r.get("parallel") or 0, -score(r)))
    choice = near[0]
    result["choice"] = choice
    result["alternatives"] = sorted([r for r in good if r is not choice], key=score,
                                    reverse=True)[:3]
    # How far the same layout stretches before the slowest requests drop below the target.
    same = sorted([r for r in rows if layout_key(r) == layout_key(choice)],
                  key=lambda r: r["concurrency"])
    holds = [r["concurrency"] for r in same if meets(r, min_user_tps, max_p95_s)]
    result["headroom"] = max(holds) if holds else None
    return result


def serve_commands(choice, meta, model=None, host="0.0.0.0", base_port=8081, proxy_port=8080,
                   server_bin=None):
    """Shell lines that bring up the chosen layout for real use."""
    cfg = (meta or {}).get("config", {})
    backend = cfg.get("backend", "llamacpp")
    instances = int(choice.get("instances") or 1)
    parallel = choice.get("parallel")
    model = model or cfg.get("model") or "MODEL.gguf"
    extra = split_args((meta or {}).get("server_args_full") or "")
    user_ctx = flag_value(extra, "--ctx-size", "-c")
    extra = strip_flags(extra, OVERRIDDEN_FLAGS)
    ctx = cfg.get("ctx_per_slot")
    single = instances == 1
    # Behind the proxy, the instances only need to be reachable from this machine.
    bind = host if single else "127.0.0.1"
    lines = []
    for i in range(instances):
        port = proxy_port if single else base_port + i
        if backend == "vllm":
            cmd = [server_bin or "vllm", "serve", model, "--host", bind, "--port", str(port)]
            if parallel:
                cmd += ["--max-num-seqs", str(parallel)]
            gpu_mem = cfg.get("vllm_gpu_memory")
            if gpu_mem:
                cmd += ["--gpu-memory-utilization", f"{gpu_mem / instances:.2f}"]
            if ctx:
                cmd += ["--max-model-len", str(ctx)]
        else:
            cmd = [server_bin or "llama-server", "-m", model, "--host", bind, "--port", str(port)]
            if parallel:
                cmd += ["--parallel", str(parallel)]
                if ctx and user_ctx is None:
                    cmd += ["--ctx-size", str(int(ctx) * int(parallel))]
            if choice.get("batch") is not None:
                cmd += ["--batch-size", str(choice["batch"])]
            if choice.get("ubatch") is not None:
                cmd += ["--ubatch-size", str(choice["ubatch"])]
        lines.append(" ".join(shlex.quote(c) for c in cmd + extra))
    if not single:
        ups = " ".join(f"--upstream 127.0.0.1:{base_port + i}" for i in range(instances))
        lines.append(f"amagi-throughput-lab proxy --listen {host}:{proxy_port} {ups}")
    return lines


def nginx_snippet(instances, base_port=8081, proxy_port=8080):
    ups = "\n".join(f"    server 127.0.0.1:{base_port + i};" for i in range(instances))
    return ("upstream llama_backend {\n" + ups + "\n}\nserver {\n"
            f"    listen {proxy_port};\n    location / {{\n"
            "        proxy_pass http://llama_backend;\n        proxy_http_version 1.1;\n"
            "        proxy_set_header Connection \"\";\n        proxy_buffering off;\n"
            "    }\n}\n")


def layout_name(row):
    np = row.get("parallel")
    text = f"{row.get('instances')} instance(s) x --parallel {np if np is not None else '(as running)'}"
    if row.get("batch") is not None:
        text += f" --batch-size {row['batch']}"
    if row.get("ubatch") is not None:
        text += f" --ubatch-size {row['ubatch']}"
    return text


def describe(rec, meta=None, model=None):
    """Plain-text recommendation for the terminal."""
    out = []
    u, x = rec["users"], rec["min_user_tps"]
    if rec["target_concurrency"] is None:
        return "No results to recommend from."
    c = rec["target_concurrency"]
    if not rec["covered"]:
        out.append(f"note: the sweep never tested {u} concurrent requests; using the highest "
                   f"tested ({c}). Add {u} to --concurrency for a real answer.")
    if (meta or {}).get("simulated"):
        out.append("note: these results came from the fake server (synthetic numbers).")
    ch = rec["choice"]
    if ch is None:
        fb = rec["fallback"]
        out.append(f"No layout kept the slowest 10% of requests at {x:g} tok/s with {c} "
                   "requests in flight.")
        if fb:
            out.append(f"Closest: {layout_name(fb)}: "
                       f"slowest-10% {fb['user_tps_p10']:.1f} tok/s, total "
                       f"{fb['throughput_tps']:.0f} tok/s. Lower the target, shorten replies, "
                       "or use a smaller model.")
        return "\n".join(out)
    tpj = f", {ch['tok_per_j']:.2f} tok/J" if ch.get("tok_per_j") else ""
    out.append(f"For {u} people at >= {x:g} tok/s each (tested at {c} requests in flight):")
    out.append("  " + layout_name(ch))
    out.append(f"  total {ch['throughput_tps']:.0f} tok/s, per person p50 "
               f"{ch['user_tps_p50']:.1f} / slowest-10% {ch['user_tps_p10']:.1f} tok/s, "
               f"p95 wait {ch['latency_p95_s']:.1f}s{tpj}")
    if rec["headroom"] and rec["headroom"] > c:
        out.append(f"  same layout still meets the target at {rec['headroom']} requests in flight")
    out.append("")
    out.append("Run it with:")
    out.extend("  " + line for line in serve_commands(ch, meta, model=model))
    if rec["alternatives"]:
        out.append("")
        out.append("Also met the target:")
        for alt in rec["alternatives"]:
            out.append(f"  {layout_name(alt)}: {alt['throughput_tps']:.0f} "
                       f"tok/s, slowest-10% {alt['user_tps_p10']:.1f} tok/s"
                       + (f", {alt['tok_per_j']:.2f} tok/J" if alt.get("tok_per_j") else ""))
    return "\n".join(out)
