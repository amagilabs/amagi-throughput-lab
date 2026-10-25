---
name: amagi-throughput-lab
description: Size a local LLM server for a group of people. Sweeps llama-server (or vLLM) layouts, instances x --parallel slots x concurrent requests, on the user's own GPU, measures total tok/s, per-person tok/s for the slowest 10% of requests, p95 wait and tokens per joule from nvidia-smi, writes an HTML/PNG report, and prints the exact server command to run for N users. Use when asked how many people one GPU can serve, which --parallel / -np value to use, whether to run one or several llama-server instances, how throughput scales with concurrency, or how energy-efficient a serving setup is.
---

# amagi-throughput-lab

A CLI that launches llama-server (or `vllm serve`) in each layout you ask for, fires concurrent
requests at it with a different prompt per request, and records one row per
(instances, --parallel, --batch-size, --ubatch-size, concurrency) cell in `sweep.csv`. Then it
picks the layout that serves N people best and prints the commands to run it.

Based on llama-throughput-lab by Alex Ziskind (MIT).

## When to use this skill

- "How many people can my 3090 (or other GPU) serve with this model?"
- "What --parallel / -np should I use?" or "One llama-server or two?"
- "What's the throughput at 10 concurrent users?" or "What do people see when it's busy?"
- "How many tokens per joule does this setup get?"

## How to use it

Install once: `pipx install amagi-throughput-lab` (or `pip install -e .` from this repo).

```bash
# Launch and sweep. Needs the model file and llama-server on PATH (or --server-bin).
amagi-throughput-lab sweep -m ~/models/model.gguf --server-args "-ngl 99 -fa on" \
    --instances 1,2 --parallel 1,4,8,16 --concurrency 1,2,4,8,10,16,32 \
    --max-temp 83 -o results/model-3090

# Re-read results with a different target.
amagi-throughput-lab recommend results/model-3090 --users 10 --min-user-tps 15
amagi-throughput-lab recommend results/model-3090 --optimize energy
amagi-throughput-lab report results/model-3090          # report.html + report.png

# Measure servers that are already running instead of launching.
amagi-throughput-lab sweep --attach http://127.0.0.1:8080 --concurrency 1,4,10 -o results/live

# vLLM.
amagi-throughput-lab sweep --backend vllm -m Qwen/Qwen2.5-7B-Instruct --parallel 8,32 -o results/vllm

# Serve several instances behind one port, no nginx.
amagi-throughput-lab proxy --listen 0.0.0.0:8080 --upstream 127.0.0.1:8081 --upstream 127.0.0.1:8082

# Try it without a GPU (synthetic numbers, reports say SIMULATED).
amagi-throughput-lab sweep --server-bin "amagi-throughput-lab fake-server" -m fake.gguf --no-power -o results/demo
```

Before sweeping on a shared machine, check that the GPU is idle (`nvidia-smi`): a sweep starts
and stops servers and loads the GPU fully, and any other job on the card skews both its numbers
and yours. Use `--max-temp` to stop if the card runs hot.

Exit codes: 0 ok, 2 usage error, 3 the run failed (no cell succeeded, no layout met the target,
or a results file was missing).

## Reading the results

- `throughput_tps`: all generated tokens in the cell over the cell's wall time.
- `user_tps_p50` / `user_tps_p10`: one request's tokens over its whole wait (queueing, prefill,
  decode). p10 is what the slowest 10% of people see; the recommendation uses it.
- `latency_p95_s`: 95th percentile time for one reply.
- `tok_per_j`: generated tokens over GPU board energy in the cell (nvidia-smi `power.draw`,
  sampled every 200 ms and integrated). Board power only, not the whole PC.
- `status`: `ok`, `errors` (some requests failed, see `note`), or `start_failed` (the server did
  not come up, often out of VRAM; `note` and `logs/` say why).

`meta.json` has the server version, GPU name and settings. It stores file names, not folder paths.
If `meta.json` says `"simulated": true`, the numbers came from the fake server; never present them
as real measurements.

## Notes

- `--ctx-size` is set to `--ctx-per-slot` x `--parallel` so every slot gets the same context.
  Passing your own `-c` in `--server-args` overrides that.
- Requests force the full `--n-predict` tokens (`ignore_eos`) and turn off llama.cpp's prompt cache,
  so cells are comparable. `--allow-eos` lets replies stop early.
- A fixed `--base-port` that is already in use is an error, not a silent reuse; by default free
  ports are picked.
- `--cool-to C` waits before each cell until the GPU core is back to C (at most `--cool-timeout`
  seconds). Set it a few degrees above the card's real idle temperature, or every cell waits out
  the timeout. A 3090 with llama-server loaded holds ~110 W for ~15 s after a cell, then drops to
  ~25 W; `--cool-to 52` cost about 20 s per cell there. On cards whose memory runs hot,
  keep cells short (`--rounds 2`, concurrency up to 16) and watch VRAM temperature separately;
  nvidia-smi does not report GDDR6X junction temperature on GeForce cards.
