# amagi-throughput-lab

Find the llama-server settings that serve the most people from one GPU. It launches the server in each layout you ask for (instances x `--parallel` slots), sends many requests at once with a different prompt each time, and measures total tok/s, the tok/s the slowest 10% of people get, p95 wait and tokens per joule. Then it prints the command to run for your head count. vLLM works too.

![amagi-throughput-lab previewing prompts, recommending a layout for 10 people and writing the report](https://raw.githubusercontent.com/amagilabs/amagi-throughput-lab/master/assets/demo.gif)

Based on llama-throughput-lab by Alex Ziskind (MIT), github.com/alexziskind1/llama-throughput-lab

## Install

```bash
pipx install amagi-throughput-lab
```

Not on PyPI yet. Until it is, install from a clone with `pipx install .` or `pip install .`

Python 3.9 or newer. The only dependency is Pillow, for the PNG chart. You need `llama-server` from llama.cpp (or vLLM) and a model. Power readings need an NVIDIA GPU with `nvidia-smi`.

## Usage

```bash
amagi-throughput-lab sweep -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --server-args "-ngl 99 -fa on" \
    --instances 1,2 --parallel 1,4,8,16 --concurrency 1,2,4,8,10,16,32 \
    --max-temp 83 -o results/qwen7b-3090
```

That starts llama-server once per layout (1 or 2 instances, each with 1, 4, 8 or 16 slots), waits until it answers, then fires 1, 2, 4 and up to 32 requests at it at the same time. Each cell gets its own row in `results/qwen7b-3090/sweep.csv`. When the sweep ends it writes `report.html` and `report.png` and prints a recommendation for 10 people. Change the target with `--users` and `--min-user-tps`.

```bash
amagi-throughput-lab recommend results/qwen7b-3090 --users 10 --min-user-tps 15
amagi-throughput-lab recommend results/qwen7b-3090 --optimize energy     # most tok/J that still meets the target
amagi-throughput-lab recommend results/qwen7b-3090 --nginx               # nginx upstream block, if the pick has 2+ instances
amagi-throughput-lab report results/qwen7b-3090                          # redraw report.html + report.png
amagi-throughput-lab prompts -n 5                                        # the prompts a sweep sends
```

Measure servers you already run instead of launching new ones:

```bash
amagi-throughput-lab sweep --attach http://127.0.0.1:8080 --concurrency 1,4,10 -o results/live
```

vLLM, where `--parallel` becomes `--max-num-seqs`:

```bash
amagi-throughput-lab sweep --backend vllm -m Qwen/Qwen2.5-7B-Instruct --parallel 8,32 -o results/vllm
```

Run two or more instances behind one port without nginx. This also works on Windows:

```bash
amagi-throughput-lab proxy --listen 0.0.0.0:8080 --upstream 127.0.0.1:8081 --upstream 127.0.0.1:8082
```

To try the whole flow without a GPU, point it at the built-in fake server. Its numbers are made up, and every report it feeds is stamped SIMULATED:

```bash
amagi-throughput-lab sweep --server-bin "amagi-throughput-lab fake-server" -m fake.gguf --no-power -o results/demo
```

## A real run

One RTX 3090 at its 300 W limit, llama.cpp build 11753, Qwen2.5-7B-Instruct Q4_K_M fully on the GPU, 128 tokens per reply and 2,048 tokens of context per slot. The 48 cells took 25 minutes:

```bash
amagi-throughput-lab sweep --server-bin ~/gpu-suite-tools/llama.cpp/build/bin/llama-server \
    -m ~/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --server-args "-ngl 99 -fa on" \
    --instances 1,2 --parallel 1,4,8,16 --concurrency 1,2,4,8,10,16 \
    --cool-to 52 --cool-timeout 180 --max-temp 83 -o results/qwen7b-3090
```

For 10 people it picked one llama-server with 16 slots:

```
$ amagi-throughput-lab recommend examples/rtx3090-qwen2.5-7b-q4km --users 10
For 10 people at >= 10 tok/s each (tested at 10 requests in flight):
  1 instance(s) x --parallel 16
  total 662 tok/s, per person p50 66.4 / slowest-10% 65.9 tok/s, p95 wait 1.9s, 3.28 tok/J
  same layout still meets the target at 16 requests in flight

Run it with:
  llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8080 --parallel 16 --ctx-size 32768 -ngl 99 -fa on

Also met the target:
  1 instance(s) x --parallel 8: 407 tok/s, slowest-10% 32.2 tok/s, 1.72 tok/J
  2 instance(s) x --parallel 8: 404 tok/s, slowest-10% 40.4 tok/s, 1.68 tok/J
  2 instance(s) x --parallel 16: 402 tok/s, slowest-10% 40.1 tok/s, 1.64 tok/J
```

![Total tok/s, slowest-10% tok/s and tok/J for each layout on an RTX 3090](https://raw.githubusercontent.com/amagilabs/amagi-throughput-lab/master/examples/rtx3090-qwen2.5-7b-q4km/report.png)

- One slot never got past about 140 tok/s, however many people were waiting. At 10 people the slowest 10% got 13 tok/s each, and 1 reply in 20 took 9.6 s or longer.
- With 10 requests in flight, 16 slots gave 662 tok/s. Eight slots gave 407, and two servers with 8 slots each gave 404. At 8 in flight, 8 and 16 slots both gave 435. That is probably because llama.cpp's CUDA code uses one set of matrix kernels for batches of up to 8 sequences and another above that (`MMVQ_MAX_BATCH_SIZE 8` in `ggml/src/ggml-cuda/mmvq.cuh`), and on this card the bigger-batch kernels were faster per token. A server with 8 slots never batches more than 8, and two servers split the batch in half.
- Tokens per joule climbed with throughput: 3.28 at 10 people and 3.99 at 16 for the 16-slot server, against 0.47 for a single slot.
- The first attempt stopped after five cells. A separate watchdog read this card's memory chips at 106 C (nvidia-smi doesn't report GDDR6X temperature on GeForce cards), and the card is due for new thermal pads. The rerun waited for the core to cool to 52 C before each cell, about 20 s each time, and no cell ran longer than 32 s. Memory peaked at 104 C.

The full results are in [examples/rtx3090-qwen2.5-7b-q4km](examples/rtx3090-qwen2.5-7b-q4km): `sweep.csv`, `meta.json`, `report.html` and `report.png`. Another card, model or set of prompts will give different numbers.

## What it does differently from llama-throughput-lab

Alex Ziskind's [llama-throughput-lab](https://github.com/alexziskind1/llama-throughput-lab) is a dialog-menu launcher with unittest-based sweeps and an nginx round-robin setup. This tool keeps its approach (launch llama-server per configuration, size `--ctx-size` per slot, sweep `--parallel`, batch and concurrency) and changes these things:

- A different prompt on every request. Upstream sent the same prompt every time (his open issue #15), which a prefix cache can answer early. Here the prompts are 320 distinct office-style requests, each tagged with the run, cell and request number, and llama.cpp's prompt cache is off.
- Tokens per joule: `nvidia-smi` power is sampled every 200 ms during each cell and integrated, so every row gets energy, average watts and tok/J. `--max-temp` stops the sweep if the GPU gets too hot.
- `report.html` has three charts (total tok/s, slowest-10% tok/s per person, tok/J against concurrency) plus the full table, in light and dark mode. `report.png` is the same three charts in one image.
- A vLLM backend, through its OpenAI-compatible `/v1/completions`.
- Multiple instances go through a small built-in Python proxy (least-busy or round-robin), so multi-instance layouts work on Windows and on machines without nginx. `--balancer client` spreads requests from the client instead.
- It picks the layout for N people and prints the exact llama-server command (plus the proxy line for multi-instance layouts).
- The port race from his issue #18 is gone. Upstream sent server output to /dev/null and polled `/health` without checking its own process. A stale server on the same port could answer and get measured instead. Here a fixed `--base-port` that is already in use is an error, free ports are picked by default, server output goes to `logs/`, and a server that dies while loading fails right away with the end of its log in the error.

Dropped from upstream: the `dialog` menu and the CPU threads sweep.

## How it measures

- A cell is one layout (instances, `--parallel`, `--batch-size`, `--ubatch-size`) at one concurrency. It sends concurrency x `--rounds` requests (default 2 rounds), keeping `concurrency` of them in flight until the last ones finish.
- Every request generates the same number of tokens: `--n-predict` (default 128) with `ignore_eos`, so cells compare fairly. `--allow-eos` turns that off.
- `throughput_tps` is all generated tokens in the cell over the cell's wall time.
- Per-person tok/s is one request's generated tokens over its whole wait, including queueing and prompt processing. `user_tps_p50` is the median request. `user_tps_p10` is what the slowest 10% of requests got, and it's the number the recommendation checks against `--min-user-tps`.
- `latency_p95_s` is the 95th percentile time from sending a request to getting the full reply.
- `tok_per_j` is generated tokens over GPU board energy for the cell. Board power only, not the whole PC.
- `--ctx-size` is set to `--ctx-per-slot` (default 2048) x `--parallel`, so every slot gets the same context. A `-c` in `--server-args` overrides it.
- The recommendation looks at the smallest tested concurrency that is at least `--users`, keeps the layouts where the slowest 10% still got `--min-user-tps` with no errors, and picks the one with the most total tok/s (or tok/J with `--optimize energy`). When two layouts are within 3%, it takes the one with fewer instances, then fewer slots, because those use less VRAM.
- A server that can't start, usually from running out of VRAM, becomes a `start_failed` row with the reason in `note`. The sweep moves on to the next layout.

## Options

```
amagi-throughput-lab sweep [--backend llamacpp|vllm] [--server-bin CMD] [-m MODEL]
                           [--instances LIST] [--parallel LIST] [--batch LIST] [--ubatch LIST]
                           [--concurrency LIST] [--n-predict N] [--ctx-per-slot N] [--rounds N]
                           [--server-args "FLAGS"] [--host HOST] [--base-port PORT]
                           [--attach URL ...] [--balancer proxy|client]
                           [--prompt-mode distinct|same] [--prompts FILE]
                           [--temperature T] [--allow-eos] [--warmup N]
                           [--ready-timeout S] [--request-timeout S]
                           [--power-cmd CMD] [--no-power] [--gpus 0,1] [--max-temp C]
                           [--cool-to C] [--cool-timeout S] [--cell-pause S] [--vllm-gpu-memory F]
                           [--users N] [--min-user-tps X] [-o DIR]
```

- Lists are comma-separated: `--parallel 1,4,8,16`. `--batch default,512,2048` includes the server's own default.
- `--prompts` takes a `.txt` file (one prompt per line) or `.jsonl` (`{"prompt": ...}` per line) when you'd rather test with your own office's requests. `--prompt-mode same` sends one prompt to every request, to reproduce upstream's numbers.
- `--power-cmd "ssh gpu-box nvidia-smi"` reads power from another machine. `--gpus 0` sums only that GPU.
- `--max-temp 83` stops the sweep when the GPU core reaches 83 C. `--cool-to 55` waits before each cell until the core is back down to 55 C, so every cell starts from about the same temperature. It waits at most `--cool-timeout` seconds (default 600). On the test 3090 with llama-server loaded, the card held about 110 W for some 15 seconds after each cell, then dropped to about 25 W. With `--cool-to 52`, each wait took about 20 seconds.
- `--vllm-gpu-memory` is the total `--gpu-memory-utilization`, split evenly across vLLM instances.
- `recommend` takes `--users`, `--min-user-tps`, `--max-p95 S`, `--optimize throughput|energy`, `--model PATH` (for the printed command), `--nginx` and `--json`.
- Exit codes: 0 ok, 2 usage error, 3 the run failed (no cell succeeded, no layout met the target, or no results file).

## What goes in the results folder

`sweep.csv` and `meta.json` hold the numbers, the settings, the server version and the GPU name. Model and server paths are reduced to file names, and your `--power-cmd` isn't stored. Your `--server-args` text is copied as written. `commands.txt` and `logs/` keep the full server command lines and server output for debugging. They contain your local paths, so leave them out if you share a results folder.

## Use it from Claude Code

This repo ships a skill at `.claude/skills/amagi-throughput-lab/SKILL.md`. Copy that folder into your own `.claude/skills/` and Claude Code can run a sweep, read the results and answer "how many people can this GPU serve" with you.

## Limits

- The vLLM backend is tested against the built-in fake server only. It has not been run against a real vLLM server yet.
- Power and temperature need an NVIDIA GPU and `nvidia-smi`. On other GPUs, use `--no-power` and the speed numbers still work.
- Board power only. The CPU, RAM and the rest of the PC aren't counted, so tok/J flatters every layout by the same amount.
- One generation length per sweep. Long chats and long documents shift the answer toward fewer slots with more context each, so sweep with your real `--n-predict` and `--prompts` when you can.
- The sweep needs the GPU to itself. Anything else running on the card skews both its numbers and yours.

## If this saved you time

A star helps other people find it.

## More from AmagiLabs

[github.com/amagilabs](https://github.com/amagilabs) has the other small open source dev tools from the same lab.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2025 Alex Ziskind and Copyright (c) 2026 Amagisolutionsgroup, Inc. (AmagiLabs)
