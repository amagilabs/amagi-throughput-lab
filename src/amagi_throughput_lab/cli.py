"""amagi-throughput-lab command line.

    amagi-throughput-lab sweep --model model.gguf --instances 1,2 --parallel 1,4,8 \
        --concurrency 1,2,4,8,10,16 -o results/qwen7b
    amagi-throughput-lab recommend results/qwen7b --users 10 --min-user-tps 10
    amagi-throughput-lab report results/qwen7b
    amagi-throughput-lab proxy --listen 0.0.0.0:8080 --upstream 127.0.0.1:8081 --upstream 127.0.0.1:8082

Exit codes: 0 ok, 2 bad arguments, 3 the run itself failed.
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .prompts import MODES, PromptSource, load_prompts_file


def int_list(text):
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not values or any(v < 1 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive integers, got {text!r}")
    return values


def opt_int_list(text):
    """Like int_list, but 'default' (or empty) means leave the server's own value."""
    if text.strip().lower() in ("", "default"):
        return [None]
    parts = [part.strip() for part in text.split(",") if part.strip()]
    values = [None if part.lower() == "default" else part for part in parts]
    numbers = int_list(",".join(v for v in values if v is not None)) if any(values) else []
    return [None] * (None in values) + numbers


def listen_addr(text):
    host, sep, port = text.rpartition(":")
    if not sep or not port.isdigit():
        raise argparse.ArgumentTypeError(f"expected host:port, got {text!r}")
    return host or "0.0.0.0", int(port)


def build_parser():
    p = argparse.ArgumentParser(
        prog="amagi-throughput-lab",
        description="Find the llama-server (or vLLM) layout that serves N people best on your "
                    "GPU: instances x --parallel x concurrency, with tok/s per person and tok/J.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    s = sub.add_parser("sweep", help="launch servers and measure every layout x concurrency cell")
    s.add_argument("--backend", choices=["llamacpp", "vllm"], default="llamacpp")
    s.add_argument("--server-bin", default=None,
                   help="server command (default: llama-server, or vllm for --backend vllm)")
    s.add_argument("--model", "-m", default="", help="GGUF path (llama.cpp) or model id (vLLM)")
    s.add_argument("--instances", type=int_list, metavar="LIST", default=[1], help="e.g. 1,2 (default 1)")
    s.add_argument("--parallel", "-np", type=int_list, metavar="LIST", default=[1, 4, 8],
                   help="slots per instance (vLLM: --max-num-seqs), e.g. 1,4,8")
    s.add_argument("--batch", type=opt_int_list, metavar="LIST", default=[None], help="--batch-size values")
    s.add_argument("--ubatch", type=opt_int_list, metavar="LIST", default=[None], help="--ubatch-size values")
    s.add_argument("--concurrency", "-c", type=int_list, metavar="LIST", default=[1, 2, 4, 8, 16],
                   help="requests in flight, e.g. 1,2,4,8,10,16")
    s.add_argument("--n-predict", type=int, default=128, help="new tokens per request")
    s.add_argument("--ctx-per-slot", type=int, default=2048,
                   help="context per slot; --ctx-size = this x --parallel")
    s.add_argument("--rounds", type=int, default=2,
                   help="requests per cell = concurrency x rounds (default 2)")
    s.add_argument("--server-args", default="", help="extra server flags, quoted")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--base-port", type=int, default=None,
                   help="first port (default: pick free ports); refuses ports already in use")
    s.add_argument("--attach", action="append", metavar="URL",
                   help="measure already-running server(s) instead of launching; repeatable")
    s.add_argument("--balancer", choices=["proxy", "client"], default="proxy",
                   help="spread load over instances with the built-in proxy or client-side")
    s.add_argument("--prompt-mode", choices=MODES, default="distinct")
    s.add_argument("--prompts", help=".txt (one per line) or .jsonl ({\"prompt\": ...}) file")
    s.add_argument("--temperature", type=float, default=0.3)
    s.add_argument("--allow-eos", action="store_true",
                   help="let replies stop early (default forces --n-predict tokens)")
    s.add_argument("--warmup", type=int, default=2)
    s.add_argument("--ready-timeout", type=float, default=600.0)
    s.add_argument("--request-timeout", type=float, default=600.0)
    s.add_argument("--power-cmd", default="nvidia-smi", help="nvidia-smi command for power")
    s.add_argument("--no-power", action="store_true", help="skip power and tok/J")
    s.add_argument("--gpus", type=lambda t: [int(x) for x in t.split(",")], default=None,
                   help="GPU indexes to sum power over (default: all)")
    s.add_argument("--max-temp", type=float, default=None,
                   help="stop the sweep when the GPU reaches this temperature (C)")
    s.add_argument("--cool-to", type=float, default=None, metavar="C",
                   help="before each cell, wait until the GPU is back down to this temperature (C)")
    s.add_argument("--cool-timeout", type=float, default=600.0,
                   help="longest wait for --cool-to, in seconds")
    s.add_argument("--cell-pause", type=float, default=0.0, help="seconds to rest between cells")
    s.add_argument("--vllm-gpu-memory", type=float, default=0.90,
                   help="total --gpu-memory-utilization, split across vLLM instances")
    s.add_argument("--users", type=int, default=10, help="people to size for (recommendation)")
    s.add_argument("--min-user-tps", type=float, default=10.0)
    s.add_argument("-o", "--out", default="results", help="output directory")

    r = sub.add_parser("recommend", help="pick a layout for N people from a results dir")
    r.add_argument("results", help="results dir or sweep.csv")
    r.add_argument("--users", type=int, default=10)
    r.add_argument("--min-user-tps", type=float, default=10.0,
                   help="tok/s the slowest 10%% of requests must still get")
    r.add_argument("--max-p95", type=float, default=None, help="max p95 wait in seconds")
    r.add_argument("--optimize", choices=["throughput", "energy"], default="throughput")
    r.add_argument("--model", default=None, help="model path to put in the printed commands")
    r.add_argument("--nginx", action="store_true", help="also print an nginx upstream block")
    r.add_argument("--json", action="store_true", help="print the recommendation as JSON")

    rp = sub.add_parser("report", help="write report.html and report.png for a results dir")
    rp.add_argument("results")
    rp.add_argument("--users", type=int, default=10)
    rp.add_argument("--min-user-tps", type=float, default=10.0)
    rp.add_argument("--model", default=None)
    rp.add_argument("--no-png", action="store_true")
    rp.add_argument("-o", "--out", default=None, help="output dir (default: the results dir)")

    px = sub.add_parser("proxy", help="round-robin HTTP proxy over several servers (no nginx)")
    px.add_argument("--listen", type=listen_addr, default=("127.0.0.1", 8080), metavar="HOST:PORT",
                    help="address to serve on (default 127.0.0.1:8080)")
    px.add_argument("--upstream", action="append", required=True, metavar="HOST:PORT")
    px.add_argument("--strategy", choices=["round-robin", "least-busy"], default="least-busy")
    px.add_argument("--timeout", type=float, default=600.0)

    fk = sub.add_parser("fake-server", add_help=False,
                        help="run the built-in simulator (synthetic numbers, for trying the tool)")
    fk.add_argument("fake_args", nargs=argparse.REMAINDER)

    pr = sub.add_parser("prompts", help="show the prompts a sweep would send")
    pr.add_argument("-n", type=int, default=5)
    pr.add_argument("--prompt-mode", choices=MODES, default="distinct")
    pr.add_argument("--prompts", help="prompt file to preview")
    return p


def cmd_sweep(args):
    from .recommend import describe, recommend
    from .sweep import Sweep, SweepConfig

    if not args.attach and not args.model:
        print("error: --model is required unless you --attach to running servers",
              file=sys.stderr)
        return 2
    server_bin = args.server_bin or ("vllm" if args.backend == "vllm" else "llama-server")
    prompts = load_prompts_file(args.prompts) if args.prompts else None
    cfg = SweepConfig(
        backend=args.backend, server_bin=server_bin, model=args.model, host=args.host,
        base_port=args.base_port, instances=args.instances, parallel=args.parallel,
        batch=args.batch, ubatch=args.ubatch, concurrency=args.concurrency,
        n_predict=args.n_predict, ctx_per_slot=args.ctx_per_slot, server_args=args.server_args,
        balancer=args.balancer, rounds=args.rounds, prompt_mode=args.prompt_mode,
        prompts=prompts, temperature=args.temperature, ignore_eos=not args.allow_eos,
        warmup=args.warmup, ready_timeout_s=args.ready_timeout,
        request_timeout_s=args.request_timeout,
        power_cmd=None if args.no_power else args.power_cmd, gpus=args.gpus,
        max_temp_c=args.max_temp, cool_to_c=args.cool_to, cool_timeout_s=args.cool_timeout,
        cell_pause_s=args.cell_pause,
        vllm_gpu_memory=args.vllm_gpu_memory, attach=args.attach, out_dir=args.out)
    rows, meta = Sweep(cfg, log=lambda m: print(m, flush=True)).run()
    out = Path(args.out)
    print(f"\nwrote {out / 'sweep.csv'} ({len(rows)} cells)")
    if not any(r.get("status") == "ok" for r in rows):
        print("no cell succeeded; see the logs folder and the note column", file=sys.stderr)
        return 3
    rec = recommend(rows, users=args.users, min_user_tps=args.min_user_tps)
    # report.html gets shared, so its command uses the model file name from meta, not the path.
    write_reports(rows, meta, rec, out, png=True)
    print()
    print(describe(rec, meta, model=args.model or None))
    return 0


def write_reports(rows, meta, rec, out, model=None, png=True):
    from .report import render_html, render_png

    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(render_html(rows, meta, rec, model=model), encoding="utf-8")
    print(f"wrote {out / 'report.html'}")
    if png:
        render_png(rows, meta, out / "report.png")
        print(f"wrote {out / 'report.png'}")


def load(path):
    from .sweep import load_results

    p = Path(path)
    if not (p / "sweep.csv").exists() and not (p.is_file() and p.suffix == ".csv"):
        raise FileNotFoundError(f"no sweep.csv at {path}")
    return load_results(p)


def cmd_recommend(args):
    from .recommend import describe, nginx_snippet, recommend

    rows, meta = load(args.results)
    rec = recommend(rows, users=args.users, min_user_tps=args.min_user_tps,
                    optimize=args.optimize, max_p95_s=args.max_p95)
    if args.json:
        print(json.dumps(rec, indent=2))
    else:
        print(describe(rec, meta, model=args.model))
        if args.nginx and rec["choice"] and (rec["choice"].get("instances") or 1) > 1:
            print("\nnginx alternative to the built-in proxy:\n")
            print(nginx_snippet(int(rec["choice"]["instances"])))
    return 0 if rec["choice"] else 3


def cmd_report(args):
    from .recommend import recommend

    rows, meta = load(args.results)
    rec = recommend(rows, users=args.users, min_user_tps=args.min_user_tps)
    src = Path(args.results)
    out = Path(args.out) if args.out else (src if src.is_dir() else src.parent)
    write_reports(rows, meta, rec, out, model=args.model, png=not args.no_png)
    return 0


def cmd_proxy(args):
    from .proxy import RoundRobinProxy

    host, port = args.listen
    proxy = RoundRobinProxy(args.upstream, listen_host=host, listen_port=port,
                            strategy=args.strategy, timeout=args.timeout)
    print(f"proxy on {host}:{proxy.port} -> {', '.join(args.upstream)} ({args.strategy})",
          flush=True)
    try:
        proxy.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        proxy.httpd.server_close()
    return 0


def cmd_prompts(args):
    pool = load_prompts_file(args.prompts) if args.prompts else None
    src = PromptSource(args.prompt_mode, pool, run_tag="preview")
    for line in src.batch("c1", args.n):
        print(line)
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["fake-server"]:
        from .fake_server import main as fake_main
        return fake_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 2
    handlers = {"sweep": cmd_sweep, "recommend": cmd_recommend, "report": cmd_report,
                "proxy": cmd_proxy, "prompts": cmd_prompts}
    try:
        return handlers[args.command](args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
