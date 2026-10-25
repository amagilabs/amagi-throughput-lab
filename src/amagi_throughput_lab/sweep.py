"""The instances x --parallel x concurrency sweep.

Same grid as upstream's scripts/full_sweep.py: for each server layout, start the
instances, warm up, then step through the concurrency list and record one row per
cell. Changes from upstream: a distinct prompt per request, a Python round-robin
in place of nginx, a vLLM backend, power and temperature per cell, a temperature
stop, and the exact server commands written to commands.txt.
"""

import csv
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__
from .backends import make_backend, served_model_name
from .loadgen import run_cell
from .power import PowerSampler
from .prompts import PromptSource
from .proxy import RoundRobinProxy
from .servers import (ServerStartError, build_llama_cmd, build_vllm_cmd, launch, plan_ports,
                      probe_backend, server_version, split_command)

FIELDS = [
    "instances", "parallel", "batch", "ubatch", "concurrency", "requests", "ok", "errors",
    "gen_tokens", "prompt_tokens", "wall_s", "throughput_tps", "user_tps_p50", "user_tps_p10",
    "server_decode_tps_p50", "latency_p50_s", "latency_p95_s", "short_replies",
    "avg_power_w", "energy_j", "tok_per_j", "max_temp_c", "peak_mem_mib", "status", "note",
]


@dataclass
class SweepConfig:
    backend: str = "llamacpp"
    server_bin: object = "llama-server"
    model: str = ""
    host: str = "127.0.0.1"
    base_port: object = None
    instances: list = field(default_factory=lambda: [1])
    parallel: list = field(default_factory=lambda: [1, 4, 8])
    batch: list = field(default_factory=lambda: [None])
    ubatch: list = field(default_factory=lambda: [None])
    concurrency: list = field(default_factory=lambda: [1, 2, 4, 8, 16])
    n_predict: int = 128
    ctx_per_slot: int = 2048
    server_args: str = ""
    balancer: str = "proxy"
    rounds: int = 2
    prompt_mode: str = "distinct"
    prompts: list = None
    temperature: float = 0.3
    ignore_eos: bool = True
    warmup: int = 2
    ready_timeout_s: float = 600.0
    request_timeout_s: float = 600.0
    power_cmd: object = "nvidia-smi"
    gpus: list = None
    max_temp_c: float = None
    cell_pause_s: float = 0.0
    vllm_gpu_memory: float = 0.90
    attach: list = None
    out_dir: str = "results"


def fmt(value, digits=2):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def gpu_names(power_cmd):
    if not power_cmd:
        return []
    try:
        out = subprocess.run(split_command(power_cmd) + ["--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


class Sweep:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.out = Path(cfg.out_dir)
        self.rows = []
        self.stopped = ""

    # ---- server layout -------------------------------------------------
    def commands(self, instances, parallel, batch, ubatch, ports):
        cfg = self.cfg
        cmds = []
        for port in ports:
            if cfg.backend == "vllm":
                cmds.append(build_vllm_cmd(cfg.server_bin, cfg.model, cfg.host, port, parallel,
                                           gpu_memory_utilization=cfg.vllm_gpu_memory / instances,
                                           max_model_len=cfg.ctx_per_slot,
                                           extra_args=cfg.server_args))
            else:
                cmds.append(build_llama_cmd(cfg.server_bin, cfg.model, cfg.host, port, parallel,
                                            ctx_per_slot=cfg.ctx_per_slot, batch=batch,
                                            ubatch=ubatch, extra_args=cfg.server_args))
        return cmds

    def layouts(self):
        cfg = self.cfg
        if cfg.attach:
            yield len(cfg.attach), None, None, None
            return
        for n in cfg.instances:
            for p in cfg.parallel:
                for b in cfg.batch:
                    for ub in cfg.ubatch:
                        yield n, p, b, ub

    # ---- one layout ----------------------------------------------------
    def run_layout(self, layout, backend, prompts, sampler, cell_counter):
        instances, parallel, batch, ubatch = layout
        cfg = self.cfg
        base = {"instances": instances, "parallel": parallel, "batch": batch, "ubatch": ubatch}
        if cfg.attach:
            urls = list(cfg.attach)
            return self.run_cells(base, urls, backend, prompts, sampler, cell_counter)
        try:
            ports = plan_ports(cfg.host, instances, cfg.base_port)
        except ServerStartError as exc:
            return self.fail_layout(base, str(exc))
        cmds = self.commands(instances, parallel, batch, ubatch, ports)
        with (self.out / "commands.txt").open("a", encoding="utf-8") as fh:
            fh.write(f"# instances={instances} parallel={parallel} batch={batch} ubatch={ubatch}\n")
            for cmd in cmds:
                fh.write(" ".join(cmd) + "\n")
        self.log(f"starting {instances} x {cfg.backend} (parallel={parallel}) on ports {ports}")
        try:
            with launch(cmds, cfg.host, ports, self.out / "logs", cfg.ready_timeout_s,
                        probe=probe_backend(backend)) as servers:
                urls = [s.base_url for s in servers]
                if cfg.balancer == "proxy" and len(urls) > 1:
                    with RoundRobinProxy(urls, listen_host="127.0.0.1", listen_port=0) as proxy:
                        return self.run_cells(base, [proxy.base_url], backend, prompts, sampler,
                                              cell_counter)
                return self.run_cells(base, urls, backend, prompts, sampler, cell_counter)
        except ServerStartError as exc:
            self.log(f"  layout failed: {exc}".splitlines()[0])
            return self.fail_layout(base, str(exc).splitlines()[0])

    def fail_layout(self, base, note):
        for c in self.cfg.concurrency:
            if any(r["instances"] == base["instances"] and r["parallel"] == base["parallel"]
                   and r["batch"] == base["batch"] and r["ubatch"] == base["ubatch"]
                   and r["concurrency"] == c for r in self.rows):
                continue  # a cell that already ran before the failure keeps its numbers
            self.record(dict(base, concurrency=c, requests=0, ok=0, errors=0,
                             status="start_failed", note=note[:300]))
        return True

    def run_cells(self, base, urls, backend, prompts, sampler, cell_counter):
        cfg = self.cfg
        for _ in range(cfg.warmup):
            run_cell(urls, backend, ["warmup: say ok"], 8, 1, temperature=0.0,
                     request_timeout=cfg.request_timeout_s)
        for c in cfg.concurrency:
            if sampler is not None and cfg.max_temp_c is not None:
                temp = sampler.latest_temp()
                if temp is not None and temp >= cfg.max_temp_c:
                    self.stopped = f"GPU at {temp:.0f} C reached --max-temp {cfg.max_temp_c:.0f} C"
                    self.log(f"stopping: {self.stopped}")
                    return False
            cell_counter[0] += 1
            count = max(1, c * cfg.rounds)
            batch_prompts = prompts.batch(f"c{cell_counter[0]}", count)
            summary = run_cell(urls, backend, batch_prompts, cfg.n_predict, c,
                               temperature=cfg.temperature,
                               request_timeout=cfg.request_timeout_s)
            row = dict(base, concurrency=c, **{k: summary[k] for k in summary
                                               if k in FIELDS})
            if sampler is not None:
                win = sampler.window(summary["t0"], summary["t1"])
                row.update(win)
                row["tok_per_j"] = (summary["gen_tokens"] / win["energy_j"]
                                    if win["energy_j"] > 0 else None)
            row["status"] = "ok" if summary["errors"] == 0 else "errors"
            row["note"] = summary["first_error"][:300]
            self.record(row)
            tpj = row.get("tok_per_j")
            self.log(f"  inst={base['instances']} np={fmt(base['parallel'])} c={c:<4} "
                     f"{summary['throughput_tps']:8.1f} tok/s  per-user p50 "
                     f"{fmt(summary['user_tps_p50'], 1):>6}  p95 wait "
                     f"{fmt(summary['latency_p95_s'], 1):>6}s"
                     + (f"  {tpj:.2f} tok/J" if tpj else "")
                     + (f"  errors={summary['errors']}" if summary["errors"] else ""))
            if cfg.cell_pause_s:
                time.sleep(cfg.cell_pause_s)
        return True

    def record(self, row):
        full = {k: row.get(k) for k in FIELDS}
        self.rows.append(full)
        path = self.out / "sweep.csv"
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: fmt(v, 3) if isinstance(v, float) else ("" if v is None else v)
                             for k, v in full.items()})

    # ---- whole sweep -----------------------------------------------------
    def meta(self, started, version, gpus, power_ok):
        cfg = asdict(self.cfg)
        cfg.pop("prompts", None)
        # Keep the program name, drop its directory (which can name the user).
        parts = split_command(self.cfg.server_bin)
        cfg["server_bin"] = " ".join([os.path.basename(parts[0])] + parts[1:]) if parts else ""
        cfg["model"] = os.path.basename(self.cfg.model) if self.cfg.model else ""
        cfg["attach"] = len(self.cfg.attach) if self.cfg.attach else None
        cfg["power_cmd"] = bool(self.cfg.power_cmd)  # never store a command that names a host
        cfg.pop("out_dir", None)
        return {
            "tool": "amagi-throughput-lab", "tool_version": __version__,
            "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "server_version": version,
            "simulated": "fake" in version.lower(),
            "gpus": gpus, "power_sampled": power_ok,
            "python": platform.python_version(), "os": platform.system(),
            "stopped_early": self.stopped,
            "config": cfg,
            "server_args_full": self.cfg.server_args if isinstance(self.cfg.server_args, str)
            else " ".join(self.cfg.server_args),
        }

    def run(self):
        cfg = self.cfg
        self.out.mkdir(parents=True, exist_ok=True)
        for name in ("sweep.csv", "commands.txt"):
            if (self.out / name).exists():
                (self.out / name).unlink()
        started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if cfg.attach:
            version = ""
        else:
            version = server_version(cfg.server_bin)
        prompts = PromptSource(cfg.prompt_mode, cfg.prompts, run_tag=f"r{int(time.time()) % 100000}")
        model_name = cfg.model
        if cfg.backend == "vllm" and cfg.attach:
            model_name = served_model_name(cfg.attach[0])
        backend = make_backend(cfg.backend, model=model_name, ignore_eos=cfg.ignore_eos)
        gpus = gpu_names(cfg.power_cmd)
        sampler = None
        if cfg.power_cmd:
            sampler = PowerSampler(split_command(cfg.power_cmd), gpus=cfg.gpus).start()
            if not sampler.wait_for_samples(10):
                self.log("power: nvidia-smi gave no samples; tok/J will be blank "
                         + (f"({sampler.errors[0]})" if sampler.errors else ""))
                sampler.stop()
                sampler = None
        counter = [0]
        try:
            for layout in self.layouts():
                if not self.run_layout(layout, backend, prompts, sampler, counter):
                    break
        finally:
            if sampler is not None:
                sampler.stop()
            meta = self.meta(started, version, gpus, sampler is not None)
            (self.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return self.rows, meta


def load_results(path):
    """Read a results dir (or its sweep.csv) back as (rows, meta) with numbers parsed."""
    path = Path(path)
    csv_path = path / "sweep.csv" if path.is_dir() else path
    meta_path = csv_path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row = {}
            for key, value in raw.items():
                if key in ("status", "note"):
                    row[key] = value
                elif value in ("", None):
                    row[key] = None
                else:
                    try:
                        num = float(value)
                        row[key] = int(num) if num.is_integer() and key in (
                            "instances", "parallel", "batch", "ubatch", "concurrency", "requests",
                            "ok", "errors", "gen_tokens", "prompt_tokens", "short_replies") else num
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows, meta


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use the amagi-throughput-lab command")
