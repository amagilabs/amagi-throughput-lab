"""Start, health-check and stop llama-server / vLLM instances.

The command building (ctx-size = per-slot context x --parallel, dropping
--parallel/--batch-size/--ubatch from the base args) follows upstream's
start_llama_server and build_server_args.

Port race fix (upstream issue #18, "Port 8088 did not become ready"): upstream
sent server and nginx output to /dev/null, did not check that a port was free
before launching, and polled /health without checking that its own process was
still running. A stale server left on the port could answer /health and be
measured instead, and a server that crashed on start only showed up as a
timeout. Here every port is checked free first, output goes to a log file, the
wait loop stops the moment the process exits, and the error quotes the log tail.
"""

import contextlib
import os
import shlex
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .backends import HTTPFailure, http_json

OVERRIDDEN_FLAGS = {"--parallel", "-np", "--batch-size", "-b", "--ubatch-size", "--ubatch", "-ub",
                    "--host", "--port", "--model", "-m"}


class ServerStartError(RuntimeError):
    pass


def split_command(cmd):
    """'llama-server' or 'python -m x' or a list -> argv list."""
    if isinstance(cmd, (list, tuple)):
        return list(cmd)
    if not cmd:
        return []
    if os.name != "nt":
        return shlex.split(cmd)
    # Non-POSIX mode keeps backslashes in Windows paths but also keeps the quotes.
    return [part[1:-1] if len(part) > 1 and part[0] == part[-1] and part[0] in "\"'" else part
            for part in shlex.split(cmd, posix=False)]


def split_args(raw):
    """Upstream's parse_comma_args: comma-separated or shell-style server args."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return shlex.split(raw)


def flag_value(args, *flags):
    for idx, arg in enumerate(args):
        for flag in flags:
            if arg == flag and idx + 1 < len(args):
                return args[idx + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
    return None


def strip_flags(args, flags):
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in flags:
            skip = True
            continue
        if any(arg.startswith(f + "=") for f in flags):
            continue
        out.append(arg)
    return out


def build_llama_cmd(server_cmd, model, host, port, parallel, ctx_per_slot=None,
                    batch=None, ubatch=None, extra_args=()):
    extra = split_args(extra_args)
    user_ctx = flag_value(extra, "--ctx-size", "-c")
    extra = strip_flags(extra, OVERRIDDEN_FLAGS)
    cmd = split_command(server_cmd) + ["--host", host, "--port", str(port), "--model", model]
    if user_ctx is None and ctx_per_slot:
        cmd += ["--ctx-size", str(int(ctx_per_slot) * int(parallel))]
    cmd += ["--parallel", str(parallel)]
    if batch is not None:
        cmd += ["--batch-size", str(batch)]
    if ubatch is not None:
        cmd += ["--ubatch-size", str(ubatch)]
    return cmd + extra


def build_vllm_cmd(server_cmd, model, host, port, parallel, gpu_memory_utilization=None,
                   max_model_len=None, extra_args=()):
    """vLLM's closest knob to llama.cpp's --parallel is --max-num-seqs."""
    extra = strip_flags(split_args(extra_args), {"--host", "--port", "--max-num-seqs"})
    cmd = split_command(server_cmd) + ["serve", model, "--host", host, "--port", str(port),
                                      "--max-num-seqs", str(parallel)]
    if gpu_memory_utilization is not None and "--gpu-memory-utilization" not in extra:
        cmd += ["--gpu-memory-utilization", f"{gpu_memory_utilization:.2f}"]
    if max_model_len is not None and "--max-model-len" not in extra:
        cmd += ["--max-model-len", str(max_model_len)]
    return cmd + extra


def port_is_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    # Something may also be listening on another address for the same port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, int(port))) != 0


def free_port(host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def plan_ports(host, count, base_port=None):
    """Ports for `count` servers. A fixed base must be free all the way up; no silent reuse."""
    if base_port is None:
        ports = []
        while len(ports) < count:
            port = free_port(host)
            if port not in ports:
                ports.append(port)
        return ports
    ports = [int(base_port) + i for i in range(count)]
    busy = [p for p in ports if not port_is_free(host, p)]
    if busy:
        raise ServerStartError(
            f"port(s) {', '.join(map(str, busy))} on {host} are already in use. Stop the old "
            "server (a stale llama-server would answer health checks and be measured instead) "
            "or pick another --base-port."
        )
    return ports


def tail(path, lines=20):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


class Server:
    """One launched server process with a log file."""

    def __init__(self, cmd, host, port, log_path, health_path="/health"):
        self.cmd = list(cmd)
        self.host = "127.0.0.1" if host in ("0.0.0.0", "") else host
        self.port = int(port)
        self.log_path = Path(log_path)
        self.health_path = health_path
        self.process = None
        self._log = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8", errors="replace")
        self._log.write("$ " + " ".join(shlex.quote(c) for c in self.cmd) + "\n")
        self._log.flush()
        try:
            self.process = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL)
        except FileNotFoundError as exc:
            self._log.close()
            raise ServerStartError(f"cannot run {self.cmd[0]!r}: {exc}. Set --server-bin.") from exc
        return self

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def _died(self, what):
        code = self.process.returncode if self.process else None
        return ServerStartError(
            f"server on port {self.port} exited (code {code}) while {what}. "
            f"Log {self.log_path}:\n{tail(self.log_path)}"
        )

    def wait_ready(self, timeout_s, probe=None):
        """Wait for /health 200 from *our* process, then for `probe()` to succeed."""
        deadline = time.time() + timeout_s
        last = None
        url = self.base_url + self.health_path
        while time.time() < deadline:
            if not self.alive():
                raise self._died("loading")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        break
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"  # 503 = still loading the model
            except OSError as exc:
                last = str(exc)
            time.sleep(0.25)
        else:
            raise ServerStartError(
                f"server on port {self.port} not healthy after {timeout_s:.0f}s (last: {last}). "
                f"Log {self.log_path}:\n{tail(self.log_path)}"
            )
        if probe is None:
            return
        while time.time() < deadline:
            if not self.alive():
                raise self._died("warming up")
            try:
                probe(self.base_url)
                return
            except HTTPFailure as exc:
                if exc.status != 503:
                    raise ServerStartError(f"probe request to port {self.port} failed: {exc}") from exc
                last = "HTTP 503"
            except OSError as exc:
                last = str(exc)
            time.sleep(0.25)
        raise ServerStartError(f"server on port {self.port} never answered a request ({last})")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self._log is not None and not self._log.closed:
            self._log.close()


@contextlib.contextmanager
def launch(cmds, host, ports, log_dir, ready_timeout_s, probe=None, health_path="/health",
           startup_delay_s=0.0):
    """Start servers one by one (each loads before the next asks for VRAM); stop all on exit."""
    servers = []
    try:
        for i, (cmd, port) in enumerate(zip(cmds, ports)):
            server = Server(cmd, host, port, Path(log_dir) / f"server_{i}_{port}.log", health_path)
            servers.append(server)
            server.start()
            server.wait_ready(ready_timeout_s, probe=probe)
            if startup_delay_s:
                time.sleep(startup_delay_s)
        yield servers
    finally:
        for server in reversed(servers):
            server.stop()


def server_version(server_cmd):
    """`llama-server --version` first line, or '' if it can't be read."""
    try:
        out = subprocess.run(split_command(server_cmd) + ["--version"], capture_output=True,
                             text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    for line in (out.stdout + "\n" + out.stderr).splitlines():
        if "version" in line.lower():
            return line.strip()
    return ""


def probe_backend(backend):
    """A 1-token request, used to confirm the model answers before timing starts."""
    def probe(base_url):
        http_json(base_url + backend.path, backend.payload("ping", 1, 0.0), timeout=30)
    return probe
