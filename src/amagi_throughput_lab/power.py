"""GPU power sampling with nvidia-smi, joined to each benchmark cell for tokens per joule.

`nvidia-smi --query-gpu=... -lms 200` prints one line per GPU every 200 ms. A reader
thread stamps each line with this machine's clock when it arrives, so the samples
line up with the request timestamps from the load generator. Energy for a cell is
the trapezoid integral of summed board power between the cell's start and end.

`--power-cmd "ssh gpu-box nvidia-smi"` reads a GPU on another machine; the
arrival stamp then includes the ssh delay (well under a second on a LAN), which
is small next to a cell that runs for tens of seconds.
"""

import subprocess
import threading
import time

QUERY = "index,power.draw,temperature.gpu,memory.used"


def parse_line(line):
    """'0, 291.72, 66, 22204' -> (gpu, watts, temp_c, mem_mib) or None for junk/N/A lines."""
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 4:
        return None
    try:
        gpu = int(parts[0])
        watts = float(parts[1].replace("W", "").strip())
    except ValueError:
        return None

    def num(text):
        try:
            return float(text.replace("MiB", "").strip())
        except ValueError:
            return None

    return gpu, watts, num(parts[2]), num(parts[3])


def integrate(samples, t0, t1, gpus=None):
    """Joules between t0 and t1 from [(t, gpu, watts, ...)] samples.

    Each GPU is integrated on its own (trapezoids between its samples, held flat to
    the window edges), then the GPUs are summed. Returns (joules, avg_watts, n_samples).
    """
    if t1 <= t0:
        return 0.0, 0.0, 0
    by_gpu = {}
    for s in samples:
        if gpus is not None and s[1] not in gpus:
            continue
        by_gpu.setdefault(s[1], []).append((s[0], s[2]))
    total, used = 0.0, 0
    for points in by_gpu.values():
        points.sort()
        inside = [p for p in points if t0 <= p[0] <= t1]
        before = [p for p in points if p[0] < t0]
        after = [p for p in points if p[0] > t1]
        edge0 = before[-1][1] if before else (inside[0][1] if inside else None)
        edge1 = after[0][1] if after else (inside[-1][1] if inside else None)
        if edge0 is None or edge1 is None:
            continue
        series = [(t0, edge0)] + inside + [(t1, edge1)]
        for (ta, wa), (tb, wb) in zip(series, series[1:]):
            total += (tb - ta) * (wa + wb) / 2.0
        used += len(inside)
    return total, total / (t1 - t0), used


class PowerSampler:
    """Background nvidia-smi reader. Use as a context manager around a sweep."""

    def __init__(self, cmd=("nvidia-smi",), interval_ms=200, gpus=None, clock=time.time):
        self.cmd = list(cmd) + [f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
                                f"-lms={int(interval_ms)}"]
        self.gpus = set(gpus) if gpus is not None else None
        self.clock = clock
        self.samples = []
        self.errors = []
        self._lock = threading.Lock()
        self._proc = None
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        try:
            self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          stdin=subprocess.DEVNULL, text=True, bufsize=1)
        except OSError as exc:
            raise RuntimeError(f"cannot run {self.cmd[0]!r} for power sampling: {exc}. "
                               "Use --no-power to skip tok/J.") from exc
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            self.feed(line)

    def feed(self, line, t=None):
        parsed = parse_line(line)
        if parsed is None:
            if line.strip():
                self.errors.append(line.strip())
            return
        gpu, watts, temp, mem = parsed
        if self.gpus is not None and gpu not in self.gpus:
            return
        with self._lock:
            self.samples.append((self.clock() if t is None else t, gpu, watts, temp, mem))

    def wait_for_samples(self, timeout_s=10.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self.samples:
                    return True
            if self._proc is not None and self._proc.poll() is not None:
                break
            time.sleep(0.1)
        return bool(self.samples)

    def window(self, t0, t1):
        """Energy and peaks for one cell."""
        with self._lock:
            samples = list(self.samples)
        joules, avg_w, n = integrate(samples, t0, t1)
        inside = [s for s in samples if t0 <= s[0] <= t1]
        temps = [s[3] for s in inside if s[3] is not None]
        mems = {}
        for s in inside:
            if s[4] is not None:
                mems[s[1]] = max(mems.get(s[1], 0.0), s[4])
        return {
            "energy_j": joules,
            "avg_power_w": avg_w,
            "power_samples": n,
            "max_temp_c": max(temps) if temps else None,
            "peak_mem_mib": sum(mems.values()) if mems else None,
        }

    def latest_temp(self):
        with self._lock:
            temps = [s[3] for s in self.samples[-8:] if s[3] is not None]
        return max(temps) if temps else None

    def stop(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)
