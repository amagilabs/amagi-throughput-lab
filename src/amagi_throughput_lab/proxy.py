"""A small round-robin HTTP proxy, so several llama-server instances look like one.

Upstream wrote an nginx config for this, which needs nginx installed (awkward on
Windows). This is the standard library only. It forwards any method and path,
passes streamed (chunked / server-sent events) replies through as they arrive,
and picks the next upstream either in turn or by fewest open requests.
It has no TLS and no auth: keep it on a trusted LAN or behind your real proxy.
"""

import http.client
import itertools
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
               "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def parse_upstream(text):
    """'host:port' or 'http://host:port' -> (host, port)."""
    text = text.strip()
    if "://" in text:
        text = text.split("://", 1)[1]
    text = text.rstrip("/")
    host, sep, port = text.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"upstream must look like host:port, got {text!r}")
    return host or "127.0.0.1", int(port)


class Picker:
    def __init__(self, upstreams, strategy="round-robin"):
        if not upstreams:
            raise ValueError("need at least one upstream")
        if strategy not in ("round-robin", "least-busy"):
            raise ValueError("strategy must be round-robin or least-busy")
        self.upstreams = list(upstreams)
        self.strategy = strategy
        self.active = {u: 0 for u in self.upstreams}
        self.served = {u: 0 for u in self.upstreams}
        self._cycle = itertools.cycle(self.upstreams)
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            if self.strategy == "least-busy":
                # Ties go to whoever is next in turn, so idle servers still alternate.
                start = next(self._cycle)
                order = self.upstreams[self.upstreams.index(start):] + \
                    self.upstreams[:self.upstreams.index(start)]
                choice = min(order, key=lambda u: self.active[u])
            else:
                choice = next(self._cycle)
            self.active[choice] += 1
            self.served[choice] += 1
            return choice

    def release(self, upstream):
        with self._lock:
            self.active[upstream] -= 1


def make_handler(picker, timeout):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet by default
            pass

        def _forward(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            upstream = picker.acquire()
            conn = http.client.HTTPConnection(*upstream, timeout=timeout)
            try:
                headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
                try:
                    conn.request(self.command, self.path, body=body, headers=headers)
                    resp = conn.getresponse()
                except OSError as exc:
                    msg = f"upstream {upstream[0]}:{upstream[1]} unreachable: {exc}".encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                    return
                self.send_response(resp.status, resp.reason)
                for key, value in resp.getheaders():
                    if key.lower() not in HOP_HEADERS:
                        self.send_header(key, value)
                declared = resp.getheader("Content-Length")
                if declared is not None:
                    self.send_header("Content-Length", declared)
                    self.end_headers()
                    self.wfile.write(resp.read())
                    return
                # Unknown length (streaming): relay as chunked, flushing each piece.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            finally:
                conn.close()
                picker.release(upstream)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _forward

        def do_HEAD(self):
            self._forward()

    return Handler


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second process bind a port that is already in
    # use, which is how two servers end up "sharing" one. Only reuse on POSIX.
    allow_reuse_address = os.name != "nt"


class RoundRobinProxy:
    """Bound on construction, so the port is known to be ours before anyone connects."""

    def __init__(self, upstreams, listen_host="127.0.0.1", listen_port=0,
                 strategy="round-robin", timeout=600):
        self.picker = Picker([parse_upstream(u) if isinstance(u, str) else tuple(u)
                              for u in upstreams], strategy)
        self.httpd = _Server((listen_host, int(listen_port)), make_handler(self.picker, timeout))
        self.host, self.port = self.httpd.server_address[:2]
        self._thread = None

    @property
    def base_url(self):
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}"

    def start(self):
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self):
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
