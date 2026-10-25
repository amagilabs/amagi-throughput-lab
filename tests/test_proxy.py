import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from amagi_throughput_lab.proxy import Picker, RoundRobinProxy, parse_upstream


class Upstream:
    """A tiny server that says who it is, can stream, and can hold a request open."""

    def __init__(self, name):
        self.hits = 0
        self.release = threading.Event()
        self.release.set()
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits += 1
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for i in range(3):
                        self.wfile.write(f"data: {name}-{i}\n\n".encode())
                        self.wfile.flush()
                    self.close_connection = True
                    return
                outer.release.wait(10)
                body = json.dumps({"name": name, "path": self.path}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                data = self.rfile.read(length)
                body = json.dumps({"name": name, "echo": data.decode()}).encode()
                self.send_response(201)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.addr = f"127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def upstreams():
    ups = [Upstream("a"), Upstream("b")]
    yield ups
    for u in ups:
        u.close()


def get(proxy, path, method="GET", body=None):
    conn = http.client.HTTPConnection(proxy.host, proxy.port, timeout=10)
    conn.request(method, path, body=body)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, data


def test_parse_upstream():
    assert parse_upstream("http://10.0.0.5:8081/") == ("10.0.0.5", 8081)
    assert parse_upstream(":9000") == ("127.0.0.1", 9000)
    with pytest.raises(ValueError):
        parse_upstream("localhost")


def test_round_robin_alternates(upstreams):
    with RoundRobinProxy([u.addr for u in upstreams]) as proxy:
        names = [json.loads(get(proxy, "/health")[1])["name"] for _ in range(4)]
    assert names == ["a", "b", "a", "b"]


def test_post_body_and_status_pass_through(upstreams):
    with RoundRobinProxy([upstreams[0].addr]) as proxy:
        status, data = get(proxy, "/completion", "POST", b'{"prompt": "hi"}')
    assert status == 201 and json.loads(data)["echo"] == '{"prompt": "hi"}'


def test_streaming_passthrough(upstreams):
    with RoundRobinProxy([upstreams[0].addr]) as proxy:
        status, data = get(proxy, "/stream")
    assert status == 200
    assert data.decode() == "data: a-0\n\ndata: a-1\n\ndata: a-2\n\n"


def test_least_busy_skips_a_busy_upstream(upstreams):
    a, b = upstreams
    a.release.clear()  # requests to "a" hang until released
    with RoundRobinProxy([a.addr, b.addr], strategy="least-busy") as proxy:
        held = threading.Thread(target=get, args=(proxy, "/slow"))
        held.start()
        for _ in range(50):  # wait until the proxy has the request open on "a"
            if proxy.picker.active[parse_upstream(a.addr)] == 1:
                break
            threading.Event().wait(0.05)
        names = [json.loads(get(proxy, "/x")[1])["name"] for _ in range(3)]
        a.release.set()
        held.join(10)
    assert names == ["b", "b", "b"]


def test_dead_upstream_gives_502():
    with RoundRobinProxy(["127.0.0.1:9"]) as proxy:
        status, data = get(proxy, "/health")
    assert status == 502 and b"unreachable" in data


def test_picker_validation():
    with pytest.raises(ValueError):
        Picker([])
    with pytest.raises(ValueError):
        Picker([("h", 1)], strategy="random")
