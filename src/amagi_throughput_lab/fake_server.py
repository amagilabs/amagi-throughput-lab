"""A stand-in for llama-server / vLLM, for tests and for trying the tool without a GPU.

Every number it produces is SYNTHETIC. It sleeps to imitate a server with
`--parallel` slots: requests beyond the slot count queue, prefill takes
prompt_tokens / --prefill-tps seconds, and each slot decodes at
--tps / (1 + --contention x (busy_slots - 1)) tokens per second. Its
`--version` line says "fake", and sweep reports carry a SIMULATED banner
whenever results came from it.

    python -m amagi_throughput_lab.fake_server --port 9000 --parallel 4
"""

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION_LINE = "version: fake (amagi-throughput-lab simulator, synthetic numbers)"


def parse_args(argv):
    p = argparse.ArgumentParser(prog="fake_server", add_help=True)
    p.add_argument("positional", nargs="*")  # vLLM style: serve MODEL
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", "-m", default="fake-model.gguf")
    p.add_argument("--parallel", "-np", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--tps", type=float, default=60.0, help="single-slot decode tokens/s")
    p.add_argument("--prefill-tps", type=float, default=2000.0)
    p.add_argument("--contention", type=float, default=0.12)
    p.add_argument("--load-s", type=float, default=0.3, help="seconds of 503 before healthy")
    p.add_argument("--version", action="store_true")
    args, _unknown = p.parse_known_args(argv)  # ignore every other llama-server flag
    if args.max_num_seqs:
        args.parallel = args.max_num_seqs
    if len(args.positional) >= 2 and args.positional[0] == "serve":
        args.model = args.positional[1]
    return args


class FakeLLM:
    def __init__(self, parallel, tps, prefill_tps, contention, load_s, model):
        self.slots = threading.Semaphore(max(1, parallel))
        self.tps = tps
        self.prefill_tps = prefill_tps
        self.contention = contention
        self.model = model
        self.busy = 0
        self.lock = threading.Lock()
        self.ready_at = time.time() + load_s

    def generate(self, prompt, n_predict):
        prompt_tokens = max(1, len(prompt.split()) * 4 // 3)
        with self.slots:
            with self.lock:
                self.busy += 1
                busy = self.busy
            try:
                prefill_s = prompt_tokens / self.prefill_tps
                rate = self.tps / (1.0 + self.contention * (busy - 1))
                decode_s = n_predict / rate
                time.sleep(prefill_s + decode_s)
            finally:
                with self.lock:
                    self.busy -= 1
        return prompt_tokens, prefill_s, decode_s


def make_handler(llm):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if time.time() < llm.ready_at:
                return self._send(503, {"error": {"message": "Loading model"}})
            if self.path == "/health":
                return self._send(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._send(200, {"data": [{"id": llm.model}]})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            if time.time() < llm.ready_at:
                return self._send(503, {"error": {"message": "Loading model"}})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})
            prompt = str(req.get("prompt", ""))
            if self.path == "/completion":
                n = int(req.get("n_predict", 16))
                pt, pre, dec = llm.generate(prompt, n)
                return self._send(200, {
                    "content": "simulated " * n, "tokens_predicted": n, "tokens_evaluated": pt,
                    "simulated": True,
                    "timings": {"prompt_n": pt, "prompt_ms": pre * 1000, "predicted_n": n,
                                "predicted_ms": dec * 1000, "predicted_per_second": n / dec},
                })
            if self.path == "/v1/completions":
                n = int(req.get("max_tokens", 16))
                pt, _pre, _dec = llm.generate(prompt, n)
                return self._send(200, {
                    "object": "text_completion", "model": llm.model, "simulated": True,
                    "choices": [{"text": "simulated " * n, "index": 0, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": pt, "completion_tokens": n,
                              "total_tokens": pt + n},
                })
            return self._send(404, {"error": "not found"})

    return Handler


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        print(VERSION_LINE)
        return 0
    llm = FakeLLM(args.parallel, args.tps, args.prefill_tps, args.contention, args.load_s,
                  args.model)
    ThreadingHTTPServer.daemon_threads = True
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(llm))
    print(f"fake server on {args.host}:{args.port} parallel={args.parallel} (synthetic)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
