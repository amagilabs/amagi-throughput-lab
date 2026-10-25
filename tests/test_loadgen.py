import pytest

from amagi_throughput_lab.loadgen import RoundRobin, percentile, summarize


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5], 10) == 5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([0, 10], 10) == pytest.approx(1.0)


def test_round_robin_cycles():
    rr = RoundRobin(["http://a/", "http://b"])
    assert [rr.next() for _ in range(4)] == ["http://a", "http://b"] * 2


def _ok(start, end, gen=100, prompt=10):
    return {"ok": True, "start": start, "end": end, "latency_s": end - start, "gen_tokens": gen,
            "prompt_tokens": prompt, "server_tps": 50.0}


def test_summarize():
    results = [_ok(0, 1), _ok(0, 2), _ok(0, 4, gen=80),
               {"ok": False, "error": "HTTP 500", "start": 0, "end": 1}]
    s = summarize(results, 0.0, 4.0, n_predict=100)
    assert s["requests"] == 4 and s["ok"] == 3 and s["errors"] == 1
    assert s["first_error"] == "HTTP 500"
    assert s["gen_tokens"] == 280 and s["throughput_tps"] == pytest.approx(70.0)
    assert s["user_tps_p50"] == pytest.approx(50.0)  # 100/1, 100/2, 80/4 -> median 50
    assert s["short_replies"] == 1
    assert s["latency_p95_s"] == pytest.approx(3.8)
