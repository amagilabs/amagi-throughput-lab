from amagi_throughput_lab.recommend import (describe, nginx_snippet, recommend, serve_commands,
                                            target_concurrency)


def row(inst, par, conc, tps, p10, p50=None, p95=1.0, tpj=None, errors=0, status="ok"):
    return {"instances": inst, "parallel": par, "batch": None, "ubatch": None,
            "concurrency": conc, "throughput_tps": tps, "user_tps_p10": p10,
            "user_tps_p50": p50 or p10 * 1.2, "latency_p95_s": p95, "tok_per_j": tpj,
            "errors": errors, "status": status}


META = {"config": {"backend": "llamacpp", "model": "qwen.gguf", "ctx_per_slot": 2048},
        "server_args_full": "-ngl 99 -np 3"}


def test_target_concurrency_is_smallest_tested_at_or_above_users():
    rows = [row(1, 1, c, 10, 10) for c in (1, 4, 8, 16)]
    assert target_concurrency(rows, 10) == (16, True)
    assert target_concurrency(rows, 32) == (16, False)


def test_picks_highest_throughput_that_meets_per_user_floor():
    rows = [row(1, 4, 16, 300, 12), row(1, 16, 16, 500, 8),  # fastest misses the floor
            row(2, 8, 16, 420, 15), row(1, 8, 16, 400, 11)]
    rec = recommend(rows, users=10, min_user_tps=10)
    assert rec["target_concurrency"] == 16
    assert (rec["choice"]["instances"], rec["choice"]["parallel"]) == (2, 8)
    assert len(rec["alternatives"]) == 2


def test_near_tie_prefers_fewer_instances_then_fewer_slots():
    rows = [row(2, 4, 10, 400, 20), row(1, 8, 10, 395, 20), row(1, 16, 10, 398, 20)]
    rec = recommend(rows, users=10, min_user_tps=10)
    assert (rec["choice"]["instances"], rec["choice"]["parallel"]) == (1, 8)


def test_errors_and_p95_cap_disqualify():
    rows = [row(1, 8, 10, 500, 30, errors=1), row(1, 4, 10, 300, 30, p95=9.0),
            row(1, 2, 10, 200, 30, p95=2.0)]
    rec = recommend(rows, users=10, min_user_tps=10, max_p95_s=5)
    assert rec["choice"]["parallel"] == 2


def test_energy_mode_uses_tok_per_j():
    rows = [row(1, 8, 10, 500, 30, tpj=1.5), row(1, 4, 10, 300, 30, tpj=2.0)]
    assert recommend(rows, optimize="energy")["choice"]["parallel"] == 4


def test_fallback_when_nothing_meets_target():
    rows = [row(1, 4, 10, 300, 5), row(1, 8, 10, 350, 7)]
    rec = recommend(rows, users=10, min_user_tps=10)
    assert rec["choice"] is None and rec["fallback"]["parallel"] == 8
    assert "No layout kept" in describe(rec, META)


def test_headroom_is_highest_concurrency_still_meeting_target():
    rows = [row(1, 8, c, 100 * c, 40 / c ** 0.5) for c in (1, 4, 10, 16, 32)]
    rec = recommend(rows, users=10, min_user_tps=8)
    assert rec["headroom"] == 16


def test_single_instance_command_keeps_extra_args_without_duplicating_np():
    lines = serve_commands(row(1, 8, 10, 1, 1), META)
    assert lines == ["llama-server -m qwen.gguf --host 0.0.0.0 --port 8080 --parallel 8 "
                     "--ctx-size 16384 -ngl 99"]


def test_multi_instance_commands_add_the_proxy():
    lines = serve_commands(row(2, 4, 10, 1, 1), META, model="/models/q.gguf")
    assert len(lines) == 3
    assert "--port 8081" in lines[0] and "--port 8082" in lines[1]
    assert lines[2] == ("amagi-throughput-lab proxy --listen 0.0.0.0:8080 "
                        "--upstream 127.0.0.1:8081 --upstream 127.0.0.1:8082")
    assert "upstream llama_backend" in nginx_snippet(2)


def test_vllm_commands():
    meta = {"config": {"backend": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct",
                       "ctx_per_slot": 4096, "vllm_gpu_memory": 0.9}}
    line = serve_commands(row(1, 32, 10, 1, 1), meta)[0]
    assert line.startswith("vllm serve Qwen/Qwen2.5-7B-Instruct")
    assert "--max-num-seqs 32" in line and "--gpu-memory-utilization 0.90" in line


def test_describe_flags_uncovered_and_simulated():
    rows = [row(1, 4, 4, 300, 20)]
    text = describe(recommend(rows, users=10), dict(META, simulated=True))
    assert "never tested 10" in text and "synthetic" in text
    assert "Run it with:" in text
