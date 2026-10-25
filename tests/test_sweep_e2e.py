"""End to end: launch the simulator as the server, sweep, read results back, report."""

import json
import sys

import pytest

from amagi_throughput_lab.cli import main
from amagi_throughput_lab.sweep import FIELDS, Sweep, SweepConfig, load_results


def cfg(tmp_path, fake_cmd, **kw):
    base = dict(server_bin=fake_cmd, model="fake.gguf", instances=[1], parallel=[1, 4],
                concurrency=[1, 4], n_predict=16, rounds=2, warmup=1, power_cmd=None,
                ready_timeout_s=30, request_timeout_s=30, out_dir=str(tmp_path / "out"))
    base.update(kw)
    return SweepConfig(**base)


def test_llamacpp_sweep_writes_csv_meta_and_commands(tmp_path, fake_cmd):
    rows, meta = Sweep(cfg(tmp_path, fake_cmd), log=lambda m: None).run()
    assert len(rows) == 4 and all(r["status"] == "ok" for r in rows)
    assert all(r["requests"] == r["concurrency"] * 2 == r["ok"] for r in rows)
    assert all(r["gen_tokens"] == 16 * r["requests"] for r in rows)
    out = tmp_path / "out"
    loaded, loaded_meta = load_results(out)
    assert [r["concurrency"] for r in loaded] == [1, 4, 1, 4]
    assert list(loaded[0]) == FIELDS
    assert loaded_meta["simulated"] is True and meta["config"]["model"] == "fake.gguf"
    assert meta["config"]["server_bin"].startswith("python")  # directory stripped
    commands = (out / "commands.txt").read_text(encoding="utf-8")
    assert "--parallel 4" in commands and "--ctx-size 8192" in commands


def test_more_slots_beat_one_slot_under_load(tmp_path, fake_cmd):
    slow = fake_cmd + ["--tps", "100"]  # decode time dominates HTTP overhead
    rows, _ = Sweep(cfg(tmp_path, slow, concurrency=[8]), log=lambda m: None).run()
    one, four = rows
    assert four["throughput_tps"] > one["throughput_tps"] * 1.5


def test_vllm_backend_against_simulator(tmp_path, fake_cmd):
    c = cfg(tmp_path, fake_cmd, backend="vllm", model="Qwen/fake", parallel=[4],
            concurrency=[4])
    rows, meta = Sweep(c, log=lambda m: None).run()
    assert rows[0]["status"] == "ok" and rows[0]["gen_tokens"] == 16 * 8
    assert "serve Qwen/fake" in (tmp_path / "out" / "commands.txt").read_text(encoding="utf-8")


def test_two_instances_go_through_the_proxy(tmp_path, fake_cmd):
    rows, _ = Sweep(cfg(tmp_path, fake_cmd, instances=[2], parallel=[1], concurrency=[4]),
                    log=lambda m: None).run()
    assert rows[0]["status"] == "ok" and rows[0]["instances"] == 2
    logs = sorted((tmp_path / "out" / "logs").glob("server_*.log"))
    assert len(logs) == 2


def test_server_that_cannot_start_is_recorded_not_fatal(tmp_path):
    bad = [sys.executable, "-c", "import sys; print('CUDA out of memory'); sys.exit(1)"]
    rows, _ = Sweep(cfg(tmp_path, bad, parallel=[4]), log=lambda m: None).run()
    assert {r["status"] for r in rows} == {"start_failed"}
    assert "exited" in rows[0]["note"]


def test_power_is_joined_per_cell(tmp_path, fake_cmd):
    smi = tmp_path / "smi.py"
    smi.write_text("import sys, time\n"
                   "if any(a.startswith('--query-gpu=name') for a in sys.argv):\n"
                   "    print('Fake GPU 24GB'); sys.exit(0)\n"
                   "while True:\n"
                   "    print('0, 200.0, 55, 8000', flush=True); time.sleep(0.05)\n",
                   encoding="utf-8")
    c = cfg(tmp_path, fake_cmd, parallel=[4], concurrency=[4],
            power_cmd=[sys.executable, str(smi)])
    rows, meta = Sweep(c, log=lambda m: None).run()
    r = rows[0]
    assert r["avg_power_w"] == pytest.approx(200.0)
    assert r["energy_j"] == pytest.approx(200.0 * r["wall_s"], rel=0.02)
    assert r["tok_per_j"] == pytest.approx(r["gen_tokens"] / r["energy_j"])
    assert meta["gpus"] == ["Fake GPU 24GB"] and meta["config"]["power_cmd"] is True


def test_max_temp_stops_the_sweep(tmp_path, fake_cmd):
    smi = tmp_path / "hot.py"
    smi.write_text("import sys, time\n"
                   "if any(a.startswith('--query-gpu=name') for a in sys.argv): sys.exit(0)\n"
                   "while True:\n"
                   "    print('0, 300.0, 91, 8000', flush=True); time.sleep(0.05)\n",
                   encoding="utf-8")
    c = cfg(tmp_path, fake_cmd, power_cmd=[sys.executable, str(smi)], max_temp_c=85)
    rows, meta = Sweep(c, log=lambda m: None).run()
    assert rows == [] and "91 C" in meta["stopped_early"]


def test_cli_sweep_then_recommend_and_report(tmp_path, fake_cmd, capsys):
    out = tmp_path / "cli"
    server = " ".join(f'"{p}"' if " " in p else p for p in fake_cmd)
    code = main(["sweep", "--server-bin", server, "--model", "fake.gguf", "--parallel", "1,4",
                 "--concurrency", "1,4,10", "--n-predict", "16", "--no-power", "--warmup", "1",
                 "--rounds", "1", "-o", str(out)])
    text = capsys.readouterr().out
    assert code == 0, text
    assert "For 10 people" in text and "Run it with:" in text
    assert (out / "report.html").exists() and (out / "report.png").exists()
    assert main(["recommend", str(out), "--users", "4", "--json"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["target_concurrency"] == 4


def test_cool_to_waits_before_each_cell(tmp_path, fake_cmd):
    smi = tmp_path / "cooling.py"
    smi.write_text("import sys, time\n"
                   "if any(a.startswith('--query-gpu=name') for a in sys.argv): sys.exit(0)\n"
                   "t0 = time.time()\n"
                   "while True:\n"
                   "    temp = 75 if time.time() - t0 < 2.0 else 45\n"
                   "    print(f'0, 250.0, {temp}, 8000', flush=True); time.sleep(0.05)\n",
                   encoding="utf-8")
    lines = []
    c = cfg(tmp_path, fake_cmd, parallel=[4], concurrency=[1, 4],
            power_cmd=[sys.executable, str(smi)], cool_to_c=50, cool_timeout_s=30)
    rows, _ = Sweep(c, log=lines.append).run()
    assert [r["status"] for r in rows] == ["ok", "ok"]
    assert any("cooling: GPU at 75 C, waiting for 50 C" in line for line in lines)
