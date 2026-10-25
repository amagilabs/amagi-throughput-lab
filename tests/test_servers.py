import os
import socket
import sys

import pytest

from amagi_throughput_lab.servers import (Server, ServerStartError, build_llama_cmd,
                                          build_vllm_cmd, flag_value, plan_ports, split_args,
                                          split_command, strip_flags)


def test_ctx_size_is_per_slot_times_parallel():
    cmd = build_llama_cmd("llama-server", "m.gguf", "127.0.0.1", 9000, 4, ctx_per_slot=2048,
                          batch=512, ubatch=256)
    assert flag_value(cmd, "--ctx-size") == "8192"
    assert flag_value(cmd, "--parallel") == "4"
    assert flag_value(cmd, "--batch-size") == "512" and flag_value(cmd, "--ubatch-size") == "256"


def test_user_ctx_wins_and_swept_flags_are_not_duplicated():
    cmd = build_llama_cmd("llama-server", "m.gguf", "127.0.0.1", 9000, 4, ctx_per_slot=2048,
                          extra_args="-c 4096 -np 99 --port 1 -ngl 99")
    assert cmd.count("--parallel") == 1 and "-np" not in cmd
    assert cmd.count("--port") == 1 and flag_value(cmd, "--port") == "9000"
    assert "--ctx-size" not in cmd and flag_value(cmd, "-c") == "4096"
    assert cmd[-2:] == ["-ngl", "99"]


def test_split_command_handles_quoted_paths():
    if os.name == "nt":
        cmd = r'"C:\Program Files\llama\llama-server.exe" --verbose'
        assert split_command(cmd) == [r"C:\Program Files\llama\llama-server.exe", "--verbose"]
    else:
        assert split_command("'/opt/my llama/llama-server' -v") == ["/opt/my llama/llama-server",
                                                                   "-v"]
    assert split_command(["a", "b"]) == ["a", "b"]


def test_upstream_comma_args():
    assert split_args("-ngl,99,--flash-attn") == ["-ngl", "99", "--flash-attn"]
    assert strip_flags(["--parallel=4", "-x"], {"--parallel"}) == ["-x"]


def test_vllm_cmd_uses_max_num_seqs():
    cmd = build_vllm_cmd("vllm", "Qwen/Qwen2.5-7B", "127.0.0.1", 9000, 16,
                         gpu_memory_utilization=0.45, max_model_len=4096)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-7B"]
    assert flag_value(cmd, "--max-num-seqs") == "16"
    assert flag_value(cmd, "--gpu-memory-utilization") == "0.45"


def test_plan_ports_refuses_a_busy_fixed_port():
    # Upstream issue #18: a stale server on the port answered health checks.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy = sock.getsockname()[1]
        with pytest.raises(ServerStartError, match="already in use"):
            plan_ports("127.0.0.1", 2, base_port=busy)


def test_plan_ports_auto_picks_distinct_ports():
    ports = plan_ports("127.0.0.1", 3)
    assert len(set(ports)) == 3


def test_wait_ready_reports_a_server_that_died(tmp_path):
    code = "import sys; print('error: model file not found', flush=True); sys.exit(1)"
    server = Server([sys.executable, "-c", code], "127.0.0.1", 9, tmp_path / "s.log")
    server.start()
    try:
        with pytest.raises(ServerStartError) as info:
            server.wait_ready(20)
    finally:
        server.stop()
    assert "exited (code 1)" in str(info.value)
    assert "model file not found" in str(info.value)  # the log tail is in the error


def test_wait_ready_waits_through_503(tmp_path, fake_cmd):
    port = plan_ports("127.0.0.1", 1)[0]
    server = Server(fake_cmd + ["--port", str(port), "--load-s", "1.0"], "127.0.0.1", port,
                    tmp_path / "s.log")
    server.start()
    try:
        server.wait_ready(30)
        assert server.alive()
    finally:
        server.stop()
    assert not server.alive()


def test_missing_binary_is_a_clear_error(tmp_path):
    server = Server(["definitely-not-a-real-llama-server"], "127.0.0.1", 9, tmp_path / "s.log")
    with pytest.raises(ServerStartError, match="--server-bin"):
        server.start()
