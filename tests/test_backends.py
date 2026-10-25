import pytest

from amagi_throughput_lab.backends import LlamaCpp, OpenAICompletions, make_backend


def test_llamacpp_payload_disables_prompt_cache_and_forces_length():
    body = LlamaCpp().payload("hi", 64, 0.2)
    assert body["n_predict"] == 64 and body["cache_prompt"] is False and body["ignore_eos"]
    assert "ignore_eos" not in LlamaCpp(ignore_eos=False).payload("hi", 64, 0.2)


def test_llamacpp_parse_timings():
    reply = {"timings": {"predicted_n": 128, "prompt_n": 30, "predicted_ms": 2000.0}}
    assert LlamaCpp.parse(reply) == (128, 30, 64.0)


def test_llamacpp_parse_fallbacks():
    assert LlamaCpp.parse({"tokens_predicted": 5, "tokens_evaluated": 7}) == (5, 7, None)
    assert LlamaCpp.parse({"usage": {"completion_tokens": 3, "prompt_tokens": 4}}) == (3, 4, None)


def test_openai_payload_and_parse():
    b = OpenAICompletions(model="m")
    body = b.payload("hi", 32, 0.0)
    assert body["model"] == "m" and body["max_tokens"] == 32
    assert b.parse({"usage": {"completion_tokens": 32, "prompt_tokens": 9}}) == (32, 9, None)


def test_make_backend():
    assert make_backend("vllm", model="x").name == "vllm"
    assert make_backend("llamacpp").path == "/completion"
    with pytest.raises(ValueError):
        make_backend("ollama")
