import json

import pytest

from amagi_throughput_lab.prompts import (SAME_PROMPT, TASKS, TOPICS, PromptSource, builtin_pool,
                                          load_prompts_file)


def test_builtin_pool_covers_every_task_topic_pair_once():
    pool = builtin_pool()
    assert len(pool) == len(TASKS) * len(TOPICS) == 320
    assert len(set(pool)) == len(pool)


def test_neighbouring_builtin_prompts_differ_in_task():
    pool = builtin_pool()
    for a, b in zip(pool, pool[1:]):
        assert a.split()[:4] != b.split()[:4]


def test_distinct_mode_never_repeats_within_or_across_cells():
    src = PromptSource("distinct", run_tag="t")
    seen = src.batch("c1", 700) + src.batch("c2", 700)
    assert len(set(seen)) == len(seen)  # tags keep prompts unique even past the pool size
    assert seen[0].startswith("[t-c1-0] ")


def test_same_mode_is_upstream_behaviour():
    assert set(PromptSource("same").batch("c1", 5)) == {SAME_PROMPT}
    assert set(PromptSource("same", ["mine", "other"]).batch("c1", 3)) == {"mine"}


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        PromptSource("random")


def test_load_txt_skips_blanks_and_comments(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# header\nfirst\n\n second \n", encoding="utf-8")
    assert load_prompts_file(f) == ["first", "second"]


def test_load_jsonl(tmp_path):
    f = tmp_path / "p.jsonl"
    f.write_text("\n".join(json.dumps({"prompt": p}) for p in ("a", "b")), encoding="utf-8")
    assert load_prompts_file(f) == ["a", "b"]


@pytest.mark.parametrize("body", ['{"text": "x"}', "not json", ""])
def test_load_jsonl_errors(tmp_path, body):
    f = tmp_path / "p.jsonl"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_prompts_file(f)
