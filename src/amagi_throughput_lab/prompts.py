"""Prompt sources for a sweep.

Upstream sent the same prompt with every request (its open issue #15). With one
prompt, llama.cpp's prompt cache and identical decode paths make concurrency look
better than real traffic. Here every request gets its own prompt by default: a
task from a built-in pool of office-style requests, a topic, and a request tag,
so no two requests in a sweep share a prefix.
"""

import json
from pathlib import Path

TASKS = [
    "Write a short, polite email asking a coworker for an update on {topic} by Friday.",
    "Summarize the main risks of {topic} in five bullet points.",
    "Explain {topic} to a new employee in plain language.",
    "Draft a one-paragraph status update about {topic} for a manager.",
    "List three questions a client might ask about {topic}, with brief answers.",
    "Write a checklist for {topic} that fits on one page.",
    "Give two arguments for and two against {topic}.",
    "Turn these notes about {topic} into a clear meeting agenda.",
    "Write a Python function that helps track {topic}, with a docstring.",
    "Suggest a subject line and a three-sentence body for an email about {topic}.",
    "Describe a common mistake people make with {topic} and how to avoid it.",
    "Rewrite this sentence to sound more professional: we kinda dropped the ball on {topic}.",
    "Create a short FAQ entry about {topic} for the company intranet.",
    "Estimate the time needed for {topic} and explain your assumptions.",
    "Write a friendly reminder message to the team about {topic}.",
    "Outline a 30-minute training session on {topic}.",
]

TOPICS = [
    "the quarterly invoice backlog",
    "a vendor contract renewal",
    "the new printer setup",
    "backing up the shared drive",
    "onboarding a part-time assistant",
    "a late client payment",
    "moving the office to a new internet provider",
    "the holiday schedule",
    "the password manager rollout",
    "a lease renewal",
    "cleaning up old customer records",
    "replacing two aging laptops",
    "a website contact form that stops working",
    "monthly bookkeeping close",
    "a phishing email that reached three staff",
    "scheduling a quarterly client review",
    "inventory counts before year end",
    "a slow Wi-Fi network in the conference room",
    "an expense report policy",
    "the parking lot repaving",
]

MODES = ("distinct", "same")
SAME_PROMPT = "Share three optimization tips for model serving."


def builtin_pool():
    """Every task x topic combination, in a fixed interleaved order (320 prompts)."""
    # i = 16a + b gives task b and topic (17a + b) % 20; 17 and 20 are coprime,
    # so every pair appears exactly once and neighbours differ in both parts.
    pool = []
    for i in range(len(TASKS) * len(TOPICS)):
        task = TASKS[i % len(TASKS)]
        topic = TOPICS[(i // len(TASKS) + i) % len(TOPICS)]
        pool.append(task.format(topic=topic))
    return pool


def load_prompts_file(path):
    """Read prompts from a .txt (one per line) or .jsonl ({"prompt": ...} per line) file."""
    path = Path(path)
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: not valid JSON: {exc}") from exc
                text = obj.get("prompt") if isinstance(obj, dict) else None
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f'{path}:{lineno}: expected {{"prompt": "..."}}')
                prompts.append(text)
            else:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"{path}: no prompts found")
    return prompts


class PromptSource:
    """Hands out one prompt per request.

    mode="distinct": prompt i is pool[i % len(pool)] prefixed with a request tag
    that includes the cell id, so prompts never repeat within or across cells.
    mode="same": every request gets the same text (the upstream behaviour), for
    comparison.
    """

    def __init__(self, mode="distinct", pool=None, run_tag="r0"):
        if mode not in MODES:
            raise ValueError(f"prompt mode must be one of {MODES}, not {mode!r}")
        self.mode = mode
        self.custom = bool(pool)
        self.pool = list(pool) if pool else builtin_pool()
        self.run_tag = run_tag

    def prompt(self, cell, index):
        if self.mode == "same":
            # A user-supplied file's first line, else upstream's fixed prompt.
            return self.pool[0] if self.custom else SAME_PROMPT
        text = self.pool[index % len(self.pool)]
        return f"[{self.run_tag}-{cell}-{index}] {text}"

    def batch(self, cell, count):
        return [self.prompt(cell, i) for i in range(count)]
