import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FAKE = [sys.executable, "-m", "amagi_throughput_lab.fake_server"]


@pytest.fixture
def fixtures():
    return FIXTURES


@pytest.fixture
def fake_cmd():
    """A server command for the built-in simulator (fast settings for tests)."""
    return FAKE + ["--tps", "800", "--load-s", "0.3"]
