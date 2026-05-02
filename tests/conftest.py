"""Shared pytest fixtures for the S³-Sim test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fakes import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Empty FakeLLM. Tests queue responses via `fake_llm.queue(call_type, content)`."""
    return FakeLLM()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Per-test output directory. Replaces hardcoded ROOT/output writes."""
    out = tmp_path / "output"
    (out / "conversations").mkdir(parents=True)
    (out / "sft").mkdir(parents=True)
    return out
