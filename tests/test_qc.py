"""Smoke tests for `user_simulator.qc`.

Programmatic checks (D1-D4) run against fixtures inline; judges (D5/D6) are
exercised via `FakeLLM` with scripted JSON responses.
"""

from __future__ import annotations

import json

import pytest

from tests.fakes import FakeLLM
from user_simulator.data import Persona
from user_simulator.qc import score_conversation
from user_simulator.qc.checks import (
    check_profile_binding,
    check_schema,
    check_state_trajectory,
    check_structure,
)


def _persona() -> Persona:
    return Persona(
        id="p1",
        summary="I am a 39-year-old civil engineer.",
        metadata={
            "refined_summary": "I am a 39-year-old civil engineer.",
            "behavioral_metadata": {"tone_pref": "formal", "expertise_level": "expert"},
        },
    )


def _good_conv() -> dict:

    long_user = (
        "Hello, I have a detailed question about bridge design under seismic load. "
        "Specifically, I am evaluating a 120-meter cantilever span and would like "
        "guidance on appropriate damping coefficients and load factors. "
    ) * 3
    long_asst = (
        "For a 120-meter cantilever span the typical approach is to combine "
        "modal analysis with a response-spectrum method, applying load factors "
        "consistent with AASHTO LRFD and adjusting damping per the local code. "
    ) * 3
    return {
        "persona_id": "p1",
        "prompt_id": "scn_0",
        "conversation": [
            {"role": "user", "content": long_user},
            {"role": "assistant", "content": long_asst},
            {"role": "user", "content": long_user + " What about column ductility?"},
            {"role": "assistant", "content": long_asst + " Column ductility..."},
        ],
        "user_state_trajectory": [
            {"turn": 1, "user_state": "..."},
            {"turn": 2, "user_state": "..."},
            {"turn": 3, "user_state": "..."},
        ],
        "num_turns": 4,
        "ablation": "full",
        "profile_summary": "I am a 39-year-old civil engineer.",
        "behavioral_metadata": {"tone_pref": "formal"},
    }


def test_d1_schema_pass():
    ok, notes = check_schema(_good_conv())
    assert ok and not notes


def test_d1_missing_field():
    conv = _good_conv()
    del conv["num_turns"]
    ok, notes = check_schema(conv)
    assert not ok
    assert any("num_turns" in n for n in notes)


def test_d1_wrong_type():
    conv = _good_conv()
    conv["conversation"] = "not a list"
    ok, notes = check_schema(conv)
    assert not ok


def test_d2_pass():
    ok, notes = check_structure(_good_conv())
    assert ok, notes


def test_d2_too_few_turns():
    conv = _good_conv()
    conv["num_turns"] = 2
    conv["conversation"] = conv["conversation"][:2]
    ok, _ = check_structure(conv)
    assert not ok


def test_d2_consecutive_same_role():
    conv = _good_conv()
    conv["conversation"][1]["role"] = "user"
    ok, notes = check_structure(conv)
    assert not ok
    assert any("consecutive" in n for n in notes)


def test_d2_empty_content():
    conv = _good_conv()
    conv["conversation"][0]["content"] = "  "
    ok, _ = check_structure(conv)
    assert not ok


def test_d2_missing_role():
    conv = _good_conv()

    for m in conv["conversation"]:
        m["role"] = "user"
    ok, _ = check_structure(conv)
    assert not ok


def test_d3_pass_for_full_ablation():
    ok, _ = check_state_trajectory(_good_conv())
    assert ok


def test_d3_skipped_for_no_state_ablation():
    conv = _good_conv()
    conv["ablation"] = "no_state"
    conv["user_state_trajectory"] = []
    ok, _ = check_state_trajectory(conv)
    assert ok


def test_d3_too_short_trajectory():
    conv = _good_conv()
    conv["user_state_trajectory"] = []
    ok, notes = check_state_trajectory(conv)
    assert not ok
    assert any("user_state_trajectory" in n for n in notes)


def test_d4_pass():
    ok, _ = check_profile_binding(_good_conv(), _persona())
    assert ok


def test_d4_missing_persona():
    ok, notes = check_profile_binding(_good_conv(), None)
    assert not ok
    assert any("not resolvable" in n for n in notes)


def test_d4_summary_mismatch():
    conv = _good_conv()
    conv["profile_summary"] = "I am a 25-year-old chef."
    ok, notes = check_profile_binding(conv, _persona())
    assert not ok
    assert any("does not match" in n for n in notes)


@pytest.mark.asyncio
async def test_score_conversation_tier_a_with_judges():
    fake = FakeLLM()

    fake.queue("qc_consistency", json.dumps({"score": 5, "reason": "all good"}))
    fake.queue(
        "qc_conflict",
        json.dumps({"label": "no_contradiction", "offending_turn": None, "reason": "no conflicts"}),
    )
    result = await score_conversation(_good_conv(), _persona(), fake)
    assert result.qc_pass is True
    assert result.tier == "A"
    assert result.d5_persona_consistency == 5
    assert result.d6_conflict == "no_contradiction"
    assert result.failed_dims == []


@pytest.mark.asyncio
async def test_score_conversation_tier_b_borderline():
    fake = FakeLLM()

    fake.queue("qc_consistency", json.dumps({"score": 3, "reason": "borderline"}))
    fake.queue(
        "qc_conflict",
        json.dumps({"label": "no_contradiction", "offending_turn": None, "reason": "no conflicts"}),
    )
    result = await score_conversation(_good_conv(), _persona(), fake)
    assert result.tier == "B"
    assert result.qc_pass is False


@pytest.mark.asyncio
async def test_score_conversation_tier_c_contradicts():
    fake = FakeLLM()
    fake.queue("qc_consistency", json.dumps({"score": 4, "reason": "ok"}))
    fake.queue(
        "qc_conflict",
        json.dumps({"label": "contradicts", "offending_turn": 1, "reason": "claims wrong age"}),
    )
    result = await score_conversation(_good_conv(), _persona(), fake)
    assert result.tier == "C"
    assert "D6" in result.failed_dims


@pytest.mark.asyncio
async def test_score_conversation_skip_judges_tier_a():
    """Programmatic-only mode: pass on D1-D4 → tier A, no LLM calls."""
    result = await score_conversation(_good_conv(), _persona(), llm=None, skip_judges=True)
    assert result.tier == "A"
    assert result.qc_pass is True
    assert result.d5_persona_consistency is None
    assert result.d6_conflict is None


@pytest.mark.asyncio
async def test_score_conversation_skip_judges_tier_c_when_programmatic_fails():
    conv = _good_conv()
    conv["num_turns"] = 1
    conv["conversation"] = conv["conversation"][:1]
    result = await score_conversation(conv, _persona(), llm=None, skip_judges=True)
    assert result.tier == "C"
    assert "D2" in result.failed_dims


@pytest.mark.asyncio
async def test_judges_skipped_when_programmatic_fails():
    """If D1-D4 fail, we don't waste LLM calls on D5/D6."""
    fake = FakeLLM()
    conv = _good_conv()
    del conv["persona_id"]
    conv["persona_id"] = ""

    result = await score_conversation(conv, _persona(), fake)
    assert result.tier == "C"
    assert result.d5_persona_consistency is None
    assert fake.calls == 0
