"""Unit tests for `user_simulator.sft.build_sft_instance`.

Locks down the byte shape of SFT JSONL lines so the dedupe of the three
pre-refactor copies (run_rollout.py, run_deep_scenario_rollout.py,
pipeline.py) is verifiable.
"""

from __future__ import annotations

import json

import pytest

from user_simulator.ablation import AblationConfig
from user_simulator.sft import build_sft_instance


def _basic_session(**overrides) -> dict:
    base = {
        "persona_id": "profile_259",
        "prompt_id": "p_42",
        "conversation": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        "num_turns": 1,
        "termination": "max_turns",
        "profile_summary": "Test persona summary.",
        "behavioral_metadata": {"trait": "curious"},
    }
    base.update(overrides)
    return base


class TestRolloutShape:
    """Mirrors run_rollout.py's pre-refactor SFT shape."""

    def test_full_ablation_includes_profile_and_behavior(self):
        config = AblationConfig.full()
        out = build_sft_instance(_basic_session(), config)
        assert out is not None
        sysmsg = out["messages"][0]["content"]
        assert "<user_profile>" in sysmsg
        assert "Test persona summary." in sysmsg
        assert "<behavior_metadata>" in sysmsg

    def test_no_privilege_strips_profile_and_behavior(self):
        config = AblationConfig.no_privilege()
        out = build_sft_instance(_basic_session(), config)
        sysmsg = out["messages"][0]["content"]
        assert "<user_profile>" not in sysmsg
        assert "<behavior_metadata>" not in sysmsg

    def test_metadata_keys_match_legacy_run_rollout_order(self):
        config = AblationConfig.full()
        out = build_sft_instance(_basic_session(), config)

        assert list(out["metadata"].keys()) == [
            "persona_id",
            "scenario_id",
            "num_turns",
            "termination",
            "ablation",
        ]
        assert out["metadata"]["ablation"] == "full"

    def test_user_and_assistant_messages_passed_through(self):
        config = AblationConfig.full()
        out = build_sft_instance(_basic_session(), config)
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert out["messages"][1]["content"] == "hello"
        assert out["messages"][2]["content"] == "hi there"

    def test_non_user_assistant_messages_filtered(self):
        session = _basic_session()
        session["conversation"] = [
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "should be dropped"},
            {"role": "assistant", "content": "ok"},
        ]
        out = build_sft_instance(session, AblationConfig.full())
        roles = [m["role"] for m in out["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_empty_conversation_returns_none(self):
        session = _basic_session()
        session["conversation"] = []
        assert build_sft_instance(session, AblationConfig.full()) is None


class TestDeepScenarioShape:
    """Mirrors run_deep_scenario_rollout.py's pre-refactor SFT shape."""

    def test_scenario_category_included_when_present(self):
        session = _basic_session(
            scenario_category="career_advice",
            source="deep_scenario",
        )
        out = build_sft_instance(session, AblationConfig.full())

        assert list(out["metadata"].keys()) == [
            "persona_id",
            "scenario_id",
            "scenario_category",
            "num_turns",
            "termination",
            "ablation",
            "source",
        ]
        assert out["metadata"]["scenario_category"] == "career_advice"
        assert out["metadata"]["source"] == "deep_scenario"

    def test_empty_scenario_category_still_included(self):

        session = _basic_session(
            scenario_category="",
            source="deep_scenario",
        )
        out = build_sft_instance(session, AblationConfig.full())
        assert "scenario_category" in out["metadata"]
        assert out["metadata"]["scenario_category"] == ""

    def test_source_omitted_when_session_has_none(self):

        session = _basic_session()
        out = build_sft_instance(session, AblationConfig.full())
        assert "source" not in out["metadata"]


class TestBehaviorMetadataSerialisation:
    def test_dict_metadata_pretty_printed_into_system_prompt(self):
        session = _basic_session(behavioral_metadata={"a": 1, "b": [2, 3]})
        out = build_sft_instance(session, AblationConfig.full())

        sysmsg = out["messages"][0]["content"]
        assert json.dumps({"a": 1, "b": [2, 3]}, indent=2) in sysmsg

    def test_empty_metadata_omits_block(self):
        session = _basic_session(behavioral_metadata={})
        out = build_sft_instance(session, AblationConfig.full())
        assert "<behavior_metadata>" not in out["messages"][0]["content"]


@pytest.mark.parametrize(
    "factory,expects_profile",
    [
        (AblationConfig.full, True),
        (AblationConfig.no_privilege, False),
        (AblationConfig.no_behavior, True),
        (AblationConfig.no_state, True),
        (AblationConfig.oracle_profile_only, True),
    ],
)
def test_sft_include_profile_per_ablation(factory, expects_profile):
    out = build_sft_instance(_basic_session(), factory())
    sysmsg = out["messages"][0]["content"]
    if expects_profile:
        assert "<user_profile>" in sysmsg
    else:
        assert "<user_profile>" not in sysmsg
