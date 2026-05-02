"""Unit tests for the behavior-block renderer.

Targets `user_simulator.simulator.behavior.block` — pure functions, no LLM.
"""

from __future__ import annotations

import pytest

from user_simulator.simulator.behavior.block import (
    _extract_bullets,
    _infer_disclosure_stage,
    _make_behavior_block,
)


def _stub_behavior(**overrides) -> dict:
    """Minimal behavior dict shaped like the YAML library entries."""
    base = {
        "behavior_id": "stub",
        "name": "Stub Behavior",
        "description": "A stub behavior for testing.",
        "tuna_mode": "Information Seeking",
        "tuna_strategy": "stub_strategy",
        "cognitive_delegation_level": "low",
        "guidance_template": (
            "**Authenticity rules:**\n"
            "- rule one\n"
            "- rule two\n"
            "- rule three\n"
            "- rule four\n"
            "- rule five\n"
            "\n"
            "**Request type selection:**\n"
            "- guide one\n"
            "- guide two\n"
            "- guide three\n"
            "- guide four\n"
            "- guide five\n"
            "\n"
            "**Internal question:** What is the user really asking?"
        ),
        "few_shot_examples": [
            {"request_type": "ask_clarify", "user_turn": "Could you say more?"},
            {"request_type": "ask_compare", "user_turn": "How does it differ?"},
            {"request_type": "ask_recap", "user_turn": "Summarise so far."},
            {"request_type": "ask_next", "user_turn": "What about edge cases?"},
            {"request_type": "ask_final", "user_turn": "Anything I'm missing?"},
            {"request_type": "ask_extra", "user_turn": "One more thing."},
        ],
    }
    base.update(overrides)
    return base


class TestInferDisclosureStage:
    def test_force_override_takes_precedence(self):
        b = _stub_behavior()
        b["simulator_control"] = {"force_disclosure_stage": "minimal"}

        conv = [{"role": "assistant"} for _ in range(10)]
        assert _infer_disclosure_stage(b, conv) == "minimal"

    def test_force_override_with_invalid_value_falls_back(self):
        b = _stub_behavior()
        b["simulator_control"] = {"force_disclosure_stage": "bogus"}

        assert _infer_disclosure_stage(b, []) == "minimal"

    def test_low_delegation_early_turn_is_minimal(self):
        b = _stub_behavior(cognitive_delegation_level="low")
        assert _infer_disclosure_stage(b, [{"role": "assistant"}]) == "minimal"

    def test_high_delegation_early_turn_is_standard(self):
        b = _stub_behavior(cognitive_delegation_level="high")
        assert _infer_disclosure_stage(b, [{"role": "assistant"}]) == "standard"

    def test_low_delegation_mid_turn_is_standard(self):
        b = _stub_behavior(cognitive_delegation_level="low")
        conv = [{"role": "assistant"}] * 3
        assert _infer_disclosure_stage(b, conv) == "standard"

    def test_high_delegation_mid_turn_is_full(self):
        b = _stub_behavior(cognitive_delegation_level="very high")
        conv = [{"role": "assistant"}] * 3
        assert _infer_disclosure_stage(b, conv) == "full"

    def test_late_turn_is_always_full(self):
        b = _stub_behavior(cognitive_delegation_level="low")
        conv = [{"role": "assistant"}] * 6
        assert _infer_disclosure_stage(b, conv) == "full"


class TestExtractBullets:
    def test_pulls_bulleted_section(self):
        tmpl = "**Authenticity rules:**\n- alpha\n- beta\n\n**Other heading:** unrelated"
        assert _extract_bullets(tmpl, "Authenticity rules") == ["alpha", "beta"]

    def test_missing_section_returns_empty_list(self):
        assert _extract_bullets("no markers here", "Authenticity rules") == []


class TestMakeBehaviorBlock:
    def test_none_behavior_returns_empty_block_and_natural_flow(self):
        block, stage, name = _make_behavior_block(None, [])
        assert block == ""
        assert stage == "none"
        assert name == "natural_flow"

    def test_behavior_with_no_template_returns_empty(self):
        block, stage, _ = _make_behavior_block({"behavior_id": "x"}, [])
        assert block == ""
        assert stage == "none"

    def test_minimal_stage_clips_examples_and_rules(self):
        b = _stub_behavior()
        b["simulator_control"] = {"force_disclosure_stage": "minimal"}
        block, stage, _ = _make_behavior_block(b, [])
        assert stage == "minimal"

        assert block.count("1. [ask_clarify]") == 1
        assert block.count("2. [ask_compare]") == 1

        assert "3. [ask_recap]" not in block

        assert "private_deliberation_focus" not in block

    def test_full_stage_includes_internal_question(self):
        b = _stub_behavior()
        b["simulator_control"] = {"force_disclosure_stage": "full"}
        block, stage, _ = _make_behavior_block(b, [])
        assert stage == "full"
        assert "<private_deliberation_focus>" in block
        assert "What is the user really asking?" in block

        assert "5. [ask_final]" in block
        assert "6. [ask_extra]" not in block

    def test_force_include_few_shot_false_strips_examples(self):
        b = _stub_behavior()
        b["simulator_control"] = {
            "force_disclosure_stage": "full",
            "force_include_few_shot": False,
        }
        block, _, _ = _make_behavior_block(b, [])
        assert "<progressive_examples>" not in block

    def test_returns_behavior_id_and_name_in_block(self):
        b = _stub_behavior(behavior_id="my_id", name="My Display Name")
        b["simulator_control"] = {"force_disclosure_stage": "standard"}
        block, _, name = _make_behavior_block(b, [])
        assert "<behavior_id>my_id</behavior_id>" in block
        assert "<behavior_name>My Display Name</behavior_name>" in block
        assert name == "My Display Name"

    @pytest.mark.parametrize(
        "stage,expected_example_count",
        [
            ("minimal", 2),
            ("standard", 3),
            ("full", 5),
        ],
    )
    def test_example_count_per_stage(self, stage, expected_example_count):
        b = _stub_behavior()
        b["simulator_control"] = {"force_disclosure_stage": stage}
        block, _, _ = _make_behavior_block(b, [])

        for i in range(1, expected_example_count + 1):
            assert f"{i}. [" in block
        assert f"{expected_example_count + 1}. [" not in block
