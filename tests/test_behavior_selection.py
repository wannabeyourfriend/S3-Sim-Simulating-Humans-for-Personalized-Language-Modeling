"""Unit tests for behavior selection.

Random selection runs against the real loaded library (deterministic via
random.seed). Controller-based selection uses FakeLLM to script the LLM
response and verifies the decision parsing + fallback paths.
"""

from __future__ import annotations

import json
import random

import pytest

from user_simulator.data import Persona
from user_simulator.simulator.behavior.library import (
    _BEHAVIOR_ORDER,
    _BEHAVIORS,
)
from user_simulator.simulator.behavior.selection import (
    _select_behavior_random,
    _select_behavior_with_controller,
)


def _make_persona() -> Persona:
    return Persona(
        id="test_persona",
        summary="A retired engineer who values precise, factual answers.",
    )


class TestRandom:
    def test_returns_a_loaded_behavior(self):
        random.seed(0)
        b = _select_behavior_random()
        assert b.get("behavior_id") in _BEHAVIORS
        assert b.get("guidance_template")

    def test_seeded_selection_is_deterministic(self):
        random.seed(42)
        first = _select_behavior_random()["behavior_id"]
        random.seed(42)
        second = _select_behavior_random()["behavior_id"]
        assert first == second

    def test_returns_a_dict_copy_not_a_reference(self):
        random.seed(0)
        b = _select_behavior_random()
        bid = b["behavior_id"]

        b["behavior_id"] = "MUTATED"
        assert _BEHAVIORS[bid].get("behavior_id") != "MUTATED"


class TestController:
    @pytest.fixture
    def persona(self) -> Persona:
        return _make_persona()

    @pytest.fixture
    def conversation(self) -> list[dict]:
        return [
            {"role": "user", "content": "How do I install nginx?"},
            {"role": "assistant", "content": "Run apt install nginx."},
        ]

    async def test_valid_index_returns_corresponding_behavior(
        self, fake_llm, persona, conversation
    ):
        target_idx = 0
        target_id = _BEHAVIOR_ORDER[target_idx]
        fake_llm.queue(
            "behavior_controller",
            json.dumps({"selected_behavior_index": target_idx}),
        )
        out = await _select_behavior_with_controller(
            persona,
            conversation,
            "current state",
            turn_number=1,
            total_turns=8,
            previous_behaviors=[],
            llm=fake_llm,
        )
        assert out["controller_source"] == "llm"
        assert out["behavior"]["behavior_id"] == target_id

    async def test_out_of_range_index_falls_back_to_random(self, fake_llm, persona, conversation):
        fake_llm.queue(
            "behavior_controller",
            json.dumps({"selected_behavior_index": 9999}),
        )
        random.seed(0)
        out = await _select_behavior_with_controller(
            persona,
            conversation,
            "current state",
            turn_number=1,
            total_turns=8,
            previous_behaviors=[],
            llm=fake_llm,
        )

        assert out["controller_source"] == "llm"
        assert out["behavior"]["behavior_id"] in _BEHAVIORS

    async def test_malformed_json_response_falls_back_to_random(
        self, fake_llm, persona, conversation
    ):
        fake_llm.queue("behavior_controller", "not valid json at all")
        random.seed(0)
        out = await _select_behavior_with_controller(
            persona,
            conversation,
            "current state",
            turn_number=1,
            total_turns=8,
            previous_behaviors=[],
            llm=fake_llm,
        )
        assert out["controller_source"] == "fallback"
        assert out["behavior"]["behavior_id"] in _BEHAVIORS

    async def test_disclosure_stage_override_propagates(self, fake_llm, persona, conversation):
        fake_llm.queue(
            "behavior_controller",
            json.dumps(
                {
                    "selected_behavior_index": 0,
                    "disclosure_stage": "full",
                    "include_few_shot": True,
                }
            ),
        )
        out = await _select_behavior_with_controller(
            persona,
            conversation,
            "current state",
            turn_number=1,
            total_turns=8,
            previous_behaviors=[],
            llm=fake_llm,
        )
        ctrl = out["behavior"].get("simulator_control", {})
        assert ctrl.get("force_disclosure_stage") == "full"
        assert ctrl.get("force_include_few_shot") is True

    async def test_invalid_disclosure_stage_is_ignored(self, fake_llm, persona, conversation):
        fake_llm.queue(
            "behavior_controller",
            json.dumps(
                {
                    "selected_behavior_index": 0,
                    "disclosure_stage": "bogus",
                }
            ),
        )
        out = await _select_behavior_with_controller(
            persona,
            conversation,
            "current state",
            turn_number=1,
            total_turns=8,
            previous_behaviors=[],
            llm=fake_llm,
        )
        ctrl = out["behavior"].get("simulator_control", {})

        assert ctrl.get("force_disclosure_stage") != "bogus"
