"""Unit tests for the user-simulator output parsing helpers.

Targets `user_simulator.simulator._parse_user_output`, `_extract_json`,
`_extract_end_signal`, `_strip_tags`. These are pure functions with no LLM
dependency — fast and deterministic.
"""
from __future__ import annotations

import json

import pytest

from user_simulator.simulator import (
    _extract_end_signal,
    _extract_json,
    _parse_user_output,
    _strip_tags,
)


# ── _extract_end_signal ───────────────────────────────────────────────────────

class TestExtractEndSignal:
    def test_leading_end_tag_strips_and_signals(self):
        msg, end = _extract_end_signal("<|End Conversation|>Goodbye then.")
        assert msg == "Goodbye then."
        assert end is True

    def test_leading_continue_tag_strips_and_does_not_signal(self):
        msg, end = _extract_end_signal("<|Continue Conversation|>Tell me more.")
        assert msg == "Tell me more."
        assert end is False

    def test_first_line_end_tag(self):
        msg, end = _extract_end_signal("<|End Conversation|>\nThanks for the help.")
        assert msg == "Thanks for the help."
        assert end is True

    def test_first_line_continue_tag(self):
        msg, end = _extract_end_signal("<|Continue Conversation|>\nAnother question.")
        assert msg == "Another question."
        assert end is False

    def test_no_tag_passes_through(self):
        msg, end = _extract_end_signal("Plain message with no signal.")
        assert msg == "Plain message with no signal."
        assert end is False

    def test_whitespace_trimmed(self):
        msg, end = _extract_end_signal("  <|End Conversation|>   bye  ")
        assert msg == "bye"
        assert end is True


# ── _strip_tags ───────────────────────────────────────────────────────────────

class TestStripTags:
    @pytest.mark.parametrize("tag", ["think", "user_state", "message", "report", "state"])
    def test_strips_known_tags(self, tag):
        text = f"prefix <{tag}>middle</{tag}> suffix"
        out = _strip_tags(text)
        assert f"<{tag}>" not in out
        assert f"</{tag}>" not in out
        assert "middle" in out

    def test_leaves_unknown_tags_intact(self):
        out = _strip_tags("<custom>kept</custom>")
        assert out == "<custom>kept</custom>"


# ── _extract_json ─────────────────────────────────────────────────────────────

class TestExtractJson:
    def test_pure_json_object(self):
        assert _extract_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}

    def test_fenced_code_block(self):
        text = 'Here is the answer:\n```json\n{"selected_behavior_index": 3}\n```\nDone.'
        assert _extract_json(text) == {"selected_behavior_index": 3}

    def test_object_embedded_in_prose(self):
        text = 'preamble {"key": "value"} trailer'
        assert _extract_json(text) == {"key": "value"}

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("not json at all")


# ── _parse_user_output: 3 fallback strategies for <user_state> ────────────────

class TestParseUserOutputUserState:
    def test_closed_user_state_tags(self):
        raw = (
            "<user_state>\n"
            "# User State Report\nIntent: explore\n"
            "</user_state>\n"
            "<message>Continue please.</message>"
        )
        out = _parse_user_output(raw)
        assert "Intent: explore" in out["user_state"]
        assert out["message"] == "Continue please."

    def test_unclosed_user_state_tag_stops_at_message(self):
        raw = (
            "<user_state>\n"
            "# User State Report\nUnclosed body content here.\n"
            "<message>The actual user message.</message>"
        )
        out = _parse_user_output(raw)
        assert "Unclosed body content here." in out["user_state"]
        assert out["message"] == "The actual user message."

    def test_no_tags_recognizes_state_report_header(self):
        # Body must be >80 chars to clear the heuristic threshold.
        body = "# User State Report\n" + "Some descriptive content. " * 20
        raw = body + "\n<message>Hello.</message>"
        out = _parse_user_output(raw)
        assert out["user_state"].startswith("# User State Report")
        assert out["message"] == "Hello."

    def test_short_body_without_tags_not_promoted(self):
        # Short body must NOT be promoted to user_state — guards against noise.
        raw = "# User State Report\nshort\n<message>hi</message>"
        out = _parse_user_output(raw)
        assert out["user_state"] == ""

    def test_empty_input_returns_empty_fields(self):
        out = _parse_user_output("")
        assert out["user_state"] == ""
        assert out["message"] == ""
        assert out["wants_to_end"] is False


# ── _parse_user_output: <think>, <message>, end-signal integration ────────────

class TestParseUserOutputThinkAndMessage:
    def test_think_block_extracted(self):
        raw = "<think>internal monologue</think><user_state>x</user_state><message>m</message>"
        out = _parse_user_output(raw)
        assert out["think"] == "internal monologue"

    def test_unclosed_message_tag_recovers_tail(self):
        raw = "<user_state>state</user_state><message>tail content"
        out = _parse_user_output(raw)
        assert out["message"] == "tail content"

    def test_message_after_user_state_close_when_no_message_tag(self):
        raw = "<user_state>state body</user_state>\nbare message after state"
        out = _parse_user_output(raw)
        assert "bare message after state" in out["message"]

    def test_end_signal_in_message_propagates(self):
        raw = "<user_state>x</user_state><message><|End Conversation|>Goodbye.</message>"
        out = _parse_user_output(raw)
        assert out["message"] == "Goodbye."
        assert out["wants_to_end"] is True

    def test_continue_signal_does_not_end(self):
        raw = "<user_state>x</user_state><message><|Continue Conversation|>More.</message>"
        out = _parse_user_output(raw)
        assert out["message"] == "More."
        assert out["wants_to_end"] is False
