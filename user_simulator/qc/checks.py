"""Programmatic QC checks (D1–D4). Pure, no LLM, fast."""

from __future__ import annotations

from user_simulator.data import Persona, count_tokens

_REQUIRED_FIELDS = (
    "persona_id",
    "prompt_id",
    "conversation",
    "num_turns",
    "ablation",
    "profile_summary",
)

_STATE_USING_ABLATIONS = {"full", "no_privilege", "no_behavior", "oracle_profile_only"}

_MIN_NUM_TURNS = 4
_MIN_TOTAL_TOKENS = 200
_MAX_TOTAL_TOKENS = 16000


def check_schema(conv: dict) -> tuple[bool, list[str]]:
    """D1: required keys present and have the right primitive shape."""
    notes: list[str] = []
    for k in _REQUIRED_FIELDS:
        if k not in conv:
            notes.append(f"missing field: {k}")
    if "conversation" in conv and not isinstance(conv["conversation"], list):
        notes.append("conversation is not a list")
    if "num_turns" in conv and not isinstance(conv["num_turns"], int):
        notes.append("num_turns is not int")
    return (not notes, notes)


def check_structure(conv: dict) -> tuple[bool, list[str]]:
    """D2: structural sanity of the conversation list."""
    notes: list[str] = []
    msgs = conv.get("conversation", []) or []
    num_turns = conv.get("num_turns", 0) or 0

    if num_turns < _MIN_NUM_TURNS:
        notes.append(f"num_turns={num_turns} < {_MIN_NUM_TURNS}")

    if not msgs:
        notes.append("empty conversation")
        return (False, notes)

    roles = [m.get("role") for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        notes.append("missing user or assistant role")

    for i in range(1, len(msgs)):
        if msgs[i].get("role") == msgs[i - 1].get("role"):
            notes.append(f"consecutive same-role turns at index {i}")
            break

    for i, m in enumerate(msgs):
        if not (m.get("content") or "").strip():
            notes.append(f"empty content at turn {i}")
            break

    total_tokens = sum(count_tokens(m.get("content", "")) for m in msgs)
    if total_tokens < _MIN_TOTAL_TOKENS:
        notes.append(f"total_tokens={total_tokens} < {_MIN_TOTAL_TOKENS}")
    if total_tokens > _MAX_TOTAL_TOKENS:
        notes.append(f"total_tokens={total_tokens} > {_MAX_TOTAL_TOKENS}")

    return (not notes, notes)


def check_state_trajectory(conv: dict) -> tuple[bool, list[str]]:
    """D3: user_state_trajectory length matches expectation for state-using ablations.

    For ablations that don't track state (e.g. 'no_state'), this is auto-pass.
    """
    notes: list[str] = []
    ablation = conv.get("ablation", "")
    if ablation not in _STATE_USING_ABLATIONS:
        return (True, notes)

    traj = conv.get("user_state_trajectory") or []
    num_turns = conv.get("num_turns", 0) or 0

    expected = max(1, num_turns - 1)
    if len(traj) < expected:
        notes.append(
            f"user_state_trajectory length {len(traj)} < expected {expected} for ablation={ablation}"
        )
    return (not notes, notes)


def check_profile_binding(conv: dict, persona: Persona | None) -> tuple[bool, list[str]]:
    """D4: persona resolves; profile_summary matches the persona's stored summary."""
    notes: list[str] = []
    persona_id = conv.get("persona_id", "")
    if not persona_id:
        notes.append("missing persona_id")
        return (False, notes)
    if persona is None:
        notes.append(f"persona {persona_id!r} not resolvable in profile store")
        return (False, notes)
    if persona.id != persona_id:
        notes.append(f"persona id mismatch: conv={persona_id!r} persona={persona.id!r}")

    expected = persona.refined_summary or persona.summary
    actual = conv.get("profile_summary", "") or ""
    if not expected:
        notes.append("persona has neither refined_summary nor summary")
    elif actual.strip() != expected.strip():
        notes.append("profile_summary does not match persona stored summary")
    return (not notes, notes)
