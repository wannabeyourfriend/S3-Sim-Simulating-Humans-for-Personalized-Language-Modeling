"""LLM-judge wrappers for D5 (persona consistency) and D6 (profile conflict)."""
from __future__ import annotations

import json
import logging
from typing import Literal

from user_simulator.data import LLM, Persona, fmt_conversation
from user_simulator.prompts import load_prompt, render

logger = logging.getLogger(__name__)

_JUDGE_CONSISTENCY = load_prompt("judge_persona_consistency")
_JUDGE_CONFLICT = load_prompt("judge_profile_conflict")

ConflictLabel = Literal["no_contradiction", "contradicts", "unclear"]


def _format_behavior_metadata(persona: Persona | None) -> str:
    if persona is None or not persona.behavioral_metadata:
        return "N/A"
    return json.dumps(persona.behavioral_metadata, indent=2, ensure_ascii=False)


def _format_profile_summary(conv: dict, persona: Persona | None) -> str:
    # Prefer the conv's stored profile_summary (matches what was used at generation
    # time). Fall back to persona's refined_summary when missing.
    return conv.get("profile_summary") or (
        persona.refined_summary if persona else ""
    ) or ""


async def judge_persona_consistency(conv: dict, persona: Persona | None,
                                    llm: LLM) -> tuple[int | None, str | None]:
    """D5: returns (score 1–5, reason) or (None, None) on parse failure."""
    prompt = render(
        _JUDGE_CONSISTENCY,
        profile_summary=_format_profile_summary(conv, persona),
        behavior_metadata=_format_behavior_metadata(persona),
        conversation=fmt_conversation(conv.get("conversation", [])),
    )
    try:
        data = await llm.chat_json(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": "Score now."}],
            temperature=0.0, max_tokens=300,
            call_type="qc_consistency",
        )
    except Exception as e:
        logger.warning("D5 judge call failed for %s/%s: %s",
                       conv.get("persona_id"), conv.get("prompt_id"), e)
        return (None, None)

    raw_score = data.get("score") if isinstance(data, dict) else None
    reason = (data or {}).get("reason") if isinstance(data, dict) else None
    if not isinstance(raw_score, int) or not (1 <= raw_score <= 5):
        return (None, reason if isinstance(reason, str) else None)
    return (raw_score, reason if isinstance(reason, str) else None)


async def judge_profile_conflict(conv: dict, persona: Persona | None,
                                 llm: LLM) -> tuple[ConflictLabel | None, int | None]:
    """D6: returns (label, offending_turn) or (None, None) on parse failure."""
    prompt = render(
        _JUDGE_CONFLICT,
        profile_summary=_format_profile_summary(conv, persona),
        behavior_metadata=_format_behavior_metadata(persona),
        conversation=fmt_conversation(conv.get("conversation", [])),
    )
    try:
        data = await llm.chat_json(
            [{"role": "system", "content": prompt},
             {"role": "user", "content": "Score now."}],
            temperature=0.0, max_tokens=300,
            call_type="qc_conflict",
        )
    except Exception as e:
        logger.warning("D6 judge call failed for %s/%s: %s",
                       conv.get("persona_id"), conv.get("prompt_id"), e)
        return (None, None)

    label = (data or {}).get("label") if isinstance(data, dict) else None
    if label not in ("no_contradiction", "contradicts", "unclear"):
        return (None, None)
    offending = (data or {}).get("offending_turn")
    if not isinstance(offending, int):
        offending = None
    return (label, offending)
