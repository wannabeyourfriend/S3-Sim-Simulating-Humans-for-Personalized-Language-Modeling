"""User-turn generators (stateful and vanilla)."""
from __future__ import annotations

import logging

from user_simulator.ablation import AblationConfig
from user_simulator.data import LLM, Persona, fmt_conversation
from user_simulator.prompts import load_prompt, render
from user_simulator.simulator.behavior.block import _make_behavior_block
from user_simulator.simulator.parsing import (
    _extract_end_signal, _parse_user_output, _strip_tags,
)
from user_simulator.simulator.persona_block import (
    _persona_behavior_metadata_str, _persona_profile_summary,
)

logger = logging.getLogger(__name__)

_TMPL_USER_S3 = load_prompt("user_s3")
_TMPL_USER_VANILLA = load_prompt("user_vanilla")

_FORMAT_REMINDER = (
    "This is turn {turn_number_plus_1} of {max_turns} in an ONGOING session (not a new session). "
    "You MUST output exactly: <user_state>...</user_state> then <message>...</message>. "
    "Do NOT answer the question — you ARE the user. "
    "Your <user_state> MUST update Cross turn memory with what happened so far. "
    "Your new message MUST differ from previous messages and advance the conversation."
)


async def generate_user_turn(persona: Persona, conversation: list[dict],
                             previous_user_state: str, llm: LLM,
                             behavior: dict | None = None,
                             turn_number: int = 1, max_turns: int = 12,
                             config: AblationConfig | None = None) -> dict:
    """Stateful user turn: maintains <user_state> as structured working memory.

    Trailing N messages (config.recent_history_window) are passed as chat
    context; full conversation history is already encoded in
    previous_user_state, which prevents repetition from long histories while
    preserving enough recent context for coherent state updates.
    """
    config = config or AblationConfig()

    behavior_block, stage, bname = _make_behavior_block(behavior, conversation)
    prompt = render(_TMPL_USER_S3,
                    profile_summary=_persona_profile_summary(persona),
                    behavior_metadata=_persona_behavior_metadata_str(persona),
                    previous_user_state=previous_user_state,
                    behavior_block=behavior_block,
                    behavior_stage=stage,
                    behavior_name=bname)

    window = config.recent_history_window
    recent = conversation[-window:] if len(conversation) > window else conversation

    reminder = (
        f"This is turn {turn_number + 1} of {max_turns} in an ONGOING session (not a new session). "
        "You MUST output exactly: <user_state>...</user_state> then <message>...</message>. "
        "Do NOT answer the question — you ARE the user. "
        "Your <user_state> MUST update Cross turn memory with what happened so far. "
        "Your new message MUST differ from previous messages and advance the conversation."
    )
    messages = [{"role": "system", "content": prompt}] + recent + [
        {"role": "user", "content": reminder}
    ]

    # Escalate temperature on retry to break out of degenerate sampling modes
    retry_temps = config.user_retry_temps
    for attempt, temp in enumerate(retry_temps, 1):
        content = await llm.chat(messages, temperature=temp,
                                 max_tokens=config.user_max_tokens)
        logger.debug("User sim (attempt %d, T=%.1f): %s", attempt, temp, content[:300])
        result = _parse_user_output(content)
        if result["user_state"]:
            return result
        logger.warning("Empty user_state (%d/%d T=%.1f), raw: %s",
                       attempt, len(retry_temps), temp, content[:200])

    logger.error("All %d retries failed — terminating session", len(retry_temps))
    return {"message": "", "wants_to_end": True, "user_state": "", "think": "",
            "_terminated": "user_state_extraction_failed"}


async def generate_user_turn_vanilla(persona: Persona, conversation: list[dict],
                                     llm: LLM, history_window: int | None = None,
                                     config: AblationConfig | None = None) -> dict:
    """Vanilla user turn: persona + conversation history, no state tracking."""
    config = config or AblationConfig()
    window = conversation[-(history_window * 2):] if history_window else conversation
    prompt = render(_TMPL_USER_VANILLA,
                    profile_summary=_persona_profile_summary(persona),
                    behavior_metadata=_persona_behavior_metadata_str(persona),
                    conversation_history=fmt_conversation(window))
    content = await llm.chat(
        [{"role": "system", "content": prompt},
         {"role": "user", "content": "Continue the conversation as this person."}],
        temperature=config.user_temperature,
        max_tokens=1024)
    msg = _strip_tags(content.strip())
    msg, wants_to_end = _extract_end_signal(msg)
    return {"message": msg, "wants_to_end": wants_to_end, "user_state": "", "think": ""}
