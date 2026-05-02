"""Select the next behavior — random sampling or LLM-controlled."""

from __future__ import annotations

import logging
import random

from user_simulator.ablation import AblationConfig
from user_simulator.data import LLM, Persona, fmt_conversation
from user_simulator.prompts import render
from user_simulator.simulator.behavior.library import (
    _BEHAVIOR_ORDER,
    _BEHAVIORS,
    _CTRL_SYSTEM_RENDERED,
    _DEFAULT_BEHAVIOR,
    _SIM_PROJECTION,
    _TMPL_CTRL_USER,
)
from user_simulator.simulator.parsing import _extract_json
from user_simulator.simulator.persona_block import (
    _persona_behavior_metadata_str,
    _persona_profile_summary,
)

logger = logging.getLogger(__name__)


def _select_behavior_random() -> dict:
    """Weighted random selection from the behavior library."""
    names = [n for n in _BEHAVIOR_ORDER if _BEHAVIORS[n].get("guidance_template")]
    if not names:
        return _DEFAULT_BEHAVIOR or {}
    default_w = _SIM_PROJECTION.get("sampling", {}).get("default_weight", 1.0)
    weights = [_BEHAVIORS[n].get("weight", default_w) for n in names]
    chosen = random.choices(names, weights=weights, k=1)[0]
    out = dict(_BEHAVIORS[chosen])
    out["behavior_id"] = chosen
    return out


async def _select_behavior_with_controller(
    persona: Persona,
    conversation: list[dict],
    current_user_state: str,
    turn_number: int,
    total_turns: int,
    previous_behaviors: list[dict],
    llm: LLM,
    config: AblationConfig | None = None,
) -> dict:
    """LLM controller picks the next behavior index."""
    config = config or AblationConfig()
    prev_text = (
        ", ".join(b.get("behavior", "") for b in previous_behaviors[-8:] if b.get("behavior"))
        or "N/A"
    )

    user_prompt = render(
        _TMPL_CTRL_USER,
        profile_summary=_persona_profile_summary(persona),
        behavior_metadata=_persona_behavior_metadata_str(persona),
        current_user_state=current_user_state or "N/A",
        conversation_prefix=fmt_conversation(conversation[-12:]) or "N/A",
        previous_behaviors=prev_text,
        turn_number=turn_number,
        total_turns=total_turns,
    )
    try:
        raw = await llm.chat(
            [
                {"role": "system", "content": _CTRL_SYSTEM_RENDERED},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.controller_temperature,
            max_tokens=config.controller_max_tokens,
            json_mode=True,
            call_type="behavior_controller",
        )
        decision = _extract_json(raw)
    except Exception as e:
        logger.warning(
            "Controller failed at turn %d: %s — raw: %s",
            turn_number,
            e,
            raw[:200] if "raw" in dir() else "N/A",
        )
        return {"behavior": _select_behavior_random(), "controller_source": "fallback"}

    idx_raw = decision.get("selected_behavior_index")
    idx = int(idx_raw) if isinstance(idx_raw, (int, str)) and str(idx_raw).isdigit() else None

    if idx is not None and 0 <= idx < len(_BEHAVIOR_ORDER):
        behavior = dict(_BEHAVIORS[_BEHAVIOR_ORDER[idx]])
        behavior["behavior_id"] = _BEHAVIOR_ORDER[idx]
        logger.debug("Controller turn %d: idx=%d → %s", turn_number, idx, behavior["behavior_id"])
    else:
        behavior = _select_behavior_random()
        logger.warning(
            "Controller turn %d: invalid idx=%s, raw decision=%s — using random: %s",
            turn_number,
            idx_raw,
            decision,
            behavior.get("behavior_id"),
        )

    ctrl = dict(behavior.get("simulator_control", {}))
    if isinstance(decision.get("include_few_shot"), bool):
        ctrl["force_include_few_shot"] = decision["include_few_shot"]
    if decision.get("disclosure_stage") in {"minimal", "standard", "full"}:
        ctrl["force_disclosure_stage"] = decision["disclosure_stage"]
    if ctrl:
        behavior["simulator_control"] = ctrl

    return {"behavior": behavior, "controller_source": "llm"}
