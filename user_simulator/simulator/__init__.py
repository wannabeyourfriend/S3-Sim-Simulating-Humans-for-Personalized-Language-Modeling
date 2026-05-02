"""Conversation rollout engine for S3-Sim.

Public API:
    rollout_conversation(...) — main rollout loop
    generate_user_turn(...) — stateful user turn (state + behavior)
    generate_user_turn_vanilla(...) — vanilla user turn (no state)

The internals are split across submodules:
    rollout       — main loop, intent guessing, initial-state template
    user_turn     — stateful and vanilla user-turn generators
    parsing       — output parsing for <user_state>, <message>, JSON, end-tags
    persona_block — persona → prompt-block helpers
    behavior/     — behavior library, selection, and block rendering

The legacy private names (`_TMPL_USER_S3`, `_make_behavior_block`,
`_BEHAVIORS`, etc.) are re-exported here so existing tests keep importing
from `user_simulator.simulator` directly. The next phase will migrate test
imports to the new public locations.
"""
from user_simulator.simulator.rollout import (
    rollout_conversation,
    _INITIAL_STATE,
    _TMPL_ASST_ORACLE,
    _TMPL_ASST_VANILLA,
    _TMPL_ASST_ORACLE_PROFILE_ONLY,
    _guess_intent,
)
from user_simulator.simulator.user_turn import (
    generate_user_turn,
    generate_user_turn_vanilla,
    _FORMAT_REMINDER,
    _TMPL_USER_S3,
    _TMPL_USER_VANILLA,
)
from user_simulator.simulator.parsing import (
    _parse_user_output,
    _strip_tags,
    _extract_end_signal,
    _extract_json,
)
from user_simulator.simulator.persona_block import (
    _persona_profile_summary,
    _persona_behavior_metadata_str,
)
from user_simulator.simulator.behavior import (
    _BEHAVIORS,
    _BEHAVIOR_ORDER,
    _SIM_PROJECTION,
    _DEFAULT_BEHAVIOR,
    _BEHAVIOR_SPEC,
    _TMPL_CTRL_SYSTEM,
    _TMPL_CTRL_USER,
    _CTRL_SYSTEM_RENDERED,
    _make_behavior_block,
    _infer_disclosure_stage,
    _extract_bullets,
    _select_behavior_random,
    _select_behavior_with_controller,
)

__all__ = [
    "rollout_conversation",
    "generate_user_turn",
    "generate_user_turn_vanilla",
]
