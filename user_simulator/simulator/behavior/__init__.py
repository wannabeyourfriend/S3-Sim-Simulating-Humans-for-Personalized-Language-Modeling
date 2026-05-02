"""Behavior library, selection, and block rendering.

The behavior subsystem provides persona-conditioned guidance that gets
injected into the user-simulator prompt at each turn. There are three
concerns:

  * library    — load YAML behaviors and the controller catalog
  * selection  — pick which behavior fires this turn (random or LLM)
  * block      — render the selected behavior into the XML guidance block
"""
from user_simulator.simulator.behavior.library import (
    _BEHAVIORS,
    _BEHAVIOR_ORDER,
    _SIM_PROJECTION,
    _DEFAULT_BEHAVIOR,
    _BEHAVIOR_SPEC,
    _TMPL_CTRL_SYSTEM,
    _TMPL_CTRL_USER,
    _CTRL_SYSTEM_RENDERED,
    _MODE_RANK,
    _load_behaviors,
    _build_behavior_catalog,
)
from user_simulator.simulator.behavior.block import (
    _make_behavior_block,
    _infer_disclosure_stage,
    _extract_bullets,
)
from user_simulator.simulator.behavior.selection import (
    _select_behavior_random,
    _select_behavior_with_controller,
)
