"""Behavior library: loads behaviors from JSONL and renders the controller catalog.

Module-level globals are populated at import time, matching the legacy
simulator.py behavior. A future refinement may make this lazy via a
`BehaviorLibrary` class + `get_library()` accessor; for now, the eager
behavior is preserved so phase 2 is a pure structural split.
"""
from __future__ import annotations

import json
from pathlib import Path

from user_simulator.prompts import load_yaml, render

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_BEHAVIORS_JSONL = _ROOT / "data" / "behavior_modes" / "behavior_modes.jsonl"

_BEHAVIOR_SPEC = load_yaml("simulator_behavior_sample")
_TMPL_CTRL_SYSTEM = _BEHAVIOR_SPEC.get("system_prompt", "")
_TMPL_CTRL_USER = _BEHAVIOR_SPEC.get("user_prompt", "")

_MODE_RANK = {
    "Meta-Conversation": 0, "Social Interaction": 1,
    "Information Seeking": 2, "Information Processing & Synthesis": 3,
    "Procedural Guidance & Execution": 4, "Content Creation & Transformation": 5,
    "Multiple (blended)": 6, "Mixed": 7,
}

# behavior_id reserved as the no-op fallback used when the controller fails
# AND the sampling pool is empty; also pulled out into _DEFAULT_BEHAVIOR.
_DEFAULT_BEHAVIOR_ID = "default_behavior"


def _load_behaviors() -> tuple[dict[str, dict], list[str], dict, dict]:
    """Load behaviors from `data/behavior_modes/behavior_modes.jsonl`.

    Returns (behaviors, order, simulator_projection, default_behavior). The
    JSONL doesn't currently carry a top-level `simulator_projection` block,
    so an empty dict is returned — selection.py falls back to a
    `default_weight=1.0` uniform sample.
    """
    behaviors: dict[str, dict] = {}
    if _BEHAVIORS_JSONL.exists():
        for line in _BEHAVIORS_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict) or not (raw.get("guidance_template") or "").strip():
                continue
            bid = (raw.get("behavior_id") or "").strip()
            if not bid:
                continue
            raw["behavior_id"] = bid
            behaviors[bid] = raw

    def _key(bid: str):
        b = behaviors[bid]
        mode = (b.get("tuna_mode") or "").strip()
        return (_MODE_RANK.get(mode, 99), mode, b.get("tuna_strategy", ""), bid)
    order = sorted(behaviors, key=_key)

    default = behaviors.get(_DEFAULT_BEHAVIOR_ID, {})
    projection: dict = {}
    return behaviors, order, projection, default


_BEHAVIORS, _BEHAVIOR_ORDER, _SIM_PROJECTION, _DEFAULT_BEHAVIOR = _load_behaviors()


def _build_behavior_catalog() -> str:
    """Indexed catalog string for the controller system prompt."""
    mode_labels = {
        "Meta-Conversation": "Mode 6: Meta-Conversation",
        "Social Interaction": "Mode 5: Social Interaction",
        "Information Seeking": "Mode 1: Information Seeking",
        "Information Processing & Synthesis": "Mode 2: Information Processing",
        "Procedural Guidance & Execution": "Mode 3: Procedural Guidance & Execution",
        "Content Creation & Transformation": "Mode 4: Content Creation & Transformation",
        "Multiple (blended)": "Compound / Blended",
        "Mixed": "Default / Mixed",
    }
    rows, cur_mode = [], None
    for idx, bid in enumerate(_BEHAVIOR_ORDER):
        b = _BEHAVIORS[bid]
        mode = (b.get("tuna_mode") or "").strip()
        if mode != cur_mode:
            cur_mode = mode
            rows.append(f"## {mode_labels.get(mode, mode or 'Other')}")
        desc = (b.get("description", "") or "").strip().replace("\n", " ")
        rows.append(
            f"[{idx}] id: {bid}\n"
            f"  name: {b.get('name', bid)}\n"
            f"  tuna_mode: {mode}\n"
            f"  tuna_strategy: {b.get('tuna_strategy', '')}\n"
            f"  cognitive_delegation_level: {b.get('cognitive_delegation_level', '')}\n"
            f"  description: {desc}"
        )
    return "\n\n".join(rows) if rows else "[0] none | Natural Conversation"


_CTRL_SYSTEM_RENDERED = render(_TMPL_CTRL_SYSTEM, behavior_catalog=_build_behavior_catalog())
