"""Render a selected behavior into the <behavior_injection> XML block.

Stages:
  * minimal   — terse: 3 request types, 2 rules. Used for early turns or low
                cognitive-delegation behaviors.
  * standard  — 5 / 4. Default for most turns.
  * full      — 99 / 99 (= no clipping) + internal question. Used for late
                turns or high-delegation behaviors.

The disclosure stage is normally inferred from the conversation length and
the behavior's cognitive_delegation_level, but can be force-overridden via
`behavior["simulator_control"]["force_disclosure_stage"]`.
"""
from __future__ import annotations

import re


def _infer_disclosure_stage(behavior: dict, conversation: list[dict]) -> str:
    forced = behavior.get("simulator_control", {}).get("force_disclosure_stage")
    if forced in {"minimal", "standard", "full"}:
        return forced
    n_asst = sum(1 for m in conversation if m.get("role") == "assistant")
    delegation = (behavior.get("cognitive_delegation_level") or "").lower()
    high = "very high" in delegation or "high" in delegation
    if n_asst <= 1:
        return "standard" if high else "minimal"
    if n_asst <= 4:
        return "full" if high else "standard"
    return "full"


def _extract_bullets(template: str, title: str) -> list[str]:
    m = re.search(rf"\*\*{re.escape(title)}:\*\*(.*?)(?:\n\s*\*\*|$)", template, re.DOTALL)
    return [ln.strip()[2:].strip() for ln in (m.group(1) if m else "").splitlines()
            if ln.strip().startswith("- ")]


def _make_behavior_block(behavior: dict | None, conversation: list[dict]) -> tuple[str, str, str]:
    """Build <behavior_injection> XML block. Returns (block, stage, name)."""
    if not behavior or not (behavior.get("guidance_template") or "").strip():
        return "", "none", "natural_flow"

    template = behavior["guidance_template"]
    stage = _infer_disclosure_stage(behavior, conversation)
    bid = behavior.get("behavior_id", "unknown")
    bname = behavior.get("name", bid)

    clip = {"minimal": (3, 2), "standard": (5, 4), "full": (99, 99)}[stage]
    request_types = [
        it.get("request_type", "").strip()
        for it in (behavior.get("few_shot_examples") or [])
        if isinstance(it, dict) and it.get("request_type", "").strip()
    ][:clip[0]]
    rules = _extract_bullets(template, "Authenticity rules")[:clip[1]]
    guidance = _extract_bullets(template, "Request type selection")[:clip[1]]

    iq_m = re.search(r"\*\*Internal question:\*\*\s*(.+)", template)
    internal_q = iq_m.group(1).strip() if iq_m and stage == "full" else ""

    examples = behavior.get("few_shot_examples") or []
    force_fs = behavior.get("simulator_control", {}).get("force_include_few_shot")
    if force_fs is False:
        examples = []
    elif stage == "minimal":
        examples = examples[:2]
    elif stage == "standard":
        examples = examples[:3]
    else:
        examples = examples[:5]
    ex_lines = []
    for i, it in enumerate(examples, 1):
        if isinstance(it, dict) and it.get("user_turn"):
            ex_lines.append(f"{i}. [{it.get('request_type', '?')}] {it['user_turn'].strip()}")

    lines = [
        "<behavior_injection>",
        f"<behavior_id>{bid}</behavior_id>",
        f"<behavior_name>{bname}</behavior_name>",
        f"<behavior_mode>{behavior.get('tuna_mode', '')}</behavior_mode>",
        f"<behavior_strategy>{behavior.get('tuna_strategy', '')}</behavior_strategy>",
        f"<cognitive_delegation_level>{behavior.get('cognitive_delegation_level', '')}</cognitive_delegation_level>",
        f"<disclosure_stage>{stage}</disclosure_stage>",
        f"<public_intent>\n{(behavior.get('description') or '').strip()}\n</public_intent>",
    ]
    if request_types:
        lines.append(f"<public_request_types>\n{chr(10).join('- ' + t for t in request_types)}\n</public_request_types>")
    if guidance:
        lines.append(f"<public_selection_guidance>\n{chr(10).join('- ' + g for g in guidance)}\n</public_selection_guidance>")
    if rules:
        lines.append(f"<public_authenticity_rules>\n{chr(10).join('- ' + r for r in rules)}\n</public_authenticity_rules>")
    if ex_lines:
        lines.append(f"<progressive_examples>\n{chr(10).join(ex_lines)}\n</progressive_examples>")
    if internal_q:
        lines.append(f"<private_deliberation_focus>\n{internal_q}\n</private_deliberation_focus>")
    lines.append("</behavior_injection>")
    return "\n".join(lines), stage, bname
