"""Quality-check toolkit for S³-Sim conversation artifacts.

Six dimensions:
  D1 schema validity         — programmatic
  D2 structural sanity       — programmatic
  D3 state-trajectory        — programmatic (only for state-using ablations)
  D4 persona–profile binding — programmatic
  D5 persona consistency     — LLM-judge (1–5 Likert)
  D6 profile-conflict        — LLM-judge (no_contradiction|contradicts|unclear)

Each conversation is scored into a `QCResult`; results are tiered:
  Tier A — qc_pass (default training set)
  Tier B — borderline (D5=3 or D6=unclear) — held for ablation
  Tier C — drop

The release-derivation script reads `qc_results.jsonl` and emits Tier-A and
Tier-B JSONLs separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from user_simulator.data import LLM, Persona

from .checks import check_schema, check_structure, check_state_trajectory, check_profile_binding
from .judges import judge_persona_consistency, judge_profile_conflict

Tier = Literal["A", "B", "C"]


@dataclass
class QCResult:
    persona_id: str
    scenario_id: str

    d1_schema: bool = False
    d2_structure: bool = False
    d3_state_traj: bool = False
    d4_profile_bind: bool = False

    d5_persona_consistency: int | None = None
    d5_reason: str | None = None
    d6_conflict: str | None = None
    d6_offending_turn: int | None = None

    qc_pass: bool = False
    tier: Tier = "C"

    failed_dims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _decide_tier(r: QCResult, skip_judges: bool) -> tuple[Tier, bool]:
    prog_ok = r.d1_schema and r.d2_structure and r.d3_state_traj and r.d4_profile_bind
    if not prog_ok:
        return "C", False
    if skip_judges:
        return "A", True

    d5_ok = r.d5_persona_consistency is not None and r.d5_persona_consistency >= 4
    d6_ok = r.d6_conflict == "no_contradiction"
    if d5_ok and d6_ok:
        return "A", True

    d5_borderline = r.d5_persona_consistency == 3
    d6_borderline = r.d6_conflict == "unclear"
    contradicts = r.d6_conflict == "contradicts"
    d5_low = r.d5_persona_consistency is not None and r.d5_persona_consistency <= 2
    if contradicts or d5_low:
        return "C", False
    if d5_borderline or d6_borderline:
        return "B", False
    return "C", False


async def score_conversation(
    conv: dict, persona: Persona | None, llm: LLM | None, skip_judges: bool = False
) -> QCResult:
    """Score one conversation against six dimensions.

    `persona` may be None if D4 binding cannot be checked; D4 then fails.
    `llm` may be None when skip_judges=True.
    """
    r = QCResult(
        persona_id=conv.get("persona_id", ""),
        scenario_id=conv.get("prompt_id", ""),
    )

    d1, d1_notes = check_schema(conv)
    r.d1_schema = d1
    if not d1:
        r.failed_dims.append("D1")
        r.notes.extend(d1_notes)

    d2, d2_notes = check_structure(conv)
    r.d2_structure = d2
    if not d2:
        r.failed_dims.append("D2")
        r.notes.extend(d2_notes)

    d3, d3_notes = check_state_trajectory(conv)
    r.d3_state_traj = d3
    if not d3:
        r.failed_dims.append("D3")
        r.notes.extend(d3_notes)

    d4, d4_notes = check_profile_binding(conv, persona)
    r.d4_profile_bind = d4
    if not d4:
        r.failed_dims.append("D4")
        r.notes.extend(d4_notes)

    prog_ok = d1 and d2 and d3 and d4
    if not prog_ok or skip_judges:
        r.tier, r.qc_pass = _decide_tier(r, skip_judges)
        return r

    assert llm is not None, "llm required when skip_judges=False"
    score, reason = await judge_persona_consistency(conv, persona, llm)
    r.d5_persona_consistency = score
    r.d5_reason = reason
    if score is None or score < 4:
        r.failed_dims.append("D5")

    label, offending_turn = await judge_profile_conflict(conv, persona, llm)
    r.d6_conflict = label
    r.d6_offending_turn = offending_turn
    if label != "no_contradiction":
        r.failed_dims.append("D6")

    r.tier, r.qc_pass = _decide_tier(r, skip_judges)
    return r


__all__ = ["QCResult", "score_conversation", "Tier"]
