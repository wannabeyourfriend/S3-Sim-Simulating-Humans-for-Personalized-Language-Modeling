"""Persona → prompt-block helpers.

Pure formatting — no I/O, no LLM. Used by user_turn, rollout, and behavior
selection to inject persona fields into prompt templates.
"""

from __future__ import annotations

import json

from user_simulator.data import Persona


def _persona_profile_summary(persona: Persona) -> str:
    return persona.refined_summary or persona.summary


def _persona_behavior_metadata_str(persona: Persona) -> str:
    bm = persona.behavioral_metadata
    return json.dumps(bm, indent=2, ensure_ascii=False) if bm else "N/A"
