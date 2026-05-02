"""Multi-turn conversation rollout.

Threads a user simulator (stateful or vanilla) and an assistant strategy
(oracle, oracle_profile_only, vanilla) through up to `max_turns` turns,
recording the conversation, user_state trajectory, and behavior trajectory.
"""
from __future__ import annotations

import logging

from user_simulator.ablation import AblationConfig
from user_simulator.data import LLM, Persona, fmt_conversation
from user_simulator.prompts import load_prompt, render
from user_simulator.simulator.behavior.block import _make_behavior_block
from user_simulator.simulator.behavior.selection import (
    _select_behavior_with_controller,
)
from user_simulator.simulator.persona_block import (
    _persona_behavior_metadata_str, _persona_profile_summary,
)
from user_simulator.simulator.user_turn import (
    generate_user_turn, generate_user_turn_vanilla,
)

logger = logging.getLogger(__name__)

_TMPL_ASST_ORACLE = load_prompt("assistant_oracle")
_TMPL_ASST_VANILLA = load_prompt("assistant_vanilla")
_TMPL_ASST_ORACLE_PROFILE_ONLY = load_prompt("assistant_vanilla_with_profile")

_INITIAL_STATE = (
    "# User State Report\n\n"
    "## Explicit Conversarional Context\n\n"
    "Turn index: 1.\n"
    "Whether this is a new session or continuation: new session.\n\n"
    "### Cross turn memory\nN/A\n\n"
    "### Conversation log\n"
    "Just sent my opening message. Waiting for the assistant's first response.\n\n"
    "## Implicit User Inner State\n\n"
    "### Stable state\n"
    "1. Long-term goal: {intent}\n"
    "2. Beliefs: to be determined based on assistant quality\n"
    "3. Values: accuracy, relevance, respect for my expertise\n"
    "4. Background constraints: as described in profile\n"
    "5. Stance: neutrally observing\n\n"
    "### Dynamic state\n"
    "1. Behavior mode: exploring\n"
    "2. Short-term intent: {intent}\n"
    "3. Emotion: mild curiosity about how the assistant will respond\n"
    "4. Internal tension: none yet\n\n"
    "### Evaluation of Last assistant turn\n"
    "No response received yet.\n\n"
    "### Next action plan\n"
    "Waiting for the assistant's response. Will evaluate relevance and depth."
)


def _guess_intent(prompt: str) -> str:
    p = prompt.lower()
    if "?" in p:
        if any(w in p for w in ["recommend", "suggest", "best", "should i"]):
            return "get_recommendation"
        if any(w in p for w in ["how to", "how do", "fix", "solve", "help me"]):
            return "solve_problem"
        return "seek_info"
    if any(w in p for w in ["feel", "stressed", "worried", "frustrated", "upset"]):
        return "vent"
    return "explore_topic"


async def rollout_conversation(
    persona: Persona, initial_prompt: str, prompt_id: str,
    llm: LLM, max_turns: int = 15, min_turns: int = 5,
    config: AblationConfig | None = None,
) -> dict:
    """Run a full multi-turn conversation rollout.

    User mode:  stateful (user_s3) or vanilla, per config.use_user_state.
    Assistant:  oracle (profile+state) or vanilla, per config.assistant_strategy.
    Behavior:   LLM-controller injection when config.use_behavior_injection=True.
    """
    config = config or AblationConfig()
    conversation = [{"role": "user", "content": initial_prompt}]
    us_trajectory, bh_trajectory = [], []
    termination = "max_turns"

    current_state = ""
    if config.use_user_state:
        current_state = _INITIAL_STATE.format(intent=_guess_intent(initial_prompt))

    profile_summary = _persona_profile_summary(persona)
    bm_str = _persona_behavior_metadata_str(persona)

    for turn in range(1, max_turns + 1):
        # ── Assistant turn ──
        if config.assistant_strategy == "oracle":
            asst_prompt = render(_TMPL_ASST_ORACLE,
                                profile_summary=profile_summary,
                                behavior_metadata=bm_str,
                                conversation_prefix=fmt_conversation(conversation),
                                ground_truth_user_state=current_state or "N/A")
        elif config.assistant_strategy == "oracle_profile_only":
            asst_prompt = render(_TMPL_ASST_ORACLE_PROFILE_ONLY,
                                profile_summary=profile_summary,
                                behavior_metadata=bm_str,
                                conversation_prefix=fmt_conversation(conversation))
        else:
            asst_prompt = render(_TMPL_ASST_VANILLA,
                                conversation_prefix=fmt_conversation(conversation))
        asst_response = await llm.chat(
            [{"role": "system", "content": asst_prompt},
             {"role": "user", "content": "Generate your response."}],
            temperature=config.assistant_temperature,
            max_tokens=config.assistant_max_tokens)
        conversation.append({"role": "assistant", "content": asst_response})

        if turn >= max_turns:
            break

        # ── Behavior selection ──
        behavior = None
        ctrl_src = "disabled"
        if config.use_behavior_injection:
            ctrl = await _select_behavior_with_controller(
                persona, conversation, current_state,
                turn, max_turns, bh_trajectory, llm, config=config,
            )
            behavior = ctrl["behavior"]
            ctrl_src = ctrl["controller_source"]

        behavior_block_text = ""
        if behavior:
            behavior_block_text, _, _ = _make_behavior_block(behavior, conversation)
            bh_trajectory.append({
                "turn": turn,
                "behavior": behavior.get("name", ""),
                "behavior_id": behavior.get("behavior_id", ""),
                "controller_source": ctrl_src,
                "guidance_block": behavior_block_text,
            })

        # ── User turn ──
        if config.use_user_state:
            result = await generate_user_turn(
                persona, conversation, current_state, llm, behavior=behavior,
                turn_number=turn, max_turns=max_turns, config=config)
        else:
            result = await generate_user_turn_vanilla(
                persona, conversation, llm,
                history_window=config.history_window, config=config)

        if result.get("_terminated"):
            termination = result["_terminated"]
            logger.warning("Terminated at turn %d: %s", turn, termination)
            break

        if config.use_user_state and result["user_state"]:
            current_state = result["user_state"]

        us_trajectory.append({
            "turn": turn,
            "think": result.get("think", ""),
            "user_state": current_state if config.use_user_state else "",
            "behavior": behavior.get("name", "") if behavior else "",
            "prompt_template": "user_s3" if config.use_user_state else "user_vanilla",
        })

        if result["message"]:
            conversation.append({"role": "user", "content": result["message"]})

        if result["wants_to_end"]:
            n_user = sum(1 for m in conversation if m["role"] == "user")
            if n_user >= min_turns:
                termination = "user_ended"
                break
        elif not result["message"]:
            termination = "empty_message"
            break

    return {
        "persona_id": persona.id,
        "prompt_id": prompt_id,
        "conversation": conversation,
        "user_state_trajectory": us_trajectory,
        "behavior_trajectory": bh_trajectory,
        "num_turns": sum(1 for m in conversation if m["role"] == "user"),
        "termination": termination,
        "ablation": config.name,
        "models": {
            "user_simulator": llm.model,
            "assistant": llm.model,
            "behavior_controller": llm.model,
        },
        "prompt_templates": {
            "user": "user_s3" if config.use_user_state else "user_vanilla",
            "assistant": {
                "oracle": "assistant_oracle",
                "oracle_profile_only": "assistant_vanilla_with_profile",
            }.get(config.assistant_strategy, "assistant_vanilla"),
        },
    }
