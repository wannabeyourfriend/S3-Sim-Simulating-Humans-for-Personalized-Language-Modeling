"""Oracle annotation and SFT data assembly.

An oracle agent with privileged access to the user's profile and user_state
generates personalized responses. For SFT, the system prompt includes the
user profile so the student model learns to use it at inference time.
"""

import json, logging
from pathlib import Path
from user_simulator.data import (
    Persona,
    LLM,
    count_tokens,
    fmt_conversation,
    load_json,
    ENC,
    CONV_DIR,
    SFT_DIR,
)
from user_simulator.ablation import AblationConfig
from user_simulator.prompts import load_prompt, render

from user_simulator.sft import BASE_SYSTEM_INSTRUCTION, build_sft_system_prompt

logger = logging.getLogger(__name__)

_ORACLE = load_prompt("assistant_oracle")
_ORACLE_NO_STATE = load_prompt("assistant_vanilla_with_profile")


async def annotate_turn(
    persona: Persona,
    conversation: list[dict],
    user_state: str,
    turn_idx: int,
    llm: LLM,
    config: AblationConfig | None = None,
) -> dict:
    """Oracle generates a personalized response for one assistant turn.

    The oracle has privileged access to the user's profile and (optionally)
    their internal user_state.
    """
    config = config or AblationConfig()
    prefix = fmt_conversation(conversation)

    profile_summary = persona.metadata.get("refined_summary", "") or persona.summary
    behavior_metadata = (
        json.dumps(persona.metadata.get("behavioral_metadata", {}), indent=2, ensure_ascii=False)
        if persona.metadata
        else "N/A"
    )

    if config.assistant_strategy == "oracle":
        prompt = render(
            _ORACLE,
            profile_summary=profile_summary,
            behavior_metadata=behavior_metadata,
            conversation_prefix=prefix,
            ground_truth_user_state=user_state,
        )
    else:
        prompt = render(
            _ORACLE_NO_STATE,
            profile_summary=profile_summary,
            behavior_metadata=behavior_metadata,
            conversation_prefix=prefix,
        )

    content, thinking = await llm.chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate your response."},
        ],
        temperature=0.7,
        max_tokens=4096,
        return_thinking=True,
    )
    if thinking:
        oracle_output = f"<think>\n{thinking}\n</think>\n{content}"
    else:
        oracle_output = content

    return {
        "turn": turn_idx,
        "input": prefix,
        "output": oracle_output,
        "ground_truth_user_state": user_state,
        "total_tokens": count_tokens(prefix) + count_tokens(oracle_output),
    }


async def annotate_conversation(
    persona: Persona, session: dict, llm: LLM, config: AblationConfig | None = None
) -> list[dict]:
    """Annotate all assistant turns in a conversation.

    Two modes, driven by config.assistant_strategy:
      - "oracle" / "oracle_profile_only": re-generate the assistant turn with
        privileged access (full state or profile only).
      - "vanilla": keep the original assistant response as-is.
    """
    config = config or AblationConfig()
    conversation = session["conversation"]
    us_traj = session.get("user_state_trajectory", [])
    instances = []

    us_by_turn = {s["turn"]: s["user_state"] for s in us_traj}
    use_oracle = config.assistant_strategy in ("oracle", "oracle_profile_only")

    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue
        prefix = conversation[:i]
        if not prefix:
            continue

        asst_idx = sum(1 for m in conversation[: i + 1] if m["role"] == "assistant")
        user_state = us_by_turn.get(asst_idx, us_by_turn.get(asst_idx - 1, ""))

        if use_oracle:
            inst = await annotate_turn(persona, prefix, user_state, asst_idx, llm, config=config)
        else:
            inst = {
                "turn": asst_idx,
                "input": fmt_conversation(prefix),
                "output": msg["content"],
                "ground_truth_user_state": user_state,
                "total_tokens": count_tokens(fmt_conversation(prefix))
                + count_tokens(msg["content"]),
            }
        inst["persona_id"] = persona.id
        inst["scenario_id"] = session.get("prompt_id", "")
        instances.append(inst)

    return instances


def assemble_sft(
    conversations_dir: Path,
    output_path: Path,
    include_profile: bool = True,
    max_tokens: int = 32000,
) -> list[dict]:
    """Collect conversation JSONs into TRL multi-turn SFT JSONL.

    Scans conversations_dir recursively for *.json files.  Works with both
    flat directories and nested persona_id/ subdirectories.

    Delegates SFT-line construction to `user_simulator.sft.build_sft_instance`
    so offline-assembled lines match the streaming-rollout output exactly.
    """
    from user_simulator.sft import build_sft_instance

    conversations_dir = Path(conversations_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(conversations_dir.rglob("*.json"))
    logger.info("Found %d JSON files in %s", len(json_files), conversations_dir)

    instances = []
    skipped = 0
    for conv_file in json_files:
        data = load_json(conv_file)

        ablation_name = data.get("ablation") or "full"
        try:
            config = AblationConfig.from_name(ablation_name)
        except ValueError:
            logger.warning(
                "Unknown ablation %s in %s; defaulting to 'full'", ablation_name, conv_file
            )
            config = AblationConfig.full()

        config.sft_include_profile = include_profile

        instance = build_sft_instance(data, config)
        if instance is None:
            continue

        total_tokens = sum(count_tokens(m["content"]) for m in instance["messages"])
        if total_tokens > max_tokens:
            skipped += 1
            continue
        instances.append(instance)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in instances:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Assembled %d instances (%d skipped by token limit) → %s",
        len(instances),
        skipped,
        output_path,
    )
    return instances


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Assemble SFT JSONL from conversation JSONs")
    parser.add_argument(
        "conversations_dir", type=Path, help="Directory containing conversation JSON files"
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--no-profile", action="store_true", help="Exclude user profile from system prompt"
    )
    parser.add_argument("--max-tokens", type=int, default=32000)
    args = parser.parse_args()

    assemble_sft(
        conversations_dir=args.conversations_dir,
        output_path=args.output,
        include_profile=not args.no_profile,
        max_tokens=args.max_tokens,
    )
