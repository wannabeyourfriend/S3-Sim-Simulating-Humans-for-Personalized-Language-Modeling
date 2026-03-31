"""Rollout 1240 conversations from rewritten prompts JSONL.
Reads data/original_rewritten_selected_prompts_us.jsonl, loads the matching
persona for each line, and runs rollout_conversation with the rewritten prompt
as starting point. Results are streamed to disk (conversation JSON + SFT JSONL).
Usage:
    uv run python rollout_1240_real_world_queries.py
    uv run python rollout_1240_real_world_queries.py --ablation full --concurrency 80
    uv run python rollout_1240_real_world_queries.py --ablation full --concurrency 80 --max-prompts 10
    uv run python rollout_1240_real_world_queries.py --ablation full --persona-ids profile_462 --max-prompts 5
"""
import argparse, asyncio, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PROMPTS_JSONL = ROOT / "data" / "rewritten_prompts" / "original_rewritten_selected_prompts_us.jsonl"
PROFILES_DIR = ROOT / "data" / "refined_profiles" / "US"

def load_prompt_lines(path: Path, persona_ids: set[str] | None = None,
                      max_prompts: int | None = None) -> list[dict]:
    """Load prompt JSONL, optionally filter by persona and limit count."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            d = json.loads(raw)
            if persona_ids and d["persona_id"] not in persona_ids:
                continue
            lines.append(d)
    if max_prompts:
        # per-persona limit
        from collections import defaultdict
        per_persona = defaultdict(list)
        for d in lines:
            per_persona[d["persona_id"]].append(d)
        lines = []
        for pid in sorted(per_persona):
            lines.extend(per_persona[pid][:max_prompts])
    return lines


async def main(ablation: str, concurrency: int, max_turns: int, min_turns: int,
               persona_ids: list[str] | None, max_prompts: int | None,
               output_dir: str | None = None):
    # Lazy imports so argparse --help is fast
    from user_simulator.data import LLM, SIM_MODEL, load_personas, save_json, CONV_DIR, SFT_DIR
    from user_simulator.ablation import AblationConfig
    from user_simulator.simulator import rollout_conversation
    from user_simulator.oracle import build_sft_system_prompt

    config = AblationConfig.from_name(ablation)
    if output_dir:
        CONV_DIR = Path(output_dir) / "conversations"
        SFT_DIR = Path(output_dir) / "sft"

    # ── Load personas into a dict ──
    personas_list = load_personas(PROFILES_DIR)
    persona_map: dict[str, object] = {p.id: p for p in personas_list}
    logger.info("Loaded %d personas from %s", len(persona_map), PROFILES_DIR)

    # ── Load prompt lines ──
    id_set = set(persona_ids) if persona_ids else None
    prompt_lines = load_prompt_lines(PROMPTS_JSONL, persona_ids=id_set, max_prompts=max_prompts)
    logger.info("Loaded %d prompt lines (filter: persona_ids=%s, max_prompts=%s)",
                len(prompt_lines), persona_ids, max_prompts)

    # Skip lines whose persona is missing
    tasks_spec = []
    for d in prompt_lines:
        persona = persona_map.get(d["persona_id"])
        if not persona:
            logger.warning("Persona %s not found, skipping", d["persona_id"])
            continue
        tasks_spec.append((persona, d))

    logger.info("Pipeline [%s]: %d rollouts, concurrency=%d, %d-%d turns",
                config.name, len(tasks_spec), concurrency, min_turns, max_turns)

    llm = LLM(model=SIM_MODEL, max_concurrent=concurrency)
    conv_dir = CONV_DIR / config.name
    conv_dir.mkdir(parents=True, exist_ok=True)
    sft_path = SFT_DIR / f"train_{config.name}.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_lock = asyncio.Lock()

    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(tasks_spec)}

    def _build_sft_instance(session: dict) -> dict | None:
        conversation = session.get("conversation", [])
        if not conversation:
            return None
        system_msg = build_sft_system_prompt(
            profile_summary=session.get("profile_summary", ""),
            behavior_metadata=json.dumps(session.get("behavioral_metadata", {}),
                                         indent=2, ensure_ascii=False)
                if session.get("behavioral_metadata") else "",
            include_profile=config.sft_include_profile,
        )
        messages = [{"role": "system", "content": system_msg}]
        for msg in conversation:
            if msg["role"] in ("user", "assistant"):
                messages.append({"role": msg["role"], "content": msg["content"]})
        return {
            "messages": messages,
            "metadata": {
                "persona_id": session.get("persona_id", ""),
                "scenario_id": session.get("prompt_id", ""),
                "num_turns": session.get("num_turns", 0),
                "termination": session.get("termination", ""),
                "ablation": config.name,
            },
        }

    async def rollout_one(persona, prompt_data, sft_file):
        prompt_id = prompt_data.get("prompt_id", "unknown")
        initial_msg = prompt_data.get("rewritten", "")
        if not initial_msg:
            counter["skipped"] += 1
            return

        safe_id = prompt_id.replace("/", "_").replace("\\", "_")
        conv_path = conv_dir / persona.id / f"{safe_id}.json"
        if conv_path.exists():
            counter["skipped"] += 1
            return

        try:
            session = await rollout_conversation(
                persona, initial_msg, prompt_id,
                llm, max_turns=max_turns, min_turns=min_turns,
                config=config,
            )
            session["profile_summary"] = persona.refined_summary
            session["behavioral_metadata"] = persona.behavioral_metadata

            conv_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(session, conv_path)

            sft_instance = _build_sft_instance(session)
            if sft_instance:
                async with sft_lock:
                    sft_file.write(json.dumps(sft_instance, ensure_ascii=False) + "\n")
                    sft_file.flush()

            counter["done"] += 1
            done = counter["done"]
            if done % 10 == 0 or done == counter["total"]:
                logger.info("[%s] Progress: %d/%d done, %d skipped, %d failed",
                            config.name, done, counter["total"],
                            counter["skipped"], counter["failed"])
        except Exception as e:
            counter["failed"] += 1
            logger.error("[%s/%s] Failed %s: %s", config.name, persona.id, safe_id, e)

    sem = asyncio.Semaphore(concurrency)

    with open(sft_path, "w", encoding="utf-8") as sft_file:
        async def bounded(persona, prompt_data):
            async with sem:
                await rollout_one(persona, prompt_data, sft_file)

        await asyncio.gather(*[
            bounded(persona, prompt_data)
            for persona, prompt_data in tasks_spec
        ])

    logger.info("[%s] Complete: %d done, %d skipped, %d failed (of %d total)",
                config.name, counter["done"], counter["skipped"],
                counter["failed"], counter["total"])
    logger.info("[%s] Conversations → %s", config.name, conv_dir)
    logger.info("[%s] SFT data → %s", config.name, sft_path)
    logger.info("LLM stats: %s", llm.stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout conversations from rewritten prompts JSONL")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_privilege", "no_behavior", "no_state"])
    parser.add_argument("--concurrency", type=int, default=80)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--persona-ids", nargs="*", help="Filter to specific persona IDs")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Max prompts per persona")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory (default: output/)")
    args = parser.parse_args()

    asyncio.run(main(
        ablation=args.ablation,
        concurrency=args.concurrency,
        max_turns=args.max_turns,
        min_turns=args.min_turns,
        persona_ids=args.persona_ids,
        output_dir=args.output_dir,
        max_prompts=args.max_prompts,
    ))
