"""Rollout deeply-personal conversations from per-persona scenarios.

Instead of reusing prompts from an existing dataset (see run_rollout.py), this
script constructs scenarios on-the-fly with the
`simulator_lifelong_scenario_constructor` prompt for each persona, then rolls
out a conversation per scenario.

Scenarios are cached per persona at data/deep_scenarios/{persona_id}.json so
reruns skip the construction step.

Usage:
    uv run python run_deep_scenario_rollout.py
    uv run python run_deep_scenario_rollout.py --ablation full --concurrency 40
    uv run python run_deep_scenario_rollout.py --persona-ids profile_259 --max-scenarios 5
"""

import argparse, asyncio, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILES = ROOT / "data" / "filterd_refined_profiles" / "summary_refined_profiles_us.jsonl"
SCENARIOS_DIR = ROOT / "data" / "deep_scenarios"

_CONSTRUCTOR_PREFIX = {
    "simulator_lifelong_scenario_constructor": "lifelong",
    "simulator_highfreq_scenario_constructor": "highfreq",
    "simulator_affective_scenario_constructor": "affective",
    "simulator_concerning_scenario_constructor": "concerning",
}


def _ctor_prefix(constructor: str) -> str:
    return _CONSTRUCTOR_PREFIX.get(
        constructor, constructor.replace("simulator_", "").replace("_scenario_constructor", "")
    )


async def construct_scenarios(
    persona, llm, constructor_tmpl: str, config, ctor_prefix: str
) -> list[dict]:
    """Call the scenario constructor LLM for one persona, return list of scenarios.

    Each scenario: {scenario_id, context_note, category, initial_prompt}.
    `scenario_id` is namespaced by `ctor_prefix` so slices from different
    constructors don't collide on (persona_id, scenario_id) keys downstream.
    """
    from user_simulator.prompts import render

    profile_summary = persona.refined_summary or persona.summary
    behavior_metadata = (
        json.dumps(persona.behavioral_metadata, indent=2, ensure_ascii=False)
        if persona.behavioral_metadata
        else "N/A"
    )

    prompt = render(
        constructor_tmpl,
        profile_summary=profile_summary,
        profile_block=profile_summary,
        behavior_metadata=behavior_metadata,
        persona_id=persona.id,
    )

    data = await llm.chat_json(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the scenarios JSON now."},
        ],
        temperature=config.scenario_constructor_temperature,
        max_tokens=config.scenario_constructor_max_tokens,
    )
    scenarios = data.get("scenarios", []) if isinstance(data, dict) else []

    for i, s in enumerate(scenarios):
        sid = s.get("scenario_id") or f"{persona.id}_scenario_{i}"
        sid = sid.replace("{persona_id}", persona.id)
        if not sid.startswith(f"{ctor_prefix}_"):
            sid = f"{ctor_prefix}_{sid}"
        s["scenario_id"] = sid
    return scenarios


async def get_or_build_scenarios(
    persona,
    llm,
    constructor_tmpl: str,
    config,
    cache_dir: Path,
    ctor_prefix: str,
    force: bool = False,
) -> list[dict]:

    cache_path = cache_dir / f"{persona.id}__{ctor_prefix}.json"
    legacy_path = cache_dir / f"{persona.id}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    if legacy_path.exists() and not force and ctor_prefix == "lifelong":
        return json.loads(legacy_path.read_text(encoding="utf-8"))
    scenarios = await construct_scenarios(persona, llm, constructor_tmpl, config, ctor_prefix)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(scenarios, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Constructed %d scenarios for %s (%s)", len(scenarios), persona.id, ctor_prefix)
    return scenarios


async def main(
    ablation: str,
    concurrency: int,
    max_turns: int,
    min_turns: int,
    persona_ids: list[str] | None,
    max_scenarios: int | None,
    output_dir: str | None,
    constructor: str,
    force_reconstruct: bool,
    profiles_path: str | None,
):
    from user_simulator.data import LLM, SIM_MODEL, load_personas, save_json, CONV_DIR, SFT_DIR
    from user_simulator.ablation import AblationConfig
    from user_simulator.simulator import rollout_conversation
    from user_simulator.sft import build_sft_instance
    from user_simulator.prompts import load_prompt

    config = AblationConfig.from_name(ablation)
    if output_dir:
        CONV_DIR = Path(output_dir) / "conversations"
        SFT_DIR = Path(output_dir) / "sft"

    constructor_tmpl = load_prompt(constructor)

    personas_list = load_personas(Path(profiles_path) if profiles_path else DEFAULT_PROFILES)
    if persona_ids:
        id_set = set(persona_ids)
        personas_list = [p for p in personas_list if p.id in id_set]
    logger.info("Processing %d personas", len(personas_list))

    llm = LLM(model=SIM_MODEL, max_concurrent=concurrency)

    ctor_prefix = _ctor_prefix(constructor)
    run_tag = f"deep_{config.name}"
    conv_dir = CONV_DIR / run_tag
    conv_dir.mkdir(parents=True, exist_ok=True)
    sft_path = SFT_DIR / f"train_{run_tag}.jsonl"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_lock = asyncio.Lock()

    logger.info(
        "Phase 1: constructing scenarios (cache: %s, prefix=%s)", SCENARIOS_DIR, ctor_prefix
    )
    scenario_tasks = [
        get_or_build_scenarios(
            p,
            llm,
            constructor_tmpl,
            config,
            SCENARIOS_DIR,
            ctor_prefix=ctor_prefix,
            force=force_reconstruct,
        )
        for p in personas_list
    ]
    all_scenarios_by_persona = await asyncio.gather(*scenario_tasks)

    tasks_spec = []
    for persona, scenarios in zip(personas_list, all_scenarios_by_persona):
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]
        for s in scenarios:
            tasks_spec.append((persona, s))

    logger.info(
        "Phase 2 [%s]: %d rollouts, concurrency=%d, %d-%d turns",
        config.name,
        len(tasks_spec),
        concurrency,
        min_turns,
        max_turns,
    )

    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(tasks_spec)}

    async def rollout_one(persona, scenario, sft_file):
        scenario_id = scenario.get("scenario_id", "unknown")
        initial_msg = scenario.get("initial_prompt", "")
        if not initial_msg:
            counter["skipped"] += 1
            return

        safe_id = scenario_id.replace("/", "_").replace("\\", "_")
        conv_path = conv_dir / persona.id / f"{safe_id}.json"
        if conv_path.exists():
            counter["skipped"] += 1
            return

        try:
            session = await rollout_conversation(
                persona,
                initial_msg,
                scenario_id,
                llm,
                max_turns=max_turns,
                min_turns=min_turns,
                config=config,
            )
            session["profile_summary"] = persona.refined_summary
            session["behavioral_metadata"] = persona.behavioral_metadata
            session["scenario_category"] = scenario.get("category", "")
            session["scenario_context_note"] = scenario.get("context_note", "")
            session["initial_prompt"] = initial_msg
            session["source"] = "deep_scenario"

            conv_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(session, conv_path)

            sft_instance = build_sft_instance(session, config)
            if sft_instance:
                async with sft_lock:
                    sft_file.write(json.dumps(sft_instance, ensure_ascii=False) + "\n")
                    sft_file.flush()

            counter["done"] += 1
            done = counter["done"]
            if done % 10 == 0 or done == counter["total"]:
                logger.info(
                    "[%s] Progress: %d/%d done, %d skipped, %d failed",
                    run_tag,
                    done,
                    counter["total"],
                    counter["skipped"],
                    counter["failed"],
                )
        except Exception as e:
            counter["failed"] += 1
            logger.error(
                "[%s/%s] Failed %s: %s: %s", run_tag, persona.id, safe_id, type(e).__name__, e
            )

    sem = asyncio.Semaphore(concurrency)

    with open(sft_path, "a", encoding="utf-8") as sft_file:

        async def bounded(persona, scenario):
            async with sem:
                await rollout_one(persona, scenario, sft_file)

        await asyncio.gather(*[bounded(persona, scenario) for persona, scenario in tasks_spec])

    logger.info(
        "[%s] Complete: %d done, %d skipped, %d failed (of %d total)",
        run_tag,
        counter["done"],
        counter["skipped"],
        counter["failed"],
        counter["total"],
    )
    logger.info("[%s] Conversations → %s", run_tag, conv_dir)
    logger.info("[%s] SFT data → %s", run_tag, sft_path)
    logger.info("LLM stats: %s", llm.stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rollout deep scenario-seeded conversations")
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_privilege", "no_behavior", "no_state", "oracle_profile_only"],
    )
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--persona-ids", nargs="*", help="Filter to specific persona IDs")
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Cap scenarios per persona (default: use all constructed)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Custom output directory (default: output/)"
    )
    parser.add_argument(
        "--constructor",
        type=str,
        default="simulator_lifelong_scenario_constructor",
        help="Scenario constructor prompt name under user_simulator/prompts/",
    )
    parser.add_argument(
        "--force-reconstruct", action="store_true", help="Regenerate scenarios even if cached"
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default=None,
        help=f"Path to personas: a *.jsonl file or a YAML directory (default: {DEFAULT_PROFILES})",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            ablation=args.ablation,
            concurrency=args.concurrency,
            max_turns=args.max_turns,
            min_turns=args.min_turns,
            persona_ids=args.persona_ids,
            max_scenarios=args.max_scenarios,
            output_dir=args.output_dir,
            constructor=args.constructor,
            force_reconstruct=args.force_reconstruct,
            profiles_path=args.profiles,
        )
    )
