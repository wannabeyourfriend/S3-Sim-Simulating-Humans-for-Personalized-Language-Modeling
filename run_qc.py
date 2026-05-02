"""Run quality-check pipeline over a directory of conversation JSONs.

Walks `--conversations-dir` recursively, scores each conversation along D1–D6,
streams `QCResult` rows to a JSONL, and writes an aggregate summary JSON.

Resumable: if `--output-dir/qc_results.jsonl` already contains a row for
`(persona_id, scenario_id)`, that conversation is skipped.

Examples:
    # Programmatic-only smoke (no LLM cost)
    uv run python run_qc.py \
        --conversations-dir output/rollout_gpt_4o_mini_real_world_us_1240_queires_full/conversations \
        --output-dir output/qc/v1_demo \
        --skip-judges

    # Full QC including judges
    uv run python run_qc.py \
        --conversations-dir output/rollout_gpt_4o_mini_real_world_us_1240_queires_full/conversations \
        --profiles-jsonl mind2dialogue/data/filterd_refined_profiles/summary_refined_profiles_us.jsonl \
        --output-dir output/qc/v1_demo \
        --concurrency 40

Profiles can be loaded from either:
  * `--profiles-jsonl PATH` — a JSONL file with one persona per line
    (fields: persona_id, summary, refined_summary, behavioral_metadata)
  * `--profiles-dir PATH` — a directory of YAML files (legacy `load_personas`)

Judge model is taken from env `JUDGE_MODEL` (default: gpt-4.1-mini), distinct
from the generator's MODEL_NAME to mitigate self-judging bias.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_profiles_jsonl(path: Path) -> dict:
    """Load personas from a JSONL file. Returns {persona_id: Persona}."""
    from user_simulator.data import Persona

    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        metadata = {}
        if raw.get("behavioral_metadata"):
            metadata["behavioral_metadata"] = raw["behavioral_metadata"]
        if raw.get("refined_summary"):
            metadata["refined_summary"] = raw["refined_summary"]
        pid = raw.get("persona_id") or raw.get("id") or ""
        if not pid:
            continue
        out[pid] = Persona(
            id=pid,
            attributes=raw.get("attributes") or {},
            summary=raw.get("summary", ""),
            fingerprint=raw.get("fingerprint") or {},
            metadata=metadata,
            selected_prompts=raw.get("selected_prompts") or [],
        )
    logger.info("loaded %d personas from %s", len(out), path)
    return out


def _load_profiles_dir(path: Path) -> dict:
    from user_simulator.data import load_personas

    return {p.id: p for p in load_personas(path)}


def _read_existing_keys(jsonl_path: Path) -> set[tuple[str, str]]:
    if not jsonl_path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            keys.add((r.get("persona_id", ""), r.get("scenario_id", "")))
        except json.JSONDecodeError:
            continue
    return keys


def _summarize(results_path: Path, summary_path: Path) -> dict:
    """Aggregate QC results into a summary JSON."""
    n = 0
    tier_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    d5_dist: Counter[int | str] = Counter()
    d6_dist: Counter[str] = Counter()
    by_ablation: Counter[tuple[str, str]] = Counter()

    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        tier_counts[r.get("tier", "C")] += 1
        for d in r.get("failed_dims") or []:
            fail_counts[d] += 1
        d5 = r.get("d5_persona_consistency")
        d5_dist[d5 if d5 is not None else "null"] += 1
        d6 = r.get("d6_conflict") or "null"
        d6_dist[d6] += 1

    summary = {
        "total": n,
        "tier_counts": dict(tier_counts),
        "tier_pass_rate": tier_counts["A"] / n if n else 0.0,
        "failed_dim_counts": dict(fail_counts),
        "d5_distribution": {str(k): v for k, v in d5_dist.items()},
        "d6_distribution": dict(d6_dist),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written → %s", summary_path)
    return summary


async def main(args):
    from user_simulator.data import LLM, load_json
    from user_simulator.qc import score_conversation

    conv_dir = Path(args.conversations_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "qc_results.jsonl"
    summary_path = out_dir / "qc_summary.json"

    profiles: dict = {}
    if args.profiles_jsonl:
        profiles = _load_profiles_jsonl(Path(args.profiles_jsonl))
    elif args.profiles_dir:
        profiles = _load_profiles_dir(Path(args.profiles_dir))
    else:
        logger.warning("No profiles source provided; D4 will fail for every conv.")

    conv_files = sorted(conv_dir.rglob("*.json"))
    if args.sample:
        conv_files = conv_files[: args.sample]
    logger.info("Found %d conversation JSONs in %s", len(conv_files), conv_dir)

    existing = _read_existing_keys(results_path)
    if existing:
        logger.info("Resuming: %d existing results in %s", len(existing), results_path)

    llm: LLM | None = None
    if not args.skip_judges:
        judge_model = os.getenv("JUDGE_MODEL") or args.judge_model
        llm = LLM(model=judge_model, max_concurrent=args.concurrency, log_calls=args.log_calls)
        logger.info("Judge model: %s; concurrency=%d", judge_model, args.concurrency)
    else:
        logger.info("Skipping judges (programmatic-only)")

    counter = {"done": 0, "skipped": 0, "failed": 0, "total": len(conv_files)}

    file_lock = asyncio.Lock()

    async def score_one(path: Path, fh):
        try:
            conv = load_json(path)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            counter["failed"] += 1
            return

        key = (conv.get("persona_id", ""), conv.get("prompt_id", ""))
        if key in existing:
            counter["skipped"] += 1
            return

        persona = profiles.get(conv.get("persona_id", ""))
        try:
            result = await score_conversation(conv, persona, llm, skip_judges=args.skip_judges)
        except Exception as e:
            logger.exception("Score failed for %s: %s", path, e)
            counter["failed"] += 1
            return

        async with file_lock:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()
        counter["done"] += 1
        if counter["done"] % 50 == 0:
            logger.info(
                "Progress: %d/%d done, %d skipped, %d failed",
                counter["done"],
                counter["total"],
                counter["skipped"],
                counter["failed"],
            )

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(path, fh):
        async with sem:
            await score_one(path, fh)

    t0 = time.time()
    with open(results_path, "a", encoding="utf-8") as fh:
        await asyncio.gather(*[bounded(p, fh) for p in conv_files])
    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs: %d scored, %d skipped, %d failed",
        elapsed,
        counter["done"],
        counter["skipped"],
        counter["failed"],
    )

    summary = _summarize(results_path, summary_path)
    logger.info(
        "Tier breakdown: %s (pass-rate=%.3f)", summary["tier_counts"], summary["tier_pass_rate"]
    )
    if llm is not None:
        logger.info("Judge LLM stats: %s", llm.stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quality-check Mind2Dialogue conversations")
    parser.add_argument(
        "--conversations-dir", required=True, help="Directory of conversation JSONs (recursive)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/qc/v1_demo",
        help="Where qc_results.jsonl + qc_summary.json go",
    )
    parser.add_argument(
        "--profiles-jsonl",
        default="data/filterd_refined_profiles/summary_refined_profiles_us.jsonl",
        help="JSONL of personas (one per line)",
    )
    parser.add_argument(
        "--profiles-dir", default=None, help="Alternative: YAML directory (legacy load_personas)"
    )
    parser.add_argument(
        "--skip-judges", action="store_true", help="Skip D5/D6 LLM judges (programmatic only)"
    )
    parser.add_argument(
        "--judge-model", default="gpt-4.1-mini", help="Override JUDGE_MODEL env var"
    )
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument(
        "--sample", type=int, default=None, help="Score only the first N conv files (smoke)"
    )
    parser.add_argument(
        "--log-calls", action="store_true", help="JSONL-log every judge call to output/llm_logs/"
    )
    args = parser.parse_args()
    asyncio.run(main(args))
