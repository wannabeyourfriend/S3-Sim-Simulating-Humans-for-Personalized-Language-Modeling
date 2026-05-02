"""Construct personalized QA-format SFT data from existing conversation JSONs.

Walks `--conversations-dir` recursively, generates QA items in the requested
styles, and emits SFT JSONL byte-compatible with the existing trainer.

If `--qc-results` is given, only Tier-A conversations (per
`output/qc/.../qc_results.jsonl`) are used as sources — guarantees that QA
data inherits the quality gate.

Examples:
    # Smoke (1 persona × 1 style)
    uv run python run_qa_construction.py \
        --conversations-dir output/rollout_gpt_4o_mini_real_world_us_1240_queires_full/conversations \
        --output-dir output/qa/v1_demo_smoke \
        --styles personamem_mcq \
        --sample 5 --concurrency 2

    # Full pilot (10 personas, all 4 styles)
    uv run python run_qa_construction.py \
        --conversations-dir output/rollout_gpt_4o_mini_real_world_us_1240_queires_full/conversations \
        --qc-results output/qc/v1_demo/qc_results.jsonl \
        --output-dir output/qa/v1_demo \
        --styles personamem_mcq prefeval_gen bigtom_tom lamp_cls \
        --concurrency 40

    # PersonaMem-only with self-consistency QC
    uv run python run_qa_construction.py \
        --conversations-dir output/.../conversations \
        --output-dir output/qa/v1_demo \
        --styles personamem_mcq \
        --self-consistency-qc \
        --concurrency 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_profiles_jsonl(path: Path) -> dict:
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


def _load_tier_a_keys(qc_jsonl: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for line in qc_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("tier") == "A":
            keys.add((r.get("persona_id", ""), r.get("scenario_id", "")))
    logger.info("loaded %d Tier-A keys from %s", len(keys), qc_jsonl)
    return keys


async def main(args):
    from user_simulator.data import LLM, load_json
    from user_simulator.qa import (
        QAStyle,
        generate_for_conv,
        qa_item_to_sft_line,
        self_consistency_check_mcq,
    )

    conv_dir = Path(args.conversations_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load profiles
    profiles: dict = {}
    if args.profiles_jsonl:
        profiles = _load_profiles_jsonl(Path(args.profiles_jsonl))

    # Tier filter
    tier_a_keys: set[tuple[str, str]] | None = None
    if args.qc_results:
        tier_a_keys = _load_tier_a_keys(Path(args.qc_results))

    # Walk conversation files
    conv_files = sorted(conv_dir.rglob("*.json"))
    if args.sample:
        conv_files = conv_files[: args.sample]
    logger.info("Found %d conversation JSONs in %s", len(conv_files), conv_dir)

    # Resolve QA styles
    styles = [QAStyle(s) for s in args.styles]
    logger.info("Generating QA styles: %s", [s.value for s in styles])

    # LLM
    gen_model = args.generator_model
    llm = LLM(model=gen_model, max_concurrent=args.concurrency,
              log_calls=args.log_calls)
    logger.info("Generator model: %s; concurrency=%d", gen_model, args.concurrency)

    # Per-style output files (append mode → resume-safe at file level)
    style_files: dict[QAStyle, Path] = {
        s: out_dir / f"{s.value}.jsonl" for s in styles
    }
    file_locks: dict[QAStyle, asyncio.Lock] = {s: asyncio.Lock() for s in styles}
    file_handles: dict[QAStyle, Any] = {}  # opened below

    counter = {"convs": 0, "items": 0, "skipped": 0, "qc_dropped": 0,
               "failed": 0}

    async def process_one(path: Path):
        try:
            session = load_json(path)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            counter["failed"] += 1
            return

        pid = session.get("persona_id", "")
        sid = session.get("prompt_id", "")
        if tier_a_keys is not None and (pid, sid) not in tier_a_keys:
            counter["skipped"] += 1
            return

        persona = profiles.get(pid)
        counter["convs"] += 1

        for style in styles:
            try:
                item = await generate_for_conv(persona, session, style, llm)
            except Exception as e:
                logger.warning("Generation failed for %s/%s/%s: %s",
                               pid, sid, style.value, e)
                counter["failed"] += 1
                continue
            if item is None:
                continue
            # Optional self-consistency for MCQ
            if args.self_consistency_qc and style is QAStyle.PERSONAMEM_MCQ:
                ok = await self_consistency_check_mcq(item, llm)
                if not ok:
                    counter["qc_dropped"] += 1
                    continue
            line = qa_item_to_sft_line(item, session)
            async with file_locks[style]:
                file_handles[style].write(json.dumps(line, ensure_ascii=False) + "\n")
                file_handles[style].flush()
            counter["items"] += 1
            if counter["items"] % 25 == 0:
                logger.info("Progress: %d items written (%d convs processed, "
                            "%d skipped, %d qc-dropped, %d failed)",
                            counter["items"], counter["convs"],
                            counter["skipped"], counter["qc_dropped"],
                            counter["failed"])

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(path):
        async with sem:
            await process_one(path)

    # Open all per-style files
    try:
        for s, p in style_files.items():
            file_handles[s] = open(p, "a", encoding="utf-8")

        t0 = time.time()
        await asyncio.gather(*[bounded(p) for p in conv_files])
        elapsed = time.time() - t0
    finally:
        for fh in file_handles.values():
            fh.close()

    logger.info(
        "Done in %.1fs: %d items, %d convs, %d skipped (non-Tier-A), %d qc-dropped, %d failed",
        elapsed, counter["items"], counter["convs"],
        counter["skipped"], counter["qc_dropped"], counter["failed"],
    )
    for s, p in style_files.items():
        try:
            n = sum(1 for _ in open(p, encoding="utf-8"))
        except OSError:
            n = 0
        logger.info("  %s → %s (%d lines)", s.value, p, n)
    logger.info("LLM stats: %s", llm.stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct QA-format SFT data from S³-Sim conversations")
    parser.add_argument("--conversations-dir", required=True,
                        help="Directory of conversation JSONs (recursive)")
    parser.add_argument("--output-dir", default="output/qa/v1_demo",
                        help="Where per-style JSONL files go")
    parser.add_argument("--qc-results", default=None,
                        help="Path to qc_results.jsonl. If given, only Tier-A "
                             "conversations are used as sources.")
    parser.add_argument("--profiles-jsonl",
                        default="data/filterd_refined_profiles/summary_refined_profiles_us.jsonl",
                        help="JSONL of personas (one per line)")
    parser.add_argument("--styles", nargs="+",
                        default=["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"],
                        choices=["personamem_mcq", "prefeval_gen", "bigtom_tom", "lamp_cls"])
    parser.add_argument("--generator-model", default="gpt-4.1-mini",
                        help="QA generator model (defaults to gpt-4.1-mini, distinct from convo generator)")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only the first N conv files (smoke)")
    parser.add_argument("--self-consistency-qc", action="store_true",
                        help="Run profile-stripped self-consistency check on MCQ items")
    parser.add_argument("--log-calls", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args))
