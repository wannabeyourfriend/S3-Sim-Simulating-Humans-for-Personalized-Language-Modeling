"""
uv run python utils/refine_us_profiles.py
uv run python utils/refine_us_profiles.py --force
uv run python utils/refine_us_profiles.py --batch-size 10
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from pathlib import Path
import yaml as _yaml
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
UTILS_PROMPTS = PROJECT_ROOT / "utils" / "prompts"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "filtered_profiles" / "US"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "refined_profiles" / "US"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
with open(UTILS_PROMPTS / "utils_profile_refiner.yaml") as f:
    _REFINER_PROMPT: str = _yaml.safe_load(f)["prompt"]

def render(template: str, **kw: str) -> str:
    for k, v in kw.items():
        template = template.replace("{" + k + "}", str(v))
    return template

async def refine_one(
    llm,
    profile_path: Path,
    output_dir: Path,
    index: int,
    total: int,
    force: bool,
) -> str:
    """Refine a single profile. Returns 'refined', 'skipped', or 'failed'."""
    profile_data: dict = _yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    pid: str = profile_data.get("persona_id", profile_path.stem)
    out_path = output_dir / f"{pid}.yaml"
    if not force and out_path.exists():
        existing = _yaml.safe_load(out_path.read_text(encoding="utf-8"))
        if existing and existing.get("behavioral_metadata"):
            log.info("[%d/%d] Skipping %s (already refined)", index, total, pid)
            return "skipped"

    log.info("[%d/%d] Refining %s ...", index, total, pid)
    attrs = profile_data.get("attributes", {})
    attrs_str = (
        _yaml.dump(attrs, default_flow_style=False)
        if attrs
        else "No detailed attributes available."
    )
    summary_str = profile_data.get("summary", "No summary available.")

    rendered = render(
        _REFINER_PROMPT,
        profile_summary=summary_str,
        profile_attributes=attrs_str,
    )

    try:
        result: dict = await llm.chat_json(
            [
                {"role": "system", "content": rendered},
                {"role": "user", "content": "Analyze this persona profile."},
            ],
            temperature=0.4,
            max_tokens=4096,
        )
    except Exception:
        log.exception("[%d/%d] LLM call failed for %s", index, total, pid)
        return "failed"
    bm = result.get("behavioral_metadata")
    rs = result.get("refined_summary")
    if not bm or not rs:
        log.error(
            "[%d/%d] Missing keys in LLM response for %s — got keys: %s",
            index,
            total,
            pid,
            list(result.keys()),
        )
        return "failed"
    enriched = dict(profile_data)
    enriched["behavioral_metadata"] = bm
    enriched["refined_summary"] = rs
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            _yaml.dump(enriched, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except Exception:
        log.exception("[%d/%d] Failed to write %s", index, total, out_path)
        return "failed"

    log.info("[%d/%d] Saved %s", index, total, out_path.name)
    return "refined"

async def run(
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    force: bool,
) -> None:
    from utils.llm_client import LLMClient

    llm = LLMClient(max_concurrent=batch_size)

    profiles = sorted(input_dir.glob("profile_*.yaml"))
    total = len(profiles)
    if total == 0:
        log.warning("No profile YAMLs found in %s", input_dir)
        return

    log.info("Found %d profiles in %s", total, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {"refined": 0, "skipped": 0, "failed": 0}
    for start in range(0, total, batch_size):
        batch = profiles[start : start + batch_size]
        tasks = [
            refine_one(
                llm,
                p,
                output_dir,
                index=start + i + 1,
                total=total,
                force=force,
            )
            for i, p in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.error("Unexpected exception in batch: %s", r)
                counts["failed"] += 1
            else:
                counts[r] += 1
    log.info(
        "Done — %d refined, %d skipped, %d failed (out of %d)",
        counts["refined"],
        counts["skipped"],
        counts["failed"],
        total,
    )
    log.info("LLM stats: %s", llm.stats)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich filtered profiles with behavioral metadata via LLM."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory with source profile YAMLs (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write enriched profiles (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of concurrent LLM calls per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-refine profiles even if they already have behavioral_metadata",
    )
    args = parser.parse_args()

    asyncio.run(run(args.input_dir, args.output_dir, args.batch_size, args.force))

if __name__ == "__main__":
    main()
