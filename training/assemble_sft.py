"""Concatenate rollout conversations and QA-format JSONL into one SFT file.

This is a thin orchestration layer around `user_simulator.oracle.assemble_sft`
and the per-style QA JSONLs produced by `run_qa_construction.py`.

Inputs (any combination):

  --conv-dirs DIR [DIR ...]   one or more rollout-conversation directories
                              (each is recursively scanned for *.json)
  --qa-files  FILE [FILE ...] one or more QA-format JSONL files (already in
                              {"messages":..., "metadata":...} schema)

Output (`--output`): one multi-turn SFT JSONL with the same line schema for
both sources, suitable for TRL `SFTTrainer.train()` or any equivalent runner.

Optional knobs:
  --include-profile / --no-profile   inject the user profile into the system
                                     prompt for conv-source lines (default on)
  --max-tokens N                     drop conv-source lines whose total token
                                     count exceeds N (default 32 000)
  --shuffle / --no-shuffle           shuffle final lines deterministically
  --seed N                           shuffle seed (default 42)

Example
-------

    uv run python -m training.assemble_sft \\
      --conv-dirs output/rollout_full/conversations \\
                  output/rollout_lifelong/conversations \\
      --qa-files  output/qa/v1/personamem_mcq.jsonl \\
                  output/qa/v1/prefeval_gen.jsonl \\
      --output    data/sft/train.jsonl \\
      --shuffle
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("skip malformed line in %s: %s", path, e)
    return out


def main(args: argparse.Namespace) -> None:
    from user_simulator.oracle import assemble_sft

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for conv_dir in args.conv_dirs or []:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            assemble_sft(
                conversations_dir=Path(conv_dir),
                output_path=tmp_path,
                include_profile=args.include_profile,
                max_tokens=args.max_tokens,
            )
            rows.extend(_read_jsonl(tmp_path))
        finally:
            tmp_path.unlink(missing_ok=True)

    for qa_file in args.qa_files or []:
        rows.extend(_read_jsonl(Path(qa_file)))

    if not rows:
        raise SystemExit("no input lines — nothing to write")

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    with open(output, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("wrote %d SFT lines → %s", len(rows), output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv-dirs", nargs="*", default=[],
                        help="Rollout conversation directories (recursively scanned)")
    parser.add_argument("--qa-files", nargs="*", default=[],
                        help="QA-format JSONL files (one line per item)")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--include-profile", dest="include_profile",
                        action="store_true", default=True,
                        help="Inject persona profile into system prompt (default)")
    parser.add_argument("--no-profile", dest="include_profile",
                        action="store_false",
                        help="Strip persona profile from system prompt")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true",
                        default=True, help="Shuffle output (default)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
