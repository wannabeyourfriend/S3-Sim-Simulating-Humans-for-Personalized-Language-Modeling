"""Print the actual behavior guidance blocks injected into user simulator prompts.

For each behavior in the library, renders the full <behavior_injection> block
at each disclosure stage (minimal/standard/full), showing exactly what the
model sees including few-shot examples.

Usage: uv run python tests/test_behavior_block.py
       uv run python tests/test_behavior_block.py --behavior analysis
       uv run python tests/test_behavior_block.py --stage full
"""

import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from user_simulator.simulator import (
    _BEHAVIORS,
    _BEHAVIOR_ORDER,
    _make_behavior_block,
    _infer_disclosure_stage,
)

SEP = "=" * 80


def print_block(behavior: dict, stage: str):
    """Force a specific disclosure stage and print the block."""

    b = dict(behavior)
    ctrl = dict(b.get("simulator_control", {}))
    ctrl["force_disclosure_stage"] = stage
    b["simulator_control"] = ctrl

    block, actual_stage, bname = _make_behavior_block(b, [])
    n_lines = block.count("\n") + 1 if block else 0
    print(f"  [{stage}] {n_lines} lines, {len(block)} chars")
    print(block)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavior", type=str, default=None, help="Specific behavior_id to print (default: all)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["minimal", "standard", "full"],
        help="Only print this stage (default: all 3)",
    )
    args = parser.parse_args()

    stages = [args.stage] if args.stage else ["minimal", "standard", "full"]
    behaviors = [args.behavior] if args.behavior else _BEHAVIOR_ORDER

    for bid in behaviors:
        if bid not in _BEHAVIORS:
            print(f"Unknown behavior: {bid}")
            print(f"Available: {_BEHAVIOR_ORDER}")
            sys.exit(1)

        b = _BEHAVIORS[bid]
        n_examples = len(b.get("few_shot_examples") or [])
        print(SEP)
        print(f"BEHAVIOR: {bid} ({b.get('name', bid)})")
        print(f"  mode: {b.get('tuna_mode', '?')}")
        print(f"  strategy: {b.get('tuna_strategy', '?')}")
        print(f"  delegation: {b.get('cognitive_delegation_level', '?')}")
        print(f"  few_shot_examples: {n_examples}")
        print(SEP)

        for stage in stages:
            print_block(b, stage)

    print(f"\nTotal behaviors: {len(_BEHAVIOR_ORDER)}")
    print(f"Behavior IDs: {_BEHAVIOR_ORDER}")


if __name__ == "__main__":
    main()
