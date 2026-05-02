"""Deduplicate and reorder chat fine-tuning JSONL files.

Defaults are intentionally conservative for release artifacts:
- duplicate identity is the canonical `messages` payload;
- the first occurrence is kept;
- output order is deterministic and de-clustered by persona/scenario/label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SEED = 42
DEFAULT_DECLUSTER_FIELDS = [
    "metadata.persona_id",
    "metadata.scenario_id",
    "metadata.correct_letter",
]


@dataclass(frozen=True)
class Record:
    line_no: int
    value: dict[str, Any]
    dedupe_key: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_path(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def dedupe_payload(record: dict[str, Any], dedupe_key: str) -> Any:
    if dedupe_key == "messages":
        return record.get("messages")
    if dedupe_key == "full-record":
        return record
    raise ValueError(f"unsupported dedupe key: {dedupe_key}")


def validate_chat_record(record: dict[str, Any], line_no: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_no}: expected non-empty `messages` list")

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"line {line_no}: message {idx} is not an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"line {line_no}: message {idx} has invalid role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"line {line_no}: message {idx} has empty content")


def load_unique_records(
    input_path: Path,
    dedupe_key: str,
) -> tuple[list[Record], list[dict[str, int]]]:
    unique: list[Record] = []
    seen: dict[str, int] = {}
    duplicates: list[dict[str, int]] = []

    with input_path.open(encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"line {line_no}: expected JSON object")
            validate_chat_record(record, line_no)

            key = canonical_json(dedupe_payload(record, dedupe_key))
            first_line = seen.get(key)
            if first_line is not None:
                duplicates.append({"first_line": first_line, "duplicate_line": line_no})
                continue

            seen[key] = line_no
            unique.append(Record(line_no=line_no, value=record, dedupe_key=key))

    return unique, duplicates


def collision_score(left: dict[str, Any], right: dict[str, Any], fields: list[str]) -> int:
    score = 0
    for field in fields:
        left_value = get_path(left, field)
        right_value = get_path(right, field)
        if left_value is not None and left_value == right_value:
            score += 1
    return score


def reorder_records(
    records: list[Record],
    seed: int,
    decluster_fields: list[str],
    window_size: int,
) -> list[Record]:
    rng = random.Random(seed)
    remaining = records[:]
    rng.shuffle(remaining)

    if not decluster_fields or window_size <= 1:
        return remaining

    output: list[Record] = []
    while remaining:
        if not output:
            output.append(remaining.pop(0))
            continue

        scan_limit = min(window_size, len(remaining))
        best_idx = 0
        best_score = None
        last = output[-1].value
        for idx in range(scan_limit):
            score = collision_score(last, remaining[idx].value, decluster_fields)
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
                if score == 0:
                    break
        output.append(remaining.pop(best_idx))

    return output


def adjacent_matches(records: list[Record], fields: list[str]) -> dict[str, int]:
    counts = {field: 0 for field in fields}
    for left, right in zip(records, records[1:]):
        for field in fields:
            left_value = get_path(left.value, field)
            right_value = get_path(right.value, field)
            if left_value is not None and left_value == right_value:
                counts[field] += 1
    return counts


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.deduped_shuffled{input_path.suffix}")


def default_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(f"{output_path.suffix}.report.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(output_path: Path, records: list[Record], overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record.value, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    tmp_path.replace(output_path)


def write_report_atomic(report_path: Path, report: dict[str, Any], overwrite: bool) -> None:
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"{report_path} exists; pass --overwrite to replace it")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=report_path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    tmp_path.replace(report_path)


def process_jsonl(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    dedupe_key: str = "messages",
    seed: int = DEFAULT_SEED,
    decluster_fields: list[str] | None = None,
    window_size: int = 256,
    overwrite: bool = False,
) -> dict[str, Any]:
    fields = decluster_fields or []
    unique_records, duplicates = load_unique_records(input_path, dedupe_key)
    before_adjacent = adjacent_matches(unique_records, fields)
    ordered_records = reorder_records(unique_records, seed, fields, window_size)
    after_adjacent = adjacent_matches(ordered_records, fields)

    if len({record.dedupe_key for record in ordered_records}) != len(ordered_records):
        raise RuntimeError("dedupe invariant failed after reordering")

    write_jsonl_atomic(output_path, ordered_records, overwrite=overwrite)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dedupe_key": dedupe_key,
        "seed": seed,
        "window_size": window_size,
        "decluster_fields": fields,
        "input_records": len(unique_records) + len(duplicates),
        "output_records": len(ordered_records),
        "duplicates_removed": len(duplicates),
        "duplicate_examples": duplicates[:20],
        "adjacent_matches_before": before_adjacent,
        "adjacent_matches_after": after_adjacent,
        "output_sha256": sha256_file(output_path),
    }
    write_report_atomic(report_path, report, overwrite=overwrite)
    return report


def parse_fields(raw_fields: str) -> list[str]:
    return [field.strip() for field in raw_fields.split(",") if field.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate and deterministically reorder chat fine-tuning JSONL data.",
    )
    parser.add_argument("input", type=Path, help="Input JSONL file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSONL path. Defaults to <input>.deduped_shuffled.jsonl.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Report JSON path. Defaults to <output>.report.json.",
    )
    parser.add_argument(
        "--dedupe-key",
        choices=["messages", "full-record"],
        default="messages",
        help="Which payload defines a duplicate sample.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--decluster-fields",
        default=",".join(DEFAULT_DECLUSTER_FIELDS),
        help="Comma-separated dotted fields used to reduce adjacent clustering.",
    )
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.input)
    report_path = args.report or default_report_path(output_path)
    report = process_jsonl(
        input_path=args.input,
        output_path=output_path,
        report_path=report_path,
        dedupe_key=args.dedupe_key,
        seed=args.seed,
        decluster_fields=parse_fields(args.decluster_fields),
        window_size=args.window_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
