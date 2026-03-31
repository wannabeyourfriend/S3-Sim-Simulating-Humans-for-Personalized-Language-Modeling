import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import yaml
def _normalize_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_value(item))
        return normalized
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, sort_keys=True)]
    text = str(value).strip()
    return [text] if text else []


def collect_fingerprint_stats(profile_paths: list[Path]) -> dict[str, Any]:
    total_personas = 0
    personas_with_fingerprint = 0
    field_presence = Counter()
    value_counters: dict[str, Counter[str]] = defaultdict(Counter)
    field_is_multivalue: dict[str, bool] = {}

    for path in profile_paths:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        total_personas += 1
        fingerprint = data.get("fingerprint")
        if not isinstance(fingerprint, dict):
            continue

        personas_with_fingerprint += 1
        for field, raw_value in fingerprint.items():
            if field.lstrip(".") == "_meta":
                continue
            values = _normalize_value(raw_value)
            if not values:
                continue
            field_presence[field] += 1
            field_is_multivalue[field] = field_is_multivalue.get(field, False) or len(values) > 1
            for value in values:
                value_counters[field][value] += 1

    fields: dict[str, Any] = {}
    for field in sorted(value_counters.keys()):
        total_value_count = sum(value_counters[field].values())
        sorted_values = sorted(
            value_counters[field].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        fields[field] = {
            "persona_coverage_count": field_presence[field],
            "persona_coverage_ratio": (
                field_presence[field] / total_personas if total_personas else 0.0
            ),
            "is_multivalue": field_is_multivalue.get(field, False),
            "total_value_count": total_value_count,
            "distinct_value_count": len(value_counters[field]),
            "values": [
                {
                    "value": value,
                    "count": count,
                    "ratio_in_personas": count / total_personas if total_personas else 0.0,
                    "ratio_in_field_values": count / total_value_count if total_value_count else 0.0,
                }
                for value, count in sorted_values
            ],
        }

    return {
        "total_personas": total_personas,
        "personas_with_fingerprint": personas_with_fingerprint,
        "fingerprint_coverage_ratio": (
            personas_with_fingerprint / total_personas if total_personas else 0.0
        ),
        "fields": fields,
    }


def render_text_report(stats: dict[str, Any], top_k: int) -> str:
    lines = [
        "Persona Fingerprint Stats",
        f"Total personas: {stats['total_personas']}",
        f"Personas with fingerprint: {stats['personas_with_fingerprint']} "
        f"({stats['fingerprint_coverage_ratio'] * 100:.1f}%)",
        "",
    ]

    for field, field_stats in stats["fields"].items():
        coverage = field_stats["persona_coverage_ratio"] * 100
        mode = "multi" if field_stats["is_multivalue"] else "single"
        lines.append(
            f"[{field}] coverage={field_stats['persona_coverage_count']}/{stats['total_personas']} "
            f"({coverage:.1f}%), mode={mode}, distinct={field_stats['distinct_value_count']}"
        )
        for item in field_stats["values"][:top_k]:
            lines.append(
                f"  - {item['value']}: {item['count']} "
                f"(persona_ratio={item['ratio_in_personas'] * 100:.1f}%, "
                f"field_ratio={item['ratio_in_field_values'] * 100:.1f}%)"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _discover_profile_paths(profile_dir: Path) -> list[Path]:
    return sorted(p for p in profile_dir.glob("*.yaml") if p.is_file())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute persona fingerprint statistics.")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("data/profiles/yaml"),
        help="Directory containing persona profile YAML files.",
    )
    parser.add_argument(
        "--profile-paths",
        type=Path,
        nargs="*",
        default=None,
        help="Specific profile YAML paths. If provided, profile-dir is ignored.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top values to show for each fingerprint field in text output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print stats in JSON format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_paths = args.profile_paths or _discover_profile_paths(args.profile_dir)
    if not profile_paths:
        raise SystemExit("No profile YAML files found.")

    stats = collect_fingerprint_stats(profile_paths)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    else:
        print(render_text_report(stats, top_k=args.top_k), end="")


if __name__ == "__main__":
    main()
