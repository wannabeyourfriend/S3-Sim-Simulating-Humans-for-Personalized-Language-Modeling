"""Assemble the curated final release dataset from existing rollout artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO / "output" / "final_release_dataset"
SEED = 42


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: str
    data_kind: str
    region: str
    scenario_family: str
    seed_source: str
    release_group: str
    rewrite_status: str = "not_applicable"
    qa_style: str | None = None
    primary: bool = True


CONVERSATION_SOURCES = [
    SourceSpec(
        name="v4_us_lifelong",
        path="output/release_v4_us/full_lifelong/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="us",
        scenario_family="lifelong",
        seed_source="scenario_constructor",
        release_group="v4_us",
    ),
    SourceSpec(
        name="v4_us_highfreq",
        path="output/release_v4_us/full_highfreq/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="us",
        scenario_family="highfreq",
        seed_source="scenario_constructor",
        release_group="v4_us",
    ),
    SourceSpec(
        name="v4_us_affective",
        path="output/release_v4_us/full_affective/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="us",
        scenario_family="affective",
        seed_source="scenario_constructor",
        release_group="v4_us",
    ),
    SourceSpec(
        name="v4_cn_lifelong",
        path="output/release_v4_xc/cn_lifelong/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="cn",
        scenario_family="lifelong",
        seed_source="scenario_constructor",
        release_group="v4_xc",
    ),
    SourceSpec(
        name="v4_jp_lifelong",
        path="output/release_v4_xc/jp_lifelong/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="jp",
        scenario_family="lifelong",
        seed_source="scenario_constructor",
        release_group="v4_xc",
    ),
    SourceSpec(
        name="real_world_initial_prompt_us",
        path="output/rollout_gpt_4o_mini_real_world_us_1240_queires_full/sft/train_full.jsonl",
        data_kind="conversation",
        region="us",
        scenario_family="random_initial_prompt",
        seed_source="data/initial_prompts",
        release_group="real_world_us_1240",
    ),
    SourceSpec(
        name="pilot_us_lifelong",
        path="output/release_us_1k/lifelong/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="us",
        scenario_family="lifelong",
        seed_source="scenario_constructor",
        release_group="pilot_us_1k",
    ),
    SourceSpec(
        name="pilot_us_highfreq",
        path="output/release_us_1k/highfreq/sft/train_deep_full.jsonl",
        data_kind="conversation",
        region="us",
        scenario_family="highfreq",
        seed_source="scenario_constructor",
        release_group="pilot_us_1k",
    ),
]

QA_SOURCES = [
    SourceSpec(
        name="pilot_personamem_mcq_v2",
        path="output/release_us_1k/qa_v2/personamem_mcq.jsonl",
        data_kind="qa",
        region="us",
        scenario_family="lifelong_highfreq",
        seed_source="scenario_constructor",
        release_group="pilot_us_1k",
        rewrite_status="v2_rewritten",
        qa_style="personamem_mcq",
    ),
    SourceSpec(
        name="pilot_prefeval_gen_v2",
        path="output/release_us_1k/qa_v2/prefeval_gen.jsonl",
        data_kind="qa",
        region="us",
        scenario_family="lifelong_highfreq",
        seed_source="scenario_constructor",
        release_group="pilot_us_1k",
        rewrite_status="v2_rewritten",
        qa_style="prefeval_gen",
    ),
    SourceSpec(
        name="pilot_bigtom_tom_v2",
        path="output/release_us_1k/qa_v2/bigtom_tom.jsonl",
        data_kind="qa",
        region="us",
        scenario_family="lifelong_highfreq",
        seed_source="scenario_constructor",
        release_group="pilot_us_1k",
        rewrite_status="v2_rewritten",
        qa_style="bigtom_tom",
    ),
    SourceSpec(
        name="pilot_lamp_cls_v2",
        path="output/release_us_1k/qa_v2/lamp_cls.jsonl",
        data_kind="qa",
        region="us",
        scenario_family="lifelong_highfreq",
        seed_source="scenario_constructor",
        release_group="pilot_us_1k",
        rewrite_status="v2_rewritten",
        qa_style="lamp_cls",
        primary=False,
    ),
]

for region, slice_name, family in [
    ("us", "full_lifelong", "lifelong"),
    ("us", "full_highfreq", "highfreq"),
    ("us", "full_affective", "affective"),
    ("cn", "cn_lifelong", "lifelong"),
    ("jp", "jp_lifelong", "lifelong"),
]:
    root = "output/release_v4_us" if region == "us" else "output/release_v4_xc"
    for style in ["personamem_mcq", "prefeval_gen", "lamp_cls"]:
        QA_SOURCES.append(
            SourceSpec(
                name=f"v4_{region}_{family}_{style}_v2",
                path=f"{root}/qa_v2/{slice_name}/{style}.jsonl",
                data_kind="qa",
                region=region,
                scenario_family=family,
                seed_source="scenario_constructor",
                release_group="v4_us" if region == "us" else "v4_xc",
                rewrite_status="v2_rewritten",
                qa_style=style,
                primary=style != "lamp_cls",
            )
        )

EXCLUDED_SOURCES = [
    {
        "reason": "sensitive add-on candidate, not part of the requested three primary scenario constructors",
        "paths": [
            "output/release_v4_us/full_concerning",
            "output/release_v4_us/qa/full_concerning",
            "output/release_v4_us/qa_v2/full_concerning",
        ],
    },
    {
        "reason": "ablations are research controls, not primary alignment training data",
        "paths": [
            "output/rollout_gpt_4o_mini_real_world_us_1240_queeries_oracle_with_no_privillige",
            "output/rollout_gpt_4o_mini_real_world_us_1240_queries_user_vanilla_ablated",
        ],
    },
    {
        "reason": "smoke, drift, eval, and already merged v4_* files are excluded to avoid leakage and double counting",
        "paths": [
            "output/smoke_*",
            "output/drift_*",
            "output/eval",
            "output/release_v4_us/sft/v4_*.jsonl",
        ],
    },
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_messages(record: dict[str, Any], source: SourceSpec, line_no: int) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{source.path}:{line_no}: expected non-empty `messages` list")
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"{source.path}:{line_no}: message {idx} is not an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(
                f"{source.path}:{line_no}: invalid message role {message.get('role')!r}"
            )
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"{source.path}:{line_no}: empty message content")


def with_release_metadata(
    record: dict[str, Any], source: SourceSpec, sample_id: str
) -> dict[str, Any]:
    copied = {
        "messages": record["messages"],
        "metadata": dict(record.get("metadata") or {}),
    }
    copied["metadata"].update(
        {
            "sample_id": sample_id,
            "release_source": source.name,
            "release_group": source.release_group,
            "release_region": source.region,
            "scenario_family": source.scenario_family,
            "seed_source": source.seed_source,
            "rewrite_status": source.rewrite_status,
        }
    )
    return copied


def load_records(
    sources: Iterable[SourceSpec],
    repo: Path,
    dedupe_by_messages: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    source_counts: dict[str, dict[str, int]] = {}
    duplicate_examples: list[dict[str, str]] = []

    for source in sources:
        path = repo / source.path
        source_counts[source.name] = {"input": 0, "kept": 0, "missing": 0, "duplicates": 0}
        if not path.exists():
            source_counts[source.name]["missing"] = 1
            continue
        with path.open(encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{source.path}:{line_no}: expected JSON object")
                validate_messages(record, source, line_no)
                source_counts[source.name]["input"] += 1

                key_payload = record["messages"] if dedupe_by_messages else record
                dedupe_key = sha256_text(canonical_json(key_payload))
                if dedupe_key in seen:
                    source_counts[source.name]["duplicates"] += 1
                    if len(duplicate_examples) < 20:
                        duplicate_examples.append(
                            {
                                "duplicate_source": source.name,
                                "duplicate_line": str(line_no),
                                "kept_sample_id": seen[dedupe_key],
                            }
                        )
                    continue

                sample_id = f"{source.data_kind}_{dedupe_key[:16]}"
                seen[dedupe_key] = sample_id
                records.append(with_release_metadata(record, source, sample_id))
                source_counts[source.name]["kept"] += 1

    return records, {"source_counts": source_counts, "duplicate_examples": duplicate_examples}


def stable_declutter_shuffle(
    records: list[dict[str, Any]],
    seed: int,
    fields: list[str],
    window_size: int = 256,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    remaining = records[:]
    rng.shuffle(remaining)
    output: list[dict[str, Any]] = []

    while remaining:
        if not output:
            output.append(remaining.pop(0))
            continue
        last_meta = output[-1].get("metadata") or {}
        best_idx = 0
        best_score = None
        for idx, candidate in enumerate(remaining[:window_size]):
            meta = candidate.get("metadata") or {}
            score = sum(
                1
                for field in fields
                if last_meta.get(field) is not None and last_meta.get(field) == meta.get(field)
            )
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score
                if score == 0:
                    break
        output.append(remaining.pop(best_idx))
    return output


def build_assistant_turn_records(
    conversation_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    turn_records: list[dict[str, Any]] = []
    for record in conversation_records:
        messages = record["messages"]
        assistant_indices = [
            idx for idx, message in enumerate(messages) if message.get("role") == "assistant"
        ]
        for turn_idx, message_idx in enumerate(assistant_indices, 1):
            prompt = messages[:message_idx]
            completion = [messages[message_idx]]
            metadata = dict(record.get("metadata") or {})
            metadata.update(
                {
                    "sample_id": f"{metadata['sample_id']}_turn_{turn_idx:02d}",
                    "parent_sample_id": metadata["sample_id"],
                    "assistant_turn_index": turn_idx,
                    "assistant_turn_count": len(assistant_indices),
                    "format": "conversational_prompt_completion",
                }
            )
            turn_records.append({"prompt": prompt, "completion": completion, "metadata": metadata})
    return turn_records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    tmp_path.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def write_readme(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# S3-Sim Final Release Dataset",
        "",
        "Curated JSONL release assembled from existing S3-Sim full-ablation rollout artifacts.",
        "The files are organized as Hugging Face dataset configs/splits and retain OpenAI/TRL-compatible chat structures.",
        "",
        "## Data Files",
        "",
        "| Config | File | Records | Intended use |",
        "|---|---:|---:|---|",
    ]
    for name, info in manifest["files"].items():
        lines.append(
            f"| `{name}` | `{info['path']}` | {info['records']} | {info['intended_use']} |"
        )
    lines.extend(
        [
            "",
            "## Training Notes",
            "",
            "- `conversation_messages` is one full dialogue per sample with a `messages` column.",
            "- `conversation_assistant_turns` is conversational `prompt`/`completion`; use this for completion-only loss on every assistant turn.",
            "- `qa_*` files are QA v2 rewritten where available. `qa_bigtom_tom` currently comes from the pilot v2 rewrite only.",
            "- `qa_alignment_mix` is a deterministic training mixture: all PersonaMem MCQ, 50% PrefEval short-answer, and all rewritten BigTom ToM.",
            "",
            "## Provenance",
            "",
            "Primary scenario constructors: lifelong, highfreq, affective. Random initial-prompt rollout is kept as a separate seed family.",
            "`concerning` and ablation-control outputs are intentionally excluded from the primary release and listed in `provenance/manifest.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_file_info(
    root: Path, relative_path: str, records: int, intended_use: str
) -> dict[str, Any]:
    path = root / relative_path
    return {
        "path": relative_path,
        "records": records,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "intended_use": intended_use,
    }


def style_counter(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter((record.get("metadata") or {}).get("qa_style", "unknown") for record in records)
    )


def assemble(output_dir: Path, seed: int) -> dict[str, Any]:
    conversation_records, conversation_report = load_records(CONVERSATION_SOURCES, REPO)
    conversation_records = stable_declutter_shuffle(
        conversation_records,
        seed=seed,
        fields=["release_region", "scenario_family", "persona_id"],
    )
    assistant_turn_records = build_assistant_turn_records(conversation_records)

    qa_primary_sources = [source for source in QA_SOURCES if source.primary]
    qa_records, qa_report = load_records(qa_primary_sources, REPO)
    qa_by_style = {
        style: stable_declutter_shuffle(
            [
                record
                for record in qa_records
                if (record.get("metadata") or {}).get("qa_style") == style
            ],
            seed=seed,
            fields=["release_region", "scenario_family", "persona_id", "correct_letter"],
        )
        for style in ["personamem_mcq", "prefeval_gen", "bigtom_tom"]
    }

    rng = random.Random(seed)
    prefeval_pool = qa_by_style["prefeval_gen"][:]
    rng.shuffle(prefeval_pool)
    prefeval_target = min(len(prefeval_pool), len(qa_by_style["personamem_mcq"]) // 2)
    qa_mix = (
        qa_by_style["personamem_mcq"] + prefeval_pool[:prefeval_target] + qa_by_style["bigtom_tom"]
    )
    qa_mix = stable_declutter_shuffle(
        qa_mix,
        seed=seed,
        fields=["qa_style", "release_region", "persona_id", "correct_letter"],
    )

    files_to_write = {
        "data/conversations/messages_train.jsonl": conversation_records,
        "data/conversations/assistant_turns_train.jsonl": assistant_turn_records,
        "data/qa/personamem_mcq_train.jsonl": qa_by_style["personamem_mcq"],
        "data/qa/prefeval_gen_train.jsonl": qa_by_style["prefeval_gen"],
        "data/qa/bigtom_tom_train.jsonl": qa_by_style["bigtom_tom"],
        "data/mixes/qa_alignment_mix_train.jsonl": qa_mix,
    }
    for relative_path, records in files_to_write.items():
        write_jsonl(output_dir / relative_path, records)

    manifest = {
        "name": "s3-sim-final-release",
        "seed": seed,
        "schema_version": "2026-05-02",
        "sources": {
            "conversation": [asdict(source) for source in CONVERSATION_SOURCES],
            "qa_primary": [asdict(source) for source in qa_primary_sources],
            "excluded": EXCLUDED_SOURCES,
            "seed_files": [
                "data/filterd_refined_profiles/summary_refined_profiles_us.jsonl",
                "data/filterd_refined_profiles/summary_refined_profiles_cn.jsonl",
                "data/filterd_refined_profiles/summary_refined_profiles_jp.jsonl",
                "data/initial_prompts/prompts_mixed_taged.jsonl",
                "user_simulator/prompts/simulator_lifelong_scenario_constructor.yaml",
                "user_simulator/prompts/simulator_highfreq_scenario_constructor.yaml",
                "user_simulator/prompts/simulator_affective_scenario_constructor.yaml",
            ],
        },
        "reports": {
            "conversation": conversation_report,
            "qa": qa_report,
            "qa_mix_style_counts": style_counter(qa_mix),
        },
        "files": {},
    }
    file_intended_use = {
        "conversation_messages": "Full conversational SFT with TRL `assistant_only_loss=True` when supported.",
        "conversation_assistant_turns": "Prompt/completion SFT; completion-only loss over every assistant turn.",
        "qa_personamem_mcq": "Persona memory MCQ reasoning supervision.",
        "qa_prefeval_gen": "Short personalized response supervision.",
        "qa_bigtom_tom": "Theory-of-mind QA reasoning supervision.",
        "qa_alignment_mix": "Default QA mixture for alignment-stage SFT.",
    }
    file_keys = [
        ("conversation_messages", "data/conversations/messages_train.jsonl"),
        ("conversation_assistant_turns", "data/conversations/assistant_turns_train.jsonl"),
        ("qa_personamem_mcq", "data/qa/personamem_mcq_train.jsonl"),
        ("qa_prefeval_gen", "data/qa/prefeval_gen_train.jsonl"),
        ("qa_bigtom_tom", "data/qa/bigtom_tom_train.jsonl"),
        ("qa_alignment_mix", "data/mixes/qa_alignment_mix_train.jsonl"),
    ]
    for key, relative_path in file_keys:
        manifest["files"][key] = collect_file_info(
            output_dir,
            relative_path,
            len(files_to_write[relative_path]),
            file_intended_use[key],
        )

    write_json(output_dir / "provenance" / "manifest.json", manifest)
    write_readme(output_dir / "README.md", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = assemble(args.output_dir, args.seed)
    summary = {
        "output_dir": str(args.output_dir),
        "files": {
            name: {
                "records": info["records"],
                "path": info["path"],
                "sha256": info["sha256"],
            }
            for name, info in manifest["files"].items()
        },
        "qa_mix_style_counts": manifest["reports"]["qa_mix_style_counts"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
