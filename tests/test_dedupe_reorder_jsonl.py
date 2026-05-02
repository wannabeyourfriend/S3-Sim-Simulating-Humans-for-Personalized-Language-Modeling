from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "dedupe_reorder_jsonl.py"
_SPEC = importlib.util.spec_from_file_location("dedupe_reorder_jsonl", _PATH)
dedupe_reorder_jsonl = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dedupe_reorder_jsonl
_SPEC.loader.exec_module(dedupe_reorder_jsonl)


def _sample(content: str, persona_id: str, scenario_id: str, correct_letter: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"Final Answer: {correct_letter}"},
        ],
        "metadata": {
            "persona_id": persona_id,
            "scenario_id": scenario_id,
            "correct_letter": correct_letter,
        },
    }


def test_process_jsonl_dedupes_messages_and_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    report_path = tmp_path / "output.report.json"

    records = [
        _sample("q1", "p1", "s1", "A"),
        _sample("q2", "p1", "s1", "B"),
        _sample("q1", "p2", "s2", "A"),
        _sample("q3", "p2", "s2", "C"),
    ]
    input_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    report = dedupe_reorder_jsonl.process_jsonl(
        input_path=input_path,
        output_path=output_path,
        report_path=report_path,
        dedupe_key="messages",
        seed=7,
        decluster_fields=["metadata.persona_id", "metadata.scenario_id"],
        window_size=8,
    )

    output_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    output_messages = {json.dumps(record["messages"], sort_keys=True) for record in output_records}

    assert report["input_records"] == 4
    assert report["output_records"] == 3
    assert report["duplicates_removed"] == 1
    assert len(output_records) == 3
    assert len(output_messages) == 3
    assert json.loads(report_path.read_text(encoding="utf-8"))["output_records"] == 3
