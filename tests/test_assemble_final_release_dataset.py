from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "assemble_final_release_dataset.py"
_SPEC = importlib.util.spec_from_file_location("assemble_final_release_dataset", _PATH)
assemble_final_release_dataset = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = assemble_final_release_dataset
_SPEC.loader.exec_module(assemble_final_release_dataset)


def test_build_assistant_turn_records_uses_each_assistant_message() -> None:
    records = [
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ],
            "metadata": {"sample_id": "conversation_abc", "persona_id": "p1"},
        }
    ]

    turn_records = assemble_final_release_dataset.build_assistant_turn_records(records)

    assert len(turn_records) == 2
    assert turn_records[0]["prompt"] == records[0]["messages"][:2]
    assert turn_records[0]["completion"] == [{"role": "assistant", "content": "a1"}]
    assert turn_records[1]["prompt"] == records[0]["messages"][:4]
    assert turn_records[1]["completion"] == [{"role": "assistant", "content": "a2"}]
    assert turn_records[1]["metadata"]["assistant_turn_index"] == 2
    assert turn_records[1]["metadata"]["assistant_turn_count"] == 2
