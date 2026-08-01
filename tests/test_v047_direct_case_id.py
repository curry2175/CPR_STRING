from __future__ import annotations

import json

from vrg.ragtruth_localization import load_ragtruth_cases


def test_v047_include_case_ids_loads_only_requested_case(tmp_path):
    source_path = tmp_path / "source_info.jsonl"
    response_path = tmp_path / "response.jsonl"
    source_rows = [
        {"source_id": "s1", "task_type": "QA", "source": "x", "source_info": {"question": "Q1", "passages": ["A"]}},
        {"source_id": "s2", "task_type": "QA", "source": "x", "source_info": {"question": "Q2", "passages": ["B"]}},
    ]
    response_rows = [
        {"id": "12046", "source_id": "s1", "response": "A", "labels": [], "split": "test", "quality": "good"},
        {"id": "99999", "source_id": "s2", "response": "B", "labels": [], "split": "test", "quality": "good"},
    ]
    source_path.write_text("\n".join(json.dumps(x) for x in source_rows) + "\n", encoding="utf-8")
    response_path.write_text("\n".join(json.dumps(x) for x in response_rows) + "\n", encoding="utf-8")

    cases, sampling = load_ragtruth_cases(
        response_path,
        source_path,
        split="",
        quality="",
        task_types=["QA"],
        limit=0,
        include_case_ids={"12046"},
    )
    assert [row["case_id"] for row in cases] == ["12046"]
    assert sampling["included_case_ids_requested"] == ["12046"]
    assert sampling["skipped"]["not_requested_case_id"] == 1
