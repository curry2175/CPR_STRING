from __future__ import annotations

import json
from pathlib import Path

from vrg.ragtruth_localization import (
    ClaimNode,
    LightClaimGraphOutput,
    _predictions_from_graph,
    build_evidence_card,
    load_ragtruth_cases,
    locate_exact_quote,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "ragtruth_fixture"


def test_v033_numeric_boundary_rejects_decimal_substring():
    response = "With a rating of 3.5 stars, the restaurant is popular."
    assert locate_exact_quote(response, "5 stars", "a1") is None
    assert locate_exact_quote(response, "3.5 stars", "a1") == (17, 26)


def test_v033_graph_projects_minimal_problem_text_not_whole_claim():
    response = "It appears to be a popular Mexican restaurant."
    parsed = LightClaimGraphOutput(claims=[ClaimNode(
        id="c1",
        sentence_id="a1",
        text="a popular Mexican restaurant",
        relation="unsupported",
        problem_text="popular",
        evidence_ids=[],
    )])
    predictions, details = _predictions_from_graph(parsed, response)
    assert [item["text"] for item in predictions] == ["popular"]
    assert details["problem_text_fallback_count"] == 0
    assert details["graph"]["nodes"][0]["resolved_problem_text"] == "popular"


def test_v033_full_evidence_keeps_paths_and_all_units():
    source = {
        "question": "What is the capital?",
        "record": {"city": "Paris", "rating": 3.5, "wifi": "free"},
    }
    card = build_evidence_card(source, "Paris has free WiFi.", max_context_chars=10_000, force_full=True)
    assert card["mode"] == "full_required"
    assert card["full_evidence_used"] is True
    assert card["selected_unit_count"] == card["all_unit_count"]
    assert "source.record.wifi" in card["text"]


def test_v033_loader_excludes_previously_evaluated_ids():
    responses = [json.loads(line) for line in (FIXTURE / "response.jsonl").read_text().splitlines() if line.strip()]
    excluded = {str(responses[0]["id"])}
    rows, info = load_ragtruth_cases(
        FIXTURE / "response.jsonl",
        FIXTURE / "source_info.jsonl",
        limit=0,
        task_types=["QA", "Summary"],
        exclude_case_ids=excluded,
    )
    assert all(row["case_id"] not in excluded for row in rows)
    assert info["skipped"]["excluded_case_id"] == 1
    assert info["excluded_case_ids_requested"] == 1


def test_v033_loader_can_require_complete_evidence(tmp_path):
    response_path = tmp_path / "response.jsonl"
    source_path = tmp_path / "source_info.jsonl"
    source_path.write_text(json.dumps({
        "source_id": "s1",
        "task_type": "QA",
        "source": "fixture",
        "source_info": {"question": "Q?", "passages": "passage 1: " + "x" * 500},
    }) + "\n", encoding="utf-8")
    response_path.write_text(json.dumps({
        "id": "r1", "source_id": "s1", "response": "A.", "labels": [], "split": "test", "quality": "good"
    }) + "\n", encoding="utf-8")
    rows, info = load_ragtruth_cases(
        response_path,
        source_path,
        task_types=["QA"],
        limit=0,
        require_full_evidence=True,
        max_context_chars=100,
    )
    assert rows == []
    assert info["skipped"]["full_evidence_too_long"] == 1
