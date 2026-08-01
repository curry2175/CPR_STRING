from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from vrg.issue_resolution import apply_resolution_semantics, contains_unsafe_claim, graph_from_card
from vrg.lightweight_uplift import LightweightAuditOutput, run_lightweight_uplift


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "cache_seed" / "v030_result_20260801_182810.json"


def test_v031_resolution_semantics_separates_clean_from_flawed_on_regression_set():
    payload = json.loads(SEED.read_text(encoding="utf-8"))
    for row in payload["cases"]:
        card = row["methods"]["small_graph_structure"]["graph_card"]
        graph = graph_from_card(card, row["text"])
        assert graph is not None
        refreshed = apply_resolution_semantics(graph, row["text"])
        active = refreshed["actionable_issues"]
        if row["label"] == "clean":
            assert active == [], row["case_id"]
        else:
            assert active, row["case_id"]
            assert set(row["gold_issue_types"]) & {x["issue_type"] for x in active}


def test_v031_revision_risk_detection_is_negation_aware():
    assert contains_unsafe_claim("Treatment protects against death.", "causal_overclaim") is True
    assert contains_unsafe_claim(
        "The evidence does not establish that Treatment protects against death.",
        "causal_overclaim",
    ) is False
    assert contains_unsafe_claim(
        "The finding is uncertain and does not establish a large benefit.",
        "magnitude_inflation",
    ) is False


class _Responses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        user = " ".join(str(kwargs.get("input")) .split())
        # The cache test concerns routing/call counts. Return a valid audit object
        # whose decision follows the explicit v031 card state.
        has_active = "actionable_defect=true state=active" in user
        parsed = LightweightAuditOutput(
            has_problem=has_active,
            vulnerable_conclusion="",
            issue_types=[],
            evidence_spans=[],
            revised_conclusion="",
            audit_summary="cache-routing test",
        )
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(
            model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        )
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            status="completed",
            output=[message],
            usage=usage,
        )


class _Client:
    def __init__(self):
        self.responses = _Responses()


def test_v031_reuses_completed_generations_and_calls_only_changed_insight(tmp_path):
    client = _Client()
    result = run_lightweight_uplift(
        benchmark_path=ROOT / "data" / "discussion_uplift_benchmark_v029.jsonl",
        output_root=tmp_path / "outputs",
        small_model="gpt-5.4-nano",
        reference_model="gpt-5.4-mini",
        include_reference=True,
        variants=["distance_16"],
        limit=12,
        reasoning_effort="low",
        max_output_tokens=2200,
        seed=2029,
        include_graph_structure_ablation=True,
        reuse_result_paths=[SEED],
        generation_cache_path=tmp_path / "generation_cache.json",
        client=client,
    )
    assert len(client.responses.calls) == 12
    assert result["cache_summary"]["cache_hits_by_method"] == {
        "small_direct": 12,
        "small_checklist": 12,
        "small_graph_structure": 12,
        "reference_direct": 12,
    }
    assert result["graph_extraction_summary"]["cache_reused_cases_this_run"] == 12
    assert result["graph_extraction_summary"]["api_extracted_cases_this_run"] == 0
    assert result["summary"]["actual_api_calls_this_run"] == 12
