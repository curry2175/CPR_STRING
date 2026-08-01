from __future__ import annotations

import json
from pathlib import Path

from optimize_alignment_thresholds import (
    ERROR_RELATIONS,
    _evaluate_config,
    _load_rows_tolerant,
    _prepare_cases,
    _search,
)
from vrg.ragtruth_dual_graph import (
    GRAPH_METHOD,
    RAW_METHOD,
    AlignmentRecord,
    DualGraphAlignmentOutput,
    ResponseAtomicityCheck,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    ResponseCoverageCheck,
)


def _row(case_id: str, response: str, *, gold: list[dict], confidence: float, relation: str) -> dict:
    graph = ResponseClaimGraphOutput(
        nodes=[
            ResponseClaimNode(
                id="R1",
                sentence_id="a1",
                node_type="claim",
                text=response,
                normalized_claim=response,
                claim_form="complete_sentence",
                evaluation_eligible=True,
                atomicity_check=ResponseAtomicityCheck(),
            )
        ],
        edges=[],
        coverage_check=ResponseCoverageCheck(),
    )
    alignment = DualGraphAlignmentOutput(
        alignments=[
            AlignmentRecord(
                response_node_id="R1",
                relation=relation,
                problem_text=response,
                label_type="unsupported",
                confidence=confidence,
                explanation="test",
            )
        ]
    )
    gold_positive = bool(gold)
    raw_scores = {
        "char_tp": 0,
        "char_fp": 0,
        "char_fn": sum(item["end"] - item["start"] for item in gold),
        "char_precision": 0.0 if gold_positive else 1.0,
        "char_recall": 0.0 if gold_positive else 1.0,
        "char_f1": 0.0 if gold_positive else 1.0,
        "gold_has_hallucination": gold_positive,
        "predicted_has_hallucination": False,
    }
    return {
        "case_id": case_id,
        "response": response,
        "gold_labels": gold,
        "methods": {
            RAW_METHOD: {"status": "ok", "predicted_spans": [], "scores": raw_scores},
            GRAPH_METHOD: {
                "status": "ok",
                "predicted_spans": [],
                "scores": raw_scores,
                "details": {},
                "generation_records": [
                    {"component": "response_graph", "status": "ok", "parsed": graph.model_dump()},
                    {
                        "component": "alignment",
                        "status": "ok",
                        "parsed": alignment.model_dump(),
                        "prompt_version": "v046-dual-graph-conservative-factuality-gated-alignment",
                    },
                ],
            },
        },
    }


def test_optimizer_finds_threshold_between_clean_fp_and_true_positive():
    clean_text = "This clean sentence should not be emitted."
    hall_text = "Unsupported water loss causes the change."
    rows = [
        _row("clean", clean_text, gold=[], confidence=0.40, relation="not_found_in_source"),
        _row(
            "hall",
            hall_text,
            gold=[{"start": 0, "end": len(hall_text), "text": hall_text, "label_type": "unsupported"}],
            confidence=0.60,
            relation="not_found_in_source",
        ),
    ]
    cases, skipped = _prepare_cases(rows)
    assert skipped == 0
    global_best, optimized, _top, _curve, _meta = _search(
        cases,
        step=0.10,
        random_starts=0,
        max_passes=2,
        seed=1,
        progress=False,
    )
    assert global_best["metrics"]["char_f1"] == 1.0
    assert optimized["metrics"]["char_f1"] == 1.0
    threshold = optimized["config"]["thresholds"]["not_found_in_source"]
    assert 0.40 < threshold <= 0.60


def test_disabled_relation_threshold_above_one_suppresses_all_candidates():
    text = "Unsupported claim."
    rows = [
        _row(
            "hall",
            text,
            gold=[{"start": 0, "end": len(text), "text": text, "label_type": "unsupported"}],
            confidence=0.99,
            relation="partially_supported_by",
        )
    ]
    cases, _ = _prepare_cases(rows)
    config = {
        "thresholds": {relation: 1.01 for relation in ERROR_RELATIONS},
        "infer_error_label": True,
        "partial_span_mode": "core",
    }
    metrics = _evaluate_config(cases, config)
    assert metrics["char_tp"] == 0
    assert metrics["char_fn"] == len(text)


def test_tolerant_jsonl_loader_ignores_only_incomplete_last_line(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"case_id": "1"}) + "\n" + '{"case_id":', encoding="utf-8")
    rows = _load_rows_tolerant(path)
    assert rows == [{"case_id": "1"}]


def test_optimizer_recovers_completed_cases_from_generation_cache_when_cases_jsonl_missing(tmp_path: Path):
    import shutil
    from optimize_alignment_thresholds import _recover_cases_from_cache, _load_rows_tolerant
    from tests.test_v040_ragtruth_dual_graph import FIXTURE, _Client
    from vrg.ragtruth_dual_graph import run_ragtruth_raw_vs_dual_graph

    dataset_dir = tmp_path / "data" / "ragtruth"
    dataset_dir.mkdir(parents=True)
    shutil.copy2(FIXTURE / "response.jsonl", dataset_dir / "response.jsonl")
    shutil.copy2(FIXTURE / "source_info.jsonl", dataset_dir / "source_info.jsonl")
    cache_path = tmp_path / "outputs" / "ragtruth_raw_vs_dual_graph_nano" / "generation_cache_v040.json"
    run_ragtruth_raw_vs_dual_graph(
        response_path=dataset_dir / "response.jsonl",
        source_path=dataset_dir / "source_info.jsonl",
        output_root=tmp_path / "temporary_run_outputs",
        model="gpt-5.4-nano",
        task_types=["QA", "Summary"],
        limit=4,
        seed=1,
        require_full_evidence=True,
        generation_cache_path=cache_path,
        client=_Client(),
    )
    recovered = _recover_cases_from_cache(tmp_path, explicit_cache=cache_path)
    rows = _load_rows_tolerant(recovered)
    assert len(rows) == 2  # recovery is intentionally QA-only, matching run_ragqa.bat
    assert all(row.get("recovered_from_generation_cache") for row in rows)
