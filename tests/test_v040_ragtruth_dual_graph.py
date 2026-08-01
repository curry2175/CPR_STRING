from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from vrg.ragtruth_dual_graph import (
    DualGraphAlignmentOutput,
    AlignmentRecord,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    SourceEvidenceGraphOutput,
    SourceGraphNode,
    run_ragtruth_raw_vs_dual_graph,
)
from vrg.ragtruth_localization import DirectSpanOutput, SpanPrediction


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "ragtruth_fixture"


class _Responses:
    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()

    def parse(self, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
            call_number = len(self.calls)
        user = " ".join(str(kwargs.get("input") or "").split())
        output_type = kwargs["text_format"]

        if output_type is DirectSpanOutput:
            if "capital of France is Lyon" in user:
                parsed = DirectSpanOutput(hallucinated_spans=[SpanPrediction(
                    sentence_id="a1", text="Lyon", label_type="contradiction", evidence_ids=["e1"]
                )])
            elif "cut mortality by 50 percent" in user:
                parsed = DirectSpanOutput(hallucinated_spans=[SpanPrediction(
                    sentence_id="a1", text="cut mortality by 50 percent", label_type="unsupported", evidence_ids=["e1"]
                )])
            else:
                parsed = DirectSpanOutput(hallucinated_spans=[])
        elif output_type is SourceEvidenceGraphOutput:
            if "capital of France" in user:
                parsed = SourceEvidenceGraphOutput(nodes=[
                    SourceGraphNode(id="S1", node_type="source_fact", text="France's capital is Paris", evidence_ids=["e1"]),
                    SourceGraphNode(id="S2", node_type="source_fact", text="Lyon is a city in France", evidence_ids=["e2"]),
                ])
            else:
                parsed = SourceEvidenceGraphOutput(nodes=[
                    SourceGraphNode(id="S1", node_type="quantitative_fact", text="The trial enrolled 100 adults", evidence_ids=["e1"]),
                    SourceGraphNode(id="S2", node_type="source_fact", text="Mortality did not differ significantly", evidence_ids=["e2"]),
                ])
        elif output_type is ResponseClaimGraphOutput:
            if "The capital of France is Lyon" in user:
                parsed = ResponseClaimGraphOutput(nodes=[ResponseClaimNode(
                    id="R1", sentence_id="a1", node_type="claim",
                    text="The capital of France is Lyon.", normalized_claim="France's capital is Lyon"
                )])
            elif "The capital of France is Paris" in user:
                parsed = ResponseClaimGraphOutput(nodes=[ResponseClaimNode(
                    id="R1", sentence_id="a1", node_type="claim",
                    text="The capital of France is Paris.", normalized_claim="France's capital is Paris"
                )])
            elif "cut mortality by 50 percent" in user:
                parsed = ResponseClaimGraphOutput(nodes=[
                    ResponseClaimNode(
                        id="R1", sentence_id="a1", node_type="quantitative_claim",
                        text="The trial enrolled 100 adults", normalized_claim="The trial enrolled 100 adults"
                    ),
                    ResponseClaimNode(
                        id="R2", sentence_id="a1", node_type="causal_claim",
                        text="the treatment cut mortality by 50 percent", normalized_claim="Treatment reduced mortality by 50 percent"
                    ),
                ])
            else:
                parsed = ResponseClaimGraphOutput(nodes=[
                    ResponseClaimNode(
                        id="R1", sentence_id="a1", node_type="quantitative_claim",
                        text="The trial enrolled 100 adults", normalized_claim="The trial enrolled 100 adults"
                    ),
                    ResponseClaimNode(
                        id="R2", sentence_id="a1", node_type="claim",
                        text="found no significant mortality difference", normalized_claim="No significant mortality difference was found"
                    ),
                ])
        elif output_type is DualGraphAlignmentOutput:
            if "The capital of France is Lyon" in user:
                parsed = DualGraphAlignmentOutput(alignments=[AlignmentRecord(
                    response_node_id="R1", source_node_ids=["S1"], relation="contradicted_by",
                    problem_text="Lyon", label_type="contradiction", confidence=0.99
                )])
            elif "The capital of France is Paris" in user:
                parsed = DualGraphAlignmentOutput(alignments=[AlignmentRecord(
                    response_node_id="R1", source_node_ids=["S1"], relation="supported_by",
                    label_type="none", confidence=0.99
                )])
            elif "cut mortality by 50 percent" in user:
                parsed = DualGraphAlignmentOutput(alignments=[
                    AlignmentRecord(
                        response_node_id="R1", source_node_ids=["S1"], relation="supported_by",
                        label_type="none", confidence=0.99
                    ),
                    AlignmentRecord(
                        response_node_id="R2", source_node_ids=["S2"], relation="contradicted_by",
                        problem_text="cut mortality by 50 percent", label_type="contradiction", confidence=0.99
                    ),
                ])
            else:
                parsed = DualGraphAlignmentOutput(alignments=[
                    AlignmentRecord(
                        response_node_id="R1", source_node_ids=["S1"], relation="supported_by",
                        label_type="none", confidence=0.99
                    ),
                    AlignmentRecord(
                        response_node_id="R2", source_node_ids=["S2"], relation="supported_by",
                        label_type="none", confidence=0.99
                    ),
                ])
        else:
            raise AssertionError(output_type)

        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(
            id=f"resp_{call_number}",
            model=kwargs["model"],
            output=[message],
            usage=usage,
        )


class _Client:
    def __init__(self):
        self.responses = _Responses()


def test_v040_both_conditions_use_nano_and_dual_graph_projects_minimal_spans(tmp_path):
    client = _Client()
    cache = tmp_path / "cache.json"
    kwargs = dict(
        response_path=FIXTURE / "response.jsonl",
        source_path=FIXTURE / "source_info.jsonl",
        output_root=tmp_path / "outputs",
        model="gpt-5.4-nano",
        task_types=["QA", "Summary"],
        limit=4,
        seed=1,
        require_full_evidence=True,
        generation_cache_path=cache,
        client=client,
    )
    first = run_ragtruth_raw_vs_dual_graph(**kwargs)

    assert len(client.responses.calls) == 14  # 4 raw + 2 unique source graphs + 4 response graphs + 4 alignments
    assert {call["model"] for call in client.responses.calls} == {"gpt-5.4-nano"}
    assert first["summary"]["actual_api_calls_this_run"] == 14
    assert first["method_summaries"]["nano_raw_direct"]["char_f1_percent"] == 100.0
    assert first["method_summaries"]["nano_dual_graph"]["char_f1_percent"] == 100.0
    assert first["method_summaries"]["nano_dual_graph"]["full_sentence_prediction_rate_percent"] == 0.0
    assert first["method_summaries"]["nano_dual_graph"]["problem_text_fallback_rate_percent"] == 0.0

    lyon_case = next(row for row in first["cases"] if row["case_id"] == "r2")
    graph_spans = lyon_case["methods"]["nano_dual_graph"]["predicted_spans"]
    assert graph_spans[0]["text"] == "Lyon"
    assert graph_spans[0]["response_node_text"] == "The capital of France is Lyon."

    second = run_ragtruth_raw_vs_dual_graph(**kwargs)
    assert len(client.responses.calls) == 14
    assert second["summary"]["actual_api_calls_this_run"] == 0
    assert second["cache_summary"]["cache_hits_by_component"] == {
        "raw_direct": 4,
        "source_graph": 4,
        "response_graph": 4,
        "alignment": 4,
    }


def test_v040_rejects_different_model_between_conditions(tmp_path):
    with pytest.raises(ValueError, match="fixes both conditions"):
        run_ragtruth_raw_vs_dual_graph(
            response_path=FIXTURE / "response.jsonl",
            source_path=FIXTURE / "source_info.jsonl",
            output_root=tmp_path / "outputs",
            model="gpt-5.4-mini",
            limit=1,
            client=_Client(),
        )
