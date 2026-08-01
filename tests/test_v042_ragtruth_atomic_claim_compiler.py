from __future__ import annotations

from types import SimpleNamespace

from vrg.ragtruth_dual_graph import (
    ResponseAtomicityCheck,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    ResponseClaimRepairOutput,
    ResponseClaimReplacement,
    _call_response_graph_compiler,
    _response_graph_diagnostics,
    _response_graph_prompts,
)


class _AtomicResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            parsed = ResponseClaimGraphOutput(nodes=[
                ResponseClaimNode(
                    id="R1",
                    sentence_id="a1",
                    node_type="claim",
                    text="The trial enrolled 120 patients and improved survival.",
                    normalized_claim="The trial enrolled 120 patients and improved survival",
                    atomicity_check=ResponseAtomicityCheck(
                        single_verdict_possible=False,
                        contains_multiple_independent_claims=True,
                        split_required=True,
                    ),
                )
            ])
        else:
            parsed = ResponseClaimRepairOutput(replacements=[
                ResponseClaimReplacement(
                    original_node_id="R1",
                    replacement_nodes=[
                        ResponseClaimNode(
                            id="R1a",
                            sentence_id="a1",
                            node_type="quantitative_claim",
                            text="The trial enrolled 120 patients",
                            normalized_claim="The trial enrolled 120 patients",
                        ),
                        ResponseClaimNode(
                            id="R1b",
                            sentence_id="a1",
                            node_type="claim",
                            text="improved survival",
                            normalized_claim="The trial improved survival",
                            inherited_context="The trial",
                            claim_form="shared_subject_clause",
                        ),
                    ],
                )
            ])
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            output=[message],
            usage=usage,
        )


class _AtomicClient:
    def __init__(self) -> None:
        self.responses = _AtomicResponses()


def test_v042_compiler_refines_compound_sentence_node_into_atomic_claims():
    case = {
        "task_instruction": "Summarize the trial.",
        "response": "The trial enrolled 120 patients and improved survival.",
    }
    system, user = _response_graph_prompts(case)
    client = _AtomicClient()
    record = _call_response_graph_compiler(
        client,
        case=case,
        model="gpt-5.4-nano",
        reasoning_effort="low",
        max_output_tokens=2000,
        system=system,
        user=user,
    )

    graph = ResponseClaimGraphOutput.model_validate(record["parsed"])
    assert len(client.responses.calls) == 2
    assert record["api_calls"] == 2
    assert record["compiler_refinement"]["attempted"] is True
    assert record["compiler_refinement"]["mode"] == "local_node_patch"
    assert record["compiler_refinement"]["remaining_quality_warning"] is False
    assert [node.text for node in graph.nodes] == [
        "The trial enrolled 120 patients",
        "improved survival",
    ]
    assert graph.nodes[1].normalized_claim == "The trial improved survival"
    assert graph.nodes[1].inherited_context == "The trial"

    diagnostics = _response_graph_diagnostics(graph, case["response"])
    assert diagnostics["claim_count"] == 2
    assert diagnostics["sentence_count"] == 1
    assert diagnostics["claims_per_sentence"] == 2.0
    assert diagnostics["whole_sentence_node_count"] == 0
    assert diagnostics["needs_refinement"] is False
    assert diagnostics["sentence_containers"][0]["claim_node_ids"] == ["R1", "R2"]


def test_v042_atomicity_heuristic_does_not_split_single_combined_outcome_term():
    graph = ResponseClaimGraphOutput(nodes=[
        ResponseClaimNode(
            id="R1",
            sentence_id="a1",
            node_type="claim",
            text="Morbidity and mortality were reduced.",
            normalized_claim="Morbidity and mortality were reduced",
        )
    ])
    diagnostics = _response_graph_diagnostics(graph, "Morbidity and mortality were reduced.")
    assert diagnostics["heuristically_compound_node_ids"] == []
    assert diagnostics["needs_refinement"] is False


def test_v042_per_case_console_reports_compiler_granularity(tmp_path):
    from pathlib import Path
    from test_v040_ragtruth_dual_graph import _Client
    from vrg.ragtruth_dual_graph import run_ragtruth_raw_vs_dual_graph

    root = Path(__file__).resolve().parents[1]
    fixture = root / "data" / "ragtruth_fixture"
    messages: list[str] = []
    result = run_ragtruth_raw_vs_dual_graph(
        response_path=fixture / "response.jsonl",
        source_path=fixture / "source_info.jsonl",
        output_root=tmp_path / "outputs",
        model="gpt-5.4-nano",
        task_types=["QA", "Summary"],
        limit=1,
        seed=1,
        require_full_evidence=True,
        generation_cache_path=tmp_path / "cache.json",
        client=_Client(),
        progress=messages.append,
        print_case_comparison=True,
    )

    text = "\n".join(messages)
    assert "Compiler" in text
    assert "claims=" in text
    assert "claims/sentence=" in text
    assert "whole-sentence nodes=" in text
    graph_summary = result["method_summaries"]["nano_dual_graph"]
    assert "mean_response_claim_node_chars" in graph_summary
    assert "compiler_refinement_case_rate_percent" in graph_summary
