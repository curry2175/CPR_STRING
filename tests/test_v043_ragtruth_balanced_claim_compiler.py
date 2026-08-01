from __future__ import annotations

from types import SimpleNamespace

from vrg.ragtruth_dual_graph import (
    AlignmentRecord,
    DualGraphAlignmentOutput,
    ResponseAtomicityCheck,
    ResponseClaimGraphOutput,
    ResponseClaimNode,
    ResponseClaimRepairOutput,
    ResponseClaimReplacement,
    _call_response_graph_compiler,
    _predictions_from_alignment,
    _response_graph_diagnostics,
    _response_graph_prompts,
    _source_graph_prompts,
)


class _LocalRepairResponses:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        output_type = kwargs["text_format"]
        if output_type is ResponseClaimGraphOutput:
            parsed = ResponseClaimGraphOutput(nodes=[
                ResponseClaimNode(
                    id="R1",
                    sentence_id="a1",
                    node_type="quantitative_claim",
                    text="The trial enrolled 120 patients",
                    normalized_claim="The trial enrolled 120 patients",
                ),
                ResponseClaimNode(
                    id="R2",
                    sentence_id="a2",
                    node_type="claim",
                    text="finally",
                    normalized_claim="finally",
                ),
            ])
        elif output_type is ResponseClaimRepairOutput:
            parsed = ResponseClaimRepairOutput(drop_node_ids=["R2"])
        else:
            raise AssertionError(output_type)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            output=[message],
            usage=usage,
        )


class _LocalRepairClient:
    def __init__(self) -> None:
        self.responses = _LocalRepairResponses()


def test_v043_prompt_requires_minimal_complete_propositions_not_lexical_fragments():
    case = {
        "task_instruction": "Answer the question.",
        "response": "Finally, the monster possesses the Water element.",
    }
    system, user = _response_graph_prompts(case)
    joined = system + " " + user
    assert "smallest semantically complete proposition" in joined
    assert "not the shortest lexical span" in joined
    assert "'Water'" in joined
    assert "'finally'" in joined
    assert "Do not split merely because" in joined


def test_v043_source_compiler_is_task_complete_not_compact_by_omission():
    case = {"task_instruction": "Answer the question."}
    evidence = {"text": "[e1] A.\n[e2] B."}
    system, user = _source_graph_prompts(case, evidence)
    joined = system + " " + user
    assert "task-complete" in joined
    assert "not through omission" in joined
    assert "negative facts" in joined


def test_v043_local_repair_drops_only_fragment_and_preserves_valid_node():
    case = {
        "task_instruction": "Summarize the trial.",
        "response": "The trial enrolled 120 patients. Finally.",
    }
    system, user = _response_graph_prompts(case)
    client = _LocalRepairClient()
    record = _call_response_graph_compiler(
        client,
        case=case,
        model="gpt-5.4-nano",
        reasoning_effort="low",
        max_output_tokens=1800,
        system=system,
        user=user,
    )
    graph = ResponseClaimGraphOutput.model_validate(record["parsed"])
    assert len(client.responses.calls) == 2
    assert {call["model"] for call in client.responses.calls} == {"gpt-5.4-nano"}
    assert record["compiler_refinement"]["mode"] == "local_node_patch"
    assert record["compiler_refinement"]["dropped_nodes"] == 1
    assert [node.text for node in graph.nodes] == ["The trial enrolled 120 patients"]


def test_v043_heuristic_compound_warning_alone_does_not_force_refinement():
    graph = ResponseClaimGraphOutput(nodes=[
        ResponseClaimNode(
            id="R1",
            sentence_id="a1",
            node_type="claim",
            text="The treatment was effective and was well tolerated.",
            normalized_claim="The treatment was effective and was well tolerated.",
            atomicity_check=ResponseAtomicityCheck(
                single_verdict_possible=True,
                contains_multiple_independent_claims=False,
                split_required=False,
            ),
        )
    ])
    diagnostics = _response_graph_diagnostics(
        graph,
        "The treatment was effective and was well tolerated.",
    )
    assert diagnostics["heuristically_compound_node_ids"] == ["R1"]
    assert diagnostics["repair_node_ids"] == []
    assert diagnostics["needs_refinement"] is False


def test_v043_discourse_marker_localization_is_discarded():
    response = "Finally, the treatment improved survival."
    graph = ResponseClaimGraphOutput(nodes=[
        ResponseClaimNode(
            id="R1",
            sentence_id="a1",
            node_type="claim",
            text="the treatment improved survival",
            normalized_claim="The treatment improved survival",
        )
    ])
    alignment = DualGraphAlignmentOutput(alignments=[
        AlignmentRecord(
            response_node_id="R1",
            relation="not_found_in_source",
            problem_text="Finally",
            label_type="unsupported",
        )
    ])
    predictions, details = _predictions_from_alignment(alignment, graph, response)
    assert predictions == []
    assert details["discarded_nonclaim_problem_text_count"] == 1


def test_v043_unsupported_single_entity_expands_to_complete_claim():
    response = "The monster possesses the Water element."
    graph = ResponseClaimGraphOutput(nodes=[
        ResponseClaimNode(
            id="R1",
            sentence_id="a1",
            node_type="claim",
            text="The monster possesses the Water element.",
            normalized_claim="The monster possesses the Water element",
            claim_form="complete_sentence",
        )
    ])
    alignment = DualGraphAlignmentOutput(alignments=[
        AlignmentRecord(
            response_node_id="R1",
            relation="not_found_in_source",
            problem_text="Water",
            label_type="unsupported",
        )
    ])
    predictions, details = _predictions_from_alignment(alignment, graph, response)
    assert predictions[0]["text"] == response
    assert predictions[0]["problem_text_expanded_to_complete_claim"] is True
    assert details["problem_text_expanded_to_complete_claim_count"] == 1


def test_v043_direct_entity_contradiction_can_remain_minimal():
    response = "The capital of France is Lyon."
    graph = ResponseClaimGraphOutput(nodes=[
        ResponseClaimNode(
            id="R1",
            sentence_id="a1",
            node_type="claim",
            text=response,
            normalized_claim="The capital of France is Lyon",
            claim_form="complete_sentence",
        )
    ])
    alignment = DualGraphAlignmentOutput(alignments=[
        AlignmentRecord(
            response_node_id="R1",
            relation="contradicted_by",
            problem_text="Lyon",
            label_type="contradiction",
        )
    ])
    predictions, _details = _predictions_from_alignment(alignment, graph, response)
    assert predictions[0]["text"] == "Lyon"
