from __future__ import annotations

from types import SimpleNamespace

from vrg.discussion_architecture import (
    DiscussionCompilerOutput,
    SpecialistReviewOutput,
)
from vrg.discussion_graph import (
    DiscussionEdge,
    DiscussionNode,
    generate_discussion_graph,
)


class _Responses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.outputs.pop(0)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140})
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            status="completed",
            output=[message],
            usage=usage,
        )


class _Client:
    def __init__(self, outputs):
        self.responses = _Responses(outputs)


def _compiler_output() -> DiscussionCompilerOutput:
    return DiscussionCompilerOutput(
        paragraph_summary="The paragraph presents an observation and a conclusion.",
        nodes=[
            DiscussionNode(
                id="d1",
                sentence_index=1,
                source_text="Treatment was associated with lower risk.",
                plain_meaning="Treatment and lower risk were associated.",
                role="observation",
                assertion_type="association",
                polarity="positive",
                certainty="observed",
            ),
            DiscussionNode(
                id="d2",
                sentence_index=2,
                source_text="Therefore, treatment prevents the outcome.",
                plain_meaning="Treatment is claimed to prevent the outcome.",
                role="conclusion",
                assertion_type="causal",
                polarity="positive",
                certainty="concludes",
            ),
        ],
        edges=[
            DiscussionEdge(
                id="e1",
                source="d1",
                target="d2",
                relation="supports",
                rationale="The association is used to support the conclusion.",
            )
        ],
    )


def test_v036_graph_native_default_uses_compiler_and_independent_specialists():
    logic = SpecialistReviewOutput(
        specialist="logic",
        reviewed_node_ids=["d1", "d2"],
        findings=[],
        review_summary="No formal contradiction.",
    )
    evidence = SpecialistReviewOutput(
        specialist="evidence",
        reviewed_node_ids=["d1", "d2"],
        findings=[],
        review_summary="No additional finding in this fixture.",
    )
    client = _Client([_compiler_output(), logic, evidence])

    result = generate_discussion_graph(
        "Treatment was associated with lower risk. Therefore, treatment prevents the outcome.",
        model="gpt-5.6",
        reasoning_effort="low",
        client=client,
        architecture_mode="graph_native_multi_agent",
    )

    assert result["schema_version"] == "0.27.0"
    assert result["prompt_version"] == "discussion_graph_native_multi_agent_v1"
    assert result["api_call_count"] == 3
    assert result["usage"]["total_tokens"] == 420
    assert [call["text_format"].__name__ for call in client.responses.calls] == [
        "DiscussionCompilerOutput",
        "SpecialistReviewOutput",
        "SpecialistReviewOutput",
    ]
    # Specialists see the compiled graph rather than the raw paragraph prompt.
    specialist_user = client.responses.calls[1]["input"][1]["content"]
    assert "shared reasoning graph" in specialist_user.lower()
    assert '"nodes"' in specialist_user


def test_v036_legacy_mode_keeps_the_original_single_pass_contract():
    from tests.test_mvp import _discussion_fixture, _HybridSequenceClient

    client = _HybridSequenceClient([_discussion_fixture()])
    result = generate_discussion_graph(
        "Treatment G reduced inflammation but the benefit also occurred without inflammation reduction.",
        model="gpt-5.6",
        reasoning_effort="low",
        client=client,
        architecture_mode="legacy_single_pass",
    )
    assert result["schema_version"] == "0.27.0"
    assert result["api_call_count"] == 1
    assert client.responses.calls[0]["text_format"].__name__ == "DiscussionGraphOutput"


def test_v036_public_result_key_contract_matches_legacy_mode():
    from tests.test_mvp import _discussion_fixture, _HybridSequenceClient

    legacy_client = _HybridSequenceClient([_discussion_fixture()])
    legacy = generate_discussion_graph(
        "Treatment G reduced inflammation but the benefit also occurred without inflammation reduction.",
        model="gpt-5.6",
        client=legacy_client,
        architecture_mode="legacy_single_pass",
    )

    compiler = DiscussionCompilerOutput(
        paragraph_summary=_discussion_fixture().paragraph_summary,
        nodes=_discussion_fixture().nodes,
        edges=_discussion_fixture().edges,
    )
    logic = SpecialistReviewOutput(specialist="logic", findings=[], reviewed_node_ids=["claim-a", "claim-b", "claim-c"])
    evidence = SpecialistReviewOutput(specialist="evidence", findings=[], reviewed_node_ids=["claim-a", "claim-b", "claim-c"])
    new_client = _Client([compiler, logic, evidence])
    graph_native = generate_discussion_graph(
        "Treatment G reduced inflammation but the benefit also occurred without inflammation reduction.",
        model="gpt-5.6",
        client=new_client,
        architecture_mode="graph_native_multi_agent",
    )

    assert set(graph_native) == set(legacy)
    assert set(graph_native["summary"]) == set(legacy["summary"])
    assert set(graph_native["graph_metrics"]) == set(legacy["graph_metrics"])
