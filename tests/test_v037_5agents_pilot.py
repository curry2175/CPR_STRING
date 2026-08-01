from __future__ import annotations

from types import SimpleNamespace

from vrg.discussion_architecture_5agents import (
    NODE_ROLE_VOCAB,
    EDGE_RELATION_VOCAB,
    PilotCompilerOutput,
    PilotFinding,
    PilotJudgeOutput,
    PilotSpecialistReviewOutput,
)
from vrg.discussion_graph import DiscussionEdge, DiscussionNode, generate_discussion_graph


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


def _compiler_output() -> PilotCompilerOutput:
    return PilotCompilerOutput(
        paragraph_summary="The paragraph presents an association and a causal conclusion.",
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


def _empty_review(name: str) -> PilotSpecialistReviewOutput:
    return PilotSpecialistReviewOutput(
        specialist=name,
        reviewed_node_ids=["d1", "d2"],
        reviewed_edge_ids=["e1"],
        findings=[],
        review_summary="No issue in fixture.",
    )


def test_v037_default_runs_four_base_calls_and_keeps_public_output_contract():
    client = _Client([
        _compiler_output(),
        _empty_review("evidence"),
        _empty_review("logic"),
        _empty_review("target"),
    ])
    text = "Treatment was associated with lower risk. Therefore, treatment prevents the outcome."
    result = generate_discussion_graph(text, model="gpt-5.6", client=client, architecture_mode="graph_native_5agents_pilot")

    assert result["schema_version"] == "0.27.0"
    assert result["prompt_version"] == "discussion_graph_native_5agents_pilot_v1"
    assert result["api_call_count"] == 4
    assert result["usage"]["total_tokens"] == 560
    assert [call["text_format"].__name__ for call in client.responses.calls] == [
        "PilotCompilerOutput",
        "PilotSpecialistReviewOutput",
        "PilotSpecialistReviewOutput",
        "PilotSpecialistReviewOutput",
    ]

    compiler_system = client.responses.calls[0]["input"][0]["content"]
    for value in NODE_ROLE_VOCAB:
        assert value in compiler_system
    for value in EDGE_RELATION_VOCAB:
        assert value in compiler_system

    evidence_user = client.responses.calls[1]["input"][1]["content"]
    logic_user = client.responses.calls[2]["input"][1]["content"]
    target_user = client.responses.calls[3]["input"][1]["content"]
    assert '"source_alignment"' in evidence_user
    assert '"status":"exact"' in evidence_user
    assert '"source_alignment"' not in logic_user
    assert '"source_alignment"' not in target_user


def test_v037_conditional_judge_is_the_fifth_agent_call():
    evidence_finding = PilotFinding(
        id="f1",
        verdict="uncertain",
        canonical_issue_type="modality_distortion",
        public_issue_type="evidence_strength_mismatch",
        severity="medium",
        title="The conclusion may strengthen the source wording",
        node_ids=["d2"],
        explanation="The normalized conclusion is stronger than the quoted wording.",
        logical_pattern="source modality -> stronger conclusion",
        confidence=0.6,
    )
    logic_finding = PilotFinding(
        id="f2",
        verdict="invalid",
        canonical_issue_type="causal_overclaim",
        public_issue_type="causal_overclaim",
        severity="high",
        title="Association is used as a causal conclusion",
        node_ids=["d1", "d2"],
        edge_ids=["e1"],
        explanation="The edge does not establish prevention.",
        logical_pattern="association -> causation",
        confidence=0.9,
    )
    client = _Client([
        _compiler_output(),
        PilotSpecialistReviewOutput(specialist="evidence", findings=[evidence_finding]),
        PilotSpecialistReviewOutput(specialist="logic", findings=[logic_finding]),
        _empty_review("target"),
        PilotJudgeOutput(accepted_finding_ids=["logic_f2"], rejected_finding_ids=["evidence_f1"]),
    ])
    text = "Treatment was associated with lower risk. Therefore, treatment prevents the outcome."
    result = generate_discussion_graph(text, model="gpt-5.6", client=client, architecture_mode="graph_native_5agents_pilot")

    assert result["api_call_count"] == 5
    assert client.responses.calls[-1]["text_format"].__name__ == "PilotJudgeOutput"
    assert any(issue["issue_type"] == "causal_overclaim" for issue in result["issues"])


def test_v037_public_keys_match_legacy_single_pass():
    from tests.test_mvp import _discussion_fixture, _HybridSequenceClient

    legacy_client = _HybridSequenceClient([_discussion_fixture()])
    legacy = generate_discussion_graph(
        "Treatment G reduced inflammation but benefit also occurred without reduction.",
        model="gpt-5.6",
        client=legacy_client,
        architecture_mode="legacy_single_pass",
    )

    compiler = PilotCompilerOutput(
        paragraph_summary=_discussion_fixture().paragraph_summary,
        nodes=_discussion_fixture().nodes,
        edges=_discussion_fixture().edges,
    )
    new_client = _Client([
        compiler,
        PilotSpecialistReviewOutput(specialist="evidence", findings=[]),
        PilotSpecialistReviewOutput(specialist="logic", findings=[]),
        PilotSpecialistReviewOutput(specialist="target", findings=[]),
    ])
    pilot = generate_discussion_graph(
        "Treatment G reduced inflammation but benefit also occurred without reduction.",
        model="gpt-5.6",
        client=new_client,
        architecture_mode="graph_native_5agents_pilot",
    )

    assert set(pilot) == set(legacy)
    assert set(pilot["summary"]) == set(legacy["summary"])
    assert set(pilot["graph_metrics"]) == set(legacy["graph_metrics"])
