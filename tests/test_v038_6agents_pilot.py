from __future__ import annotations

from types import SimpleNamespace
from threading import Lock

from vrg.discussion_architecture_6agents import (
    AssumptionAssessment,
    CompilerNodePriority,
    MissingNodeProposal,
    PilotAssumptionReviewOutput,
    PilotCompilerOutput,
    PilotFinding,
    PilotSpecialistReviewOutput,
    compact_compiler_output,
    compact_document_output,
)
from vrg.discussion_graph import (
    DiscussionEdge,
    DiscussionGraphOutput,
    DiscussionIssue,
    DiscussionNode,
    generate_discussion_graph,
)


class _Responses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self._lock = Lock()

    def _take_matching(self, kwargs):
        schema = kwargs["text_format"]
        system = kwargs["input"][0]["content"]
        specialist = None
        for name in ("evidence", "logic", "target"):
            if f"You are the {name.capitalize()} Agent" in system:
                specialist = name
                break
        for index, candidate in enumerate(self.outputs):
            if not isinstance(candidate, schema):
                continue
            if specialist is not None and getattr(candidate, "specialist", None) != specialist:
                continue
            return self.outputs.pop(index)
        raise AssertionError(f"No queued output for {schema.__name__} specialist={specialist}")

    def parse(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            parsed = self._take_matching(kwargs)
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


def _node(node_id: str, text: str, *, role: str = "claim", sentence_index: int = 1) -> DiscussionNode:
    return DiscussionNode(
        id=node_id,
        sentence_index=sentence_index,
        source_text=text,
        plain_meaning=text,
        role=role,
        assertion_type="descriptive",
        polarity="positive",
        certainty="reported",
    )


def _compiler_output() -> PilotCompilerOutput:
    return PilotCompilerOutput(
        paragraph_summary="A family relation is used to infer sex.",
        nodes=[
            _node("d1", "Bob is Mary's son.", role="observation", sentence_index=1),
            _node("d2", "Therefore Bob is male.", role="conclusion", sentence_index=2),
        ],
        edges=[DiscussionEdge(
            id="e1", source="d1", target="d2", relation="supports",
            rationale="The word son is used to support the conclusion.",
        )],
        node_priorities=[
            CompilerNodePriority(node_id="d1", importance=0.9),
            CompilerNodePriority(node_id="d2", importance=1.0),
        ],
    )


def _empty_review(name: str) -> PilotSpecialistReviewOutput:
    return PilotSpecialistReviewOutput(
        specialist=name,
        reviewed_node_ids=["d1", "d2"],
        reviewed_edge_ids=["e1"],
        findings=[],
    )


def _logic_missing_finding() -> PilotFinding:
    return PilotFinding(
        id="f1",
        verdict="invalid",
        canonical_issue_type="missing_premise",
        public_issue_type="evidence_strength_mismatch",
        severity="medium",
        title="An unstated definitional premise is used",
        node_ids=["d1", "d2"],
        edge_ids=["e1"],
        explanation="The inference uses the definition of son.",
        logical_pattern="son -> male",
        confidence=0.9,
        missing_node_proposals=[MissingNodeProposal(
            text="A son is male.",
            required_for_edge_id="e1",
            rationale="This definition licenses the inference.",
            confidence=0.95,
        )],
    )


def test_v038_default_runs_six_agent_architecture_with_conditional_calls(monkeypatch):
    monkeypatch.setenv("DISCUSSION_MAX_NODES_PER_CHUNK", "24")
    client = _Client([
        _compiler_output(),
        _empty_review("evidence"),
        _empty_review("logic"),
        _empty_review("target"),
    ])
    text = "Bob is Mary's son. Therefore Bob is male."
    result = generate_discussion_graph(text, model="gpt-5.6", client=client)

    assert result["schema_version"] == "0.27.0"
    assert result["prompt_version"] == "discussion_graph_native_6agents_balanced_compiler_v045"
    assert result["api_call_count"] == 4
    format_names = [call["text_format"].__name__ for call in client.responses.calls]
    assert format_names.count("PilotCompilerOutput") == 1
    assert format_names.count("PilotSpecialistReviewOutput") == 3
    compiler_prompt = client.responses.calls[0]["input"][0]["content"]
    assert "at most 24 nodes" in compiler_prompt
    assert "never create two nodes" in compiler_prompt


def test_v038_accepted_definition_prevents_false_missing_premise_issue():
    logic = _logic_missing_finding()
    assumption = PilotAssumptionReviewOutput(assessments=[AssumptionAssessment(
        proposal_id="a1",
        logic_finding_id="logic_f1",
        required_for_edge_id="e1",
        disposition="accepted_definition",
        necessity="required",
        proposed_edge_status="valid",
        rationale="The meaning of son includes male sex.",
        confidence=0.98,
    )])
    client = _Client([
        _compiler_output(),
        _empty_review("evidence"),
        PilotSpecialistReviewOutput(specialist="logic", findings=[logic]),
        assumption,
        _empty_review("target"),
    ])
    result = generate_discussion_graph(
        "Bob is Mary's son. Therefore Bob is male.", model="gpt-5.6", client=client
    )

    assert result["api_call_count"] == 5
    assert any(
        call["text_format"].__name__ == "PilotAssumptionReviewOutput"
        for call in client.responses.calls
    )
    assert not any("premise" in issue["title"].lower() for issue in result["issues"])


def test_v038_unsupported_critical_assumption_remains_an_issue():
    compiler = PilotCompilerOutput(
        paragraph_summary="An association is interpreted causally.",
        nodes=[
            _node("d1", "Treatment was associated with lower mortality.", role="observation", sentence_index=1),
            _node("d2", "Therefore treatment reduced mortality.", role="conclusion", sentence_index=2),
        ],
        edges=[DiscussionEdge(id="e1", source="d1", target="d2", relation="supports", rationale="Used causally.")],
    )
    logic = PilotFinding(
        id="f1", verdict="invalid", canonical_issue_type="missing_premise",
        public_issue_type="evidence_strength_mismatch", severity="high",
        title="Causal inference requires an additional premise", node_ids=["d1", "d2"], edge_ids=["e1"],
        explanation="No-confounding is required.", logical_pattern="association -> causation", confidence=0.92,
        missing_node_proposals=[MissingNodeProposal(
            text="There is no important confounding.", required_for_edge_id="e1",
            rationale="The causal conclusion requires exchangeability.", confidence=0.92,
        )],
    )
    assumption = PilotAssumptionReviewOutput(assessments=[AssumptionAssessment(
        proposal_id="a1", logic_finding_id="logic_f1", required_for_edge_id="e1",
        disposition="unsupported_critical_assumption", necessity="required",
        proposed_edge_status="invalid", rationale="The text provides no support for no confounding.", confidence=0.95,
    )])
    client = _Client([
        compiler,
        _empty_review("evidence"),
        PilotSpecialistReviewOutput(specialist="logic", findings=[logic]),
        assumption,
        _empty_review("target"),
    ])
    result = generate_discussion_graph(
        "Treatment was associated with lower mortality. Therefore treatment reduced mortality.",
        model="gpt-5.6", client=client,
    )
    assert any(issue["issue_type"] == "evidence_strength_mismatch" for issue in result["issues"])
    assert any("unjustified assumption" in issue["title"].lower() for issue in result["issues"])


def test_v038_compiler_deduplicates_and_caps_nodes_while_preserving_conclusion():
    nodes = [
        _node("d1", "The primary conclusion is supported.", role="conclusion", sentence_index=1),
        _node("d2", "The primary conclusion is supported.", role="conclusion", sentence_index=2),
    ]
    nodes.extend(_node(f"d{i}", f"Peripheral claim {i}.", sentence_index=i) for i in range(3, 25))
    compiled = PilotCompilerOutput(
        paragraph_summary="Many claims.",
        nodes=nodes,
        edges=[DiscussionEdge(id="e1", source="d3", target="d1", relation="supports", rationale="Support")],
    )
    compacted, meta = compact_compiler_output(
        compiled, node_cap=8, edge_cap=12, dedup_threshold=0.96
    )
    assert len(compacted.nodes) <= 8
    assert meta["duplicate_node_count"] >= 1
    assert meta["pruned_node_count"] > 0
    assert any(node.role == "conclusion" for node in compacted.nodes)
    assert all(edge.source != edge.target for edge in compacted.edges)


def test_v038_document_compaction_deduplicates_cross_chunk_nodes_and_remaps_issues():
    output = DiscussionGraphOutput(
        paragraph_summary="Repeated document graph.",
        nodes=[
            _node("c1_d1", "The main result was positive.", role="conclusion", sentence_index=1),
            _node("c2_d1", "The main result was positive.", role="conclusion", sentence_index=20),
            _node("c1_d2", "Important limitation.", role="limitation", sentence_index=2),
        ],
        edges=[
            DiscussionEdge(id="x1", source="c1_d2", target="c1_d1", relation="limits", rationale="Limits conclusion"),
            DiscussionEdge(id="x2", source="c1_d2", target="c2_d1", relation="limits", rationale="Duplicate link"),
        ],
        issues=[DiscussionIssue(
            id="i1", issue_type="scope_overreach", severity="medium", title="Scope issue",
            node_ids=["c2_d1"], explanation="Repeated conclusion issue.", logical_pattern="scope",
        )],
        overall_assessment="potential_issue",
    )
    compacted, meta = compact_document_output(output, node_cap=10, edge_cap=20)
    assert len(compacted.nodes) == 2
    assert meta["duplicate_node_count"] == 1
    assert compacted.issues and compacted.issues[0].node_ids[0] in {node.id for node in compacted.nodes}
    assert len(compacted.edges) == 1
