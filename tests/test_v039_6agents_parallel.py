from __future__ import annotations

import time
from threading import Lock
from types import SimpleNamespace

from vrg.discussion_architecture_6agents import (
    MissingNodeProposal,
    PilotAssumptionReviewOutput,
    PilotCompilerOutput,
    PilotFinding,
    PilotGraphPatch,
    PilotJudgeOutput,
    PilotSpecialistReviewOutput,
    run_graph_native_6agents_chunk,
)
from vrg.discussion_graph import DiscussionEdge, DiscussionNode


class _RoutedResponses:
    def __init__(self, outputs, *, specialist_delay: float = 0.0):
        self.outputs = list(outputs)
        self.calls = []
        self.specialist_delay = specialist_delay
        self.specialist_start_times: dict[str, float] = {}
        self._lock = Lock()

    @staticmethod
    def _specialist_from_system(system: str) -> str | None:
        for name in ("evidence", "logic", "target"):
            if f"You are the {name.capitalize()} Agent" in system:
                return name
        return None

    def _take_matching(self, schema, specialist):
        for index, candidate in enumerate(self.outputs):
            if not isinstance(candidate, schema):
                continue
            if specialist is not None and getattr(candidate, "specialist", None) != specialist:
                continue
            return self.outputs.pop(index)
        raise AssertionError(f"No queued output for {schema.__name__} specialist={specialist}")

    def parse(self, **kwargs):
        schema = kwargs["text_format"]
        system = kwargs["input"][0]["content"]
        specialist = self._specialist_from_system(system)
        with self._lock:
            self.calls.append(kwargs)
            parsed = self._take_matching(schema, specialist)
            if specialist:
                self.specialist_start_times[specialist] = time.perf_counter()
        if specialist and self.specialist_delay:
            time.sleep(self.specialist_delay)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        usage = SimpleNamespace(model_dump=lambda: {
            "input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
        })
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}", model=kwargs["model"], status="completed",
            output=[SimpleNamespace(type="message", content=[item])], usage=usage,
        )


class _Client:
    def __init__(self, outputs, *, specialist_delay: float = 0.0):
        self.responses = _RoutedResponses(outputs, specialist_delay=specialist_delay)


def _node(node_id: str, text: str, role: str) -> DiscussionNode:
    return DiscussionNode(
        id=node_id, sentence_index=1, source_text=text, plain_meaning=text,
        role=role, assertion_type="descriptive", polarity="positive", certainty="reported",
    )


def _compiler() -> PilotCompilerOutput:
    return PilotCompilerOutput(
        paragraph_summary="A premise supports a conclusion.",
        nodes=[_node("d1", "A is stated.", "observation"), _node("d2", "B is concluded.", "conclusion")],
        edges=[DiscussionEdge(id="e1", source="d1", target="d2", relation="supports", rationale="Used as support")],
    )


def _review(name: str, findings=None) -> PilotSpecialistReviewOutput:
    return PilotSpecialistReviewOutput(
        specialist=name, reviewed_node_ids=["d1", "d2"], reviewed_edge_ids=["e1"],
        findings=list(findings or []),
    )


def test_v039_core_specialists_start_in_parallel():
    client = _Client([
        _compiler(), _review("evidence"), _review("logic"), _review("target"),
    ], specialist_delay=0.12)
    _, responses, _, trace = run_graph_native_6agents_chunk(
        "A is stated. B is concluded.", model="gpt-5.6", reasoning_effort="low",
        max_output_tokens=3000, custom_instruction="", client=client,
    )
    starts = client.responses.specialist_start_times
    assert set(starts) == {"evidence", "logic", "target"}
    assert max(starts.values()) - min(starts.values()) < 0.08
    assert len(responses) == 4
    assert trace["execution"] == "parallel_core_specialists"
    assert any(row["stage"] == "parallel:evidence_logic_target" for row in trace["stages"])


def test_v039_assumption_not_called_without_explicit_missing_premise_proposal():
    generic = PilotFinding(
        id="f1", verdict="invalid", canonical_issue_type="unsupported_inference",
        public_issue_type="evidence_strength_mismatch", severity="medium",
        title="Unsupported inference", node_ids=["d1", "d2"], edge_ids=["e1"],
        explanation="The inference is weak.", logical_pattern="A -> B", confidence=0.85,
    )
    client = _Client([
        _compiler(), _review("evidence"), _review("logic", [generic]), _review("target"),
    ])
    _, responses, _, trace = run_graph_native_6agents_chunk(
        "A is stated. B is concluded.", model="gpt-5.6", reasoning_effort="low",
        max_output_tokens=3000, custom_instruction="", client=client,
    )
    assert len(responses) == 4
    assumption_stage = next(row for row in trace["stages"] if row["stage"] == "agent:assumption")
    assert assumption_stage["status"] == "not_needed"


def test_v039_assumption_called_for_edge_tied_missing_premise_even_with_broader_label():
    missing = PilotFinding(
        id="f1", verdict="invalid", canonical_issue_type="unsupported_inference",
        public_issue_type="evidence_strength_mismatch", severity="medium",
        title="Inference needs a premise", node_ids=["d1", "d2"], edge_ids=["e1"],
        explanation="A condition is unstated.", logical_pattern="A + hidden premise -> B", confidence=0.8,
        missing_node_proposals=[MissingNodeProposal(
            text="A required condition holds.", required_for_edge_id="e1",
            rationale="The edge depends on this condition.", confidence=0.8,
        )],
    )
    client = _Client([
        _compiler(), _review("evidence"), _review("logic", [missing]), _review("target"),
        PilotAssumptionReviewOutput(assessments=[]),
    ])
    _, responses, _, trace = run_graph_native_6agents_chunk(
        "A is stated. B is concluded.", model="gpt-5.6", reasoning_effort="low",
        max_output_tokens=3000, custom_instruction="", client=client,
    )
    assert len(responses) == 5
    assumption_stage = next(row for row in trace["stages"] if row["stage"] == "agent:assumption")
    assert assumption_stage["status"] == "ok"
    assert assumption_stage["candidate_count"] == 1


def test_v039_judge_uses_moderate_material_uncertainty_trigger():
    uncertain = PilotFinding(
        id="f1", verdict="uncertain", canonical_issue_type="unsupported_inference",
        public_issue_type="evidence_strength_mismatch", severity="medium",
        title="Material uncertainty", node_ids=["d1", "d2"], edge_ids=["e1"],
        explanation="The conclusion may or may not follow.", logical_pattern="A ? B", confidence=0.72,
    )
    judge = PilotJudgeOutput(accepted_finding_ids=["logic_f1"], rationale="Keep the material uncertainty.")
    client = _Client([
        _compiler(), _review("evidence"), _review("logic", [uncertain]), _review("target"), judge,
    ])
    _, responses, _, trace = run_graph_native_6agents_chunk(
        "A is stated. B is concluded.", model="gpt-5.6", reasoning_effort="low",
        max_output_tokens=3000, custom_instruction="", client=client,
    )
    assert len(responses) == 5
    judge_stage = next(row for row in trace["stages"] if row["stage"] == "agent:judge")
    assert judge_stage["status"] == "ok"
    assert "material_uncertainty" in judge_stage["trigger_reasons"]


def test_v039_judge_runs_for_graph_revision_proposal():
    revision = PilotFinding(
        id="f1", verdict="invalid", canonical_issue_type="wrong_edge_type",
        public_issue_type="other", severity="medium", title="Edge type is too strong",
        node_ids=["d1", "d2"], edge_ids=["e1"], explanation="Use a weaker edge.",
        logical_pattern="derives -> supports", confidence=0.9,
        patches=[PilotGraphPatch(
            operation="change_edge_type", target_id="e1", edge_ids=["e1"],
            proposed_status="conditional", proposed_edge_relation="supports_weakly",
            rationale="The evidence supports but does not entail the conclusion.", confidence=0.9,
        )],
    )
    judge = PilotJudgeOutput(accepted_finding_ids=["logic_f1"], rationale="Approve revision review.")
    client = _Client([
        _compiler(), _review("evidence"), _review("logic", [revision]), _review("target"), judge,
    ])
    _, responses, _, trace = run_graph_native_6agents_chunk(
        "A is stated. B is concluded.", model="gpt-5.6", reasoning_effort="low",
        max_output_tokens=3000, custom_instruction="", client=client,
    )
    assert len(responses) == 5
    assert "graph_revision_proposed" in trace["judge_trigger_reasons"]
