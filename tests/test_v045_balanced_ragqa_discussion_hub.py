from __future__ import annotations

from threading import Lock
from types import SimpleNamespace
from pathlib import Path

from vrg.discussion_architecture_6agents import (
    BALANCED_COMPILER_VERSION,
    DiscussionCompilerRepairOutput,
    PilotCompilerOutput,
    PilotSpecialistReviewOutput,
    build_compiler_prompt,
)
from vrg.discussion_graph import DiscussionNode, generate_discussion_graph


class _Responses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self._lock = Lock()

    def _take(self, schema, system):
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
            schema = kwargs["text_format"]
            system = kwargs["input"][0]["content"]
            parsed = self._take(schema, system)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        usage = SimpleNamespace(model_dump=lambda: {
            "input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
        })
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            status="completed",
            output=[SimpleNamespace(type="message", content=[item])],
            usage=usage,
        )


class _Client:
    def __init__(self, outputs):
        self.responses = _Responses(outputs)


def _node(node_id: str, text: str, *, role: str = "claim") -> DiscussionNode:
    return DiscussionNode(
        id=node_id,
        sentence_index=1,
        source_text=text,
        plain_meaning=text,
        role=role,
        assertion_type="descriptive",
        polarity="positive",
        certainty="reported",
    )


def _review(name: str) -> PilotSpecialistReviewOutput:
    return PilotSpecialistReviewOutput(
        specialist=name,
        reviewed_node_ids=["d1"],
        reviewed_edge_ids=[],
        findings=[],
    )


def test_v045_discussion_compiler_uses_balanced_complete_proposition_contract():
    system, _ = build_compiler_prompt("A sentence.")
    assert "Balanced Semantic Compiler" in system
    assert "smallest semantically complete proposition" in system
    assert "Do not emit isolated entities" in system
    assert "compact through deduplication" in system


def test_v045_discussion_hub_uses_local_compiler_repair_then_all_downstream_agents():
    compiler = PilotCompilerOutput(
        paragraph_summary="A result is reported and a discourse fragment appears.",
        nodes=[
            _node("d1", "The trial improved survival.", role="conclusion"),
            _node("d2", "Finally"),
        ],
        edges=[],
    )
    repair = DiscussionCompilerRepairOutput(drop_node_ids=["d2"], note="Drop discourse marker.")
    client = _Client([
        compiler,
        repair,
        _review("evidence"),
        _review("logic"),
        _review("target"),
    ])
    result = generate_discussion_graph(
        "The trial improved survival. Finally",
        model="gpt-5.6",
        client=client,
    )

    assert result["discussion_compiler"] == BALANCED_COMPILER_VERSION
    assert result["discussion_agent_pipeline"] == [
        "compiler", "evidence", "logic", "target", "assumption_conditional", "judge_conditional"
    ]
    assert [node["source_text"] for node in result["nodes"]] == ["The trial improved survival."]
    trace = result["architecture_traces"][0]
    assert trace["compiler_refinement"]["attempted"] is True
    assert trace["compiler_refinement"]["mode"] == "local_node_patch"
    formats = [call["text_format"].__name__ for call in client.responses.calls]
    assert formats.count("PilotCompilerOutput") == 1
    assert formats.count("DiscussionCompilerRepairOutput") == 1
    assert formats.count("PilotSpecialistReviewOutput") == 3


def test_v045_full_ragqa_bat_is_raw_vs_balanced_only():
    root = Path(__file__).resolve().parents[1]
    text = (root / "RUN_RAGQA_FULL_BALANCED_WINDOWS.bat").read_text(encoding="utf-8")
    assert "--limit 0" in text
    assert "--task-types QA" in text
    assert "run_ragtruth_raw_vs_dual_graph.py" in text
    assert "run_ragtruth_true_six_agent.py" not in text
    assert "No Source/Response six-agent validation" in text


def test_v045_discussion_hub_bat_forces_balanced_six_agent_architecture():
    root = Path(__file__).resolve().parents[1]
    text = (root / "RUN_DISCUSSION_HUB_BALANCED_6AGENTS_WINDOWS.bat").read_text(encoding="utf-8")
    assert "DISCUSSION_ARCHITECTURE=graph_native_6agents_balanced" in text
    assert "/discussion-lab" in text
