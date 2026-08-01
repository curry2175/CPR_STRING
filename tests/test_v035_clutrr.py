from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vrg.clutrr_evaluation import (
    GraphStructureOutput,
    KinshipEdge,
    RelationAnswerOutput,
    RelationReplayOutput,
    TextStructureOutput,
    _edge_metrics,
    _shuffle_graph,
    load_clutrr_cases,
    run_clutrr_evaluation,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "clutrr_fixture"


def test_v035_loader_stratifies_by_hop():
    rows, info = load_clutrr_cases(
        FIXTURE,
        config="gen_train23_test2to10",
        split="test",
        hop_lengths=[2, 3],
        per_hop=1,
        seed=7,
    )
    assert len(rows) == 2
    assert info["by_hop"] == {"2": 1, "3": 1}
    assert all(row["gold_graph"]["edges"] for row in rows)


def test_v035_edge_metrics_and_shuffle():
    graph = {
        "nodes": ["A", "B", "C"],
        "edges": [
            {"source": "A", "relation": "mother", "target": "B"},
            {"source": "B", "relation": "father", "target": "C"},
        ],
        "query_path_nodes": ["A", "B", "C"],
        "query_path_relations": ["mother", "father"],
    }
    metrics = _edge_metrics(graph["edges"], graph["edges"])
    assert metrics["edge_f1"] == 1.0
    shuffled = _shuffle_graph(graph, 2035, "case")
    assert shuffled["shuffle_changed_edge_count"] > 0
    assert shuffled["edges"] != graph["edges"]


class _Responses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        output_type = kwargs["text_format"]
        user = " ".join(str(kwargs.get("input") or "").split())
        if "Clarence" in user:
            answer = "grandfather"
            nodes = ["Heidi", "Wayne", "Clarence"]
            edges = [
                KinshipEdge(source="Heidi", relation="father", target="Wayne"),
                KinshipEdge(source="Wayne", relation="father", target="Clarence"),
            ]
        elif "Dora" in user:
            answer = "aunt"
            nodes = ["Amy", "Beth", "Cara", "Dora"]
            edges = [
                KinshipEdge(source="Amy", relation="mother", target="Beth"),
                KinshipEdge(source="Beth", relation="mother", target="Cara"),
                KinshipEdge(source="Cara", relation="daughter", target="Dora"),
            ]
        elif "Sam" in user:
            answer = "uncle"
            nodes = ["Nora", "Paul", "Rita", "Sam"]
            edges = [
                KinshipEdge(source="Nora", relation="father", target="Paul"),
                KinshipEdge(source="Paul", relation="mother", target="Rita"),
                KinshipEdge(source="Rita", relation="son", target="Sam"),
            ]
        else:
            answer = "grandmother"
            nodes = ["April", "Lillian", "Ashley"]
            edges = [
                KinshipEdge(source="April", relation="mother", target="Lillian"),
                KinshipEdge(source="Lillian", relation="mother", target="Ashley"),
            ]
        if output_type is RelationAnswerOutput:
            parsed = RelationAnswerOutput(answer_relation=answer, explanation="fixture")
        elif output_type is RelationReplayOutput:
            parsed = RelationReplayOutput(answer_relation=answer)
        elif output_type is TextStructureOutput:
            parsed = TextStructureOutput(
                direct_fact_notes=[f"fact {i}" for i in range(len(edges))],
                composition_plan=["combine the facts in order"],
                answer_relation=answer,
            )
        elif output_type is GraphStructureOutput:
            parsed = GraphStructureOutput(
                nodes=nodes,
                edges=edges,
                query_path_nodes=nodes,
                query_path_relations=[edge.relation for edge in edges],
                answer_relation=answer,
            )
        else:
            raise AssertionError(output_type)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(id=f"resp_{len(self.calls)}", model=kwargs["model"], output=[message], usage=usage)


class _Client:
    def __init__(self):
        self.responses = _Responses()


def test_v035_pipeline_cache_and_representation_metrics(tmp_path):
    client = _Client()
    cache = tmp_path / "cache.json"
    kwargs = dict(
        dataset_dir=FIXTURE,
        output_root=tmp_path / "outputs",
        config="gen_train23_test2to10",
        split="test",
        hop_lengths=[2, 3],
        per_hop=1,
        seed=7,
        small_model="gpt-5.4-nano",
        include_reference=False,
        include_replay_ablation=True,
        generation_cache_path=cache,
        client=client,
    )
    first = run_clutrr_evaluation(**kwargs)
    assert len(client.responses.calls) == 12  # 2 cases x 6 nano conditions
    assert first["summary"]["actual_api_calls_this_run"] == 12
    assert first["method_summaries"]["small_explicit_graph"]["accuracy_percent"] == 100.0
    assert first["method_summaries"]["small_explicit_graph"]["mean_edge_f1_percent"] == 100.0

    second = run_clutrr_evaluation(**kwargs)
    assert len(client.responses.calls) == 12
    assert second["summary"]["actual_api_calls_this_run"] == 0
    assert second["cache_summary"]["cache_hits_by_method"] == {
        "small_direct_relation": 2,
        "small_text_structure": 2,
        "small_explicit_graph": 2,
        "small_text_replay": 2,
        "small_graph_replay": 2,
        "small_shuffled_graph_replay": 2,
    }


def test_v035_structured_retry_uses_second_budget():
    from vrg.clutrr_evaluation import _call_parsed

    class RetryResponses:
        def __init__(self):
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise ValueError("Model returned no parsed structured output")
            parsed = RelationReplayOutput(answer_relation="uncle")
            item = SimpleNamespace(type="output_text", parsed=parsed)
            message = SimpleNamespace(type="message", content=[item])
            usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
            return SimpleNamespace(id="resp_retry", model=kwargs["model"], output=[message], usage=usage)

    client = SimpleNamespace(responses=RetryResponses())
    parsed, meta = _call_parsed(
        client,
        model="gpt-5.4-nano",
        reasoning_effort="low",
        max_output_tokens=1000,
        system="system",
        user="user",
        output_type=RelationReplayOutput,
    )
    assert parsed.answer_relation == "uncle"
    assert meta["api_calls"] == 2
    assert meta["retry_count"] == 1
    assert client.responses.calls[0]["max_output_tokens"] == 1000
    assert client.responses.calls[1]["max_output_tokens"] == 2600


def test_v035_method_failure_is_isolated(tmp_path):
    class FailingGraphResponses(_Responses):
        def parse(self, **kwargs):
            if kwargs["text_format"] is GraphStructureOutput:
                self.calls.append(kwargs)
                raise ValueError("Model returned no parsed structured output")
            return super().parse(**kwargs)

    client = SimpleNamespace(responses=FailingGraphResponses())
    result = run_clutrr_evaluation(
        dataset_dir=FIXTURE,
        output_root=tmp_path / "outputs",
        config="gen_train23_test2to10",
        split="test",
        hop_lengths=[2],
        per_hop=1,
        seed=7,
        small_model="gpt-5.4-nano",
        reference_model="gpt-5.4-mini",
        include_reference=True,
        include_replay_ablation=True,
        generation_cache_path=tmp_path / "cache.json",
        client=client,
    )
    methods = result["cases"][0]["methods"]
    assert methods["small_explicit_graph"]["status"] == "failed"
    assert methods["small_graph_replay"]["status"] == "skipped"
    assert methods["small_shuffled_graph_replay"]["status"] == "skipped"
    assert methods["small_text_replay"]["status"] == "ok"
    assert methods["reference_direct_relation"]["status"] == "ok"
    assert result["summary"]["failed_method_calls"] == 1
