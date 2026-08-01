from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vrg.processbench_evaluation import (
    ChecklistFirstErrorOutput,
    DependencyGraphFirstErrorOutput,
    DependencyStepNode,
    DirectFirstErrorOutput,
    StepReview,
    _method_summary,
    _normalize_graph_nodes,
    load_processbench_cases,
    run_processbench_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "processbench_fixture"


def test_v034_loader_balances_split_and_correctness():
    rows, info = load_processbench_cases(
        FIXTURE,
        splits=["gsm8k", "math"],
        limit=4,
        seed=7,
    )
    assert len(rows) == 4
    assert sum(row["gold_has_error"] for row in rows) == 2
    assert info["by_split_and_label"] == {
        "gsm8k:correct": 1,
        "gsm8k:error": 1,
        "math:correct": 1,
        "math:error": 1,
    }


def test_v034_graph_normalization_rejects_future_dependencies():
    graph = _normalize_graph_nodes(
        [
            DependencyStepNode(step_index=0, depends_on=[1], uses_problem=True, verdict="correct"),
            DependencyStepNode(step_index=1, depends_on=[0, 2], uses_problem=False, verdict="incorrect", error_type="arithmetic"),
        ],
        2,
    )
    assert graph["nodes"][0]["depends_on"] == []
    assert graph["nodes"][1]["depends_on"] == [0]
    assert len(graph["invalid_dependencies"]) == 2
    assert graph["derived_first_error_step"] == 1


class _Responses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        user = " ".join(str(kwargs.get("input") or "").split())
        output_type = kwargs["text_format"]
        if "3 + 4" in user or "2x = 6" in user:
            prediction = 1
            step_count = 3
        else:
            prediction = -1
            step_count = 2
        if output_type is DirectFirstErrorOutput:
            parsed = DirectFirstErrorOutput(first_error_step=prediction, explanation="fixture")
        elif output_type is ChecklistFirstErrorOutput:
            reviews = [
                StepReview(step_index=i, verdict="incorrect" if i == prediction else "correct", explanation="fixture")
                for i in range(step_count)
            ]
            parsed = ChecklistFirstErrorOutput(step_reviews=reviews, first_error_step=prediction)
        elif output_type is DependencyGraphFirstErrorOutput:
            nodes = [
                DependencyStepNode(
                    step_index=i,
                    depends_on=[i - 1] if i > 0 else [],
                    uses_problem=i == 0,
                    verdict="incorrect" if i == prediction else "correct",
                    error_type="arithmetic" if i == prediction else "none",
                    explanation="fixture",
                )
                for i in range(step_count)
            ]
            parsed = DependencyGraphFirstErrorOutput(nodes=nodes, first_error_step=prediction)
        else:
            raise AssertionError(output_type)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(id=f"resp_{len(self.calls)}", model=kwargs["model"], output=[message], usage=usage)


class _Client:
    def __init__(self):
        self.responses = _Responses()


def test_v034_one_call_graph_official_metric_and_cache(tmp_path):
    client = _Client()
    cache = tmp_path / "cache.json"
    kwargs = dict(
        dataset_dir=FIXTURE,
        output_root=tmp_path / "outputs",
        splits=["gsm8k", "math"],
        small_model="gpt-5.4-nano",
        include_reference=False,
        include_checklist=True,
        limit=4,
        seed=7,
        generation_cache_path=cache,
        client=client,
    )
    first = run_processbench_evaluation(**kwargs)
    assert len(client.responses.calls) == 12  # 4 cases x direct/checklist/graph
    assert first["summary"]["actual_api_calls_this_run"] == 12
    assert first["method_summaries"]["small_dependency_graph"]["official_f1_percent"] == 100.0
    assert first["method_summaries"]["small_dependency_graph"]["mean_node_coverage_percent"] == 100.0

    second = run_processbench_evaluation(**kwargs)
    assert len(client.responses.calls) == 12
    assert second["summary"]["actual_api_calls_this_run"] == 0
    assert second["cache_summary"]["cache_hits_by_method"] == {
        "small_direct_step": 4,
        "small_checklist_step": 4,
        "small_dependency_graph": 4,
    }


def test_v034_official_f1_is_harmonic_error_and_correct_accuracy():
    rows = [
        {"split": "gsm8k", "methods": {"m": {"status": "ok", "scores": {"gold_has_error": True, "predicted_has_error": True, "exact_correct": True, "valid_prediction": True, "within_one_step": True, "step_distance": 0}, "usage": {}}}},
        {"split": "gsm8k", "methods": {"m": {"status": "ok", "scores": {"gold_has_error": True, "predicted_has_error": True, "exact_correct": False, "valid_prediction": True, "within_one_step": True, "step_distance": 1}, "usage": {}}}},
        {"split": "gsm8k", "methods": {"m": {"status": "ok", "scores": {"gold_has_error": False, "predicted_has_error": False, "exact_correct": True, "valid_prediction": True, "within_one_step": False, "step_distance": None}, "usage": {}}}},
    ]
    summary = _method_summary(rows, "m")
    # error accuracy=.5, correct accuracy=1.0 => harmonic mean=2/3
    assert summary["official_f1_percent"] == 66.67
