from __future__ import annotations

from pathlib import Path

from test_v040_ragtruth_dual_graph import _Client
from vrg.ragtruth_dual_graph import run_ragtruth_raw_vs_dual_graph


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "ragtruth_fixture"


def test_v041_prints_per_case_raw_vs_dual_graph_scores(tmp_path):
    messages: list[str] = []
    result = run_ragtruth_raw_vs_dual_graph(
        response_path=FIXTURE / "response.jsonl",
        source_path=FIXTURE / "source_info.jsonl",
        output_root=tmp_path / "outputs",
        model="gpt-5.4-nano",
        task_types=["QA", "Summary"],
        limit=2,
        seed=1,
        require_full_evidence=True,
        generation_cache_path=tmp_path / "cache.json",
        client=_Client(),
        progress=messages.append,
        print_case_comparison=True,
    )

    text = "\n".join(messages)
    assert "CASE PERFORMANCE" in text
    assert "Gold" in text
    assert "Raw" in text
    assert "DualGraph" in text
    assert "Case delta DualGraph-Raw" in text
    assert "Running micro-F1 after 1 case(s)" in text
    assert "Running micro-F1 after 2 case(s)" in text
    assert result["settings"]["print_case_comparison"] is True


def test_v041_can_disable_per_case_console_output(tmp_path):
    messages: list[str] = []
    result = run_ragtruth_raw_vs_dual_graph(
        response_path=FIXTURE / "response.jsonl",
        source_path=FIXTURE / "source_info.jsonl",
        output_root=tmp_path / "outputs",
        model="gpt-5.4-nano",
        task_types=["QA", "Summary"],
        limit=1,
        seed=1,
        require_full_evidence=True,
        generation_cache_path=tmp_path / "cache.json",
        client=_Client(),
        progress=messages.append,
        print_case_comparison=False,
    )

    assert "CASE PERFORMANCE" not in "\n".join(messages)
    assert result["settings"]["print_case_comparison"] is False
