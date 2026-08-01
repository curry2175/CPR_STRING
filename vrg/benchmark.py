from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any
import json

from .verifier import _status_signature, verify_case, verify_case_incremental


@dataclass
class BenchmarkOptions:
    prefer_z3: bool = True
    repetitions: int = 5


def _find_step(case: dict[str, Any], node_id: str) -> dict[str, Any]:
    output = case.get("llm_output") or {}
    for step in output.get("reasoning_steps") or case.get("reasoning_steps") or []:
        if isinstance(step, dict) and str(step.get("id")) == node_id:
            return step
    raise ValueError(f"Reasoning node not found: {node_id}")


def apply_edit(case: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(case)
    node_id = str(edit.get("node_id") or "")
    if node_id == "final":
        updated.setdefault("llm_output", {})["answer"] = str(edit["new_value"])
    elif node_id.startswith("s"):
        _find_step(updated, node_id)["text"] = str(edit["new_value"])
    else:
        raise ValueError("Built-in benchmark supports reasoning or Final edits")
    return updated


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_incremental_scenarios(
    scenarios: list[dict[str, Any]],
    options: BenchmarkOptions,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    total_started = perf_counter()
    repetitions = max(1, min(int(options.repetitions), 30))
    for scenario in scenarios:
        case = deepcopy(scenario["case"])
        updated = apply_edit(case, scenario["edit"])
        previous_result = verify_case(case, prefer_z3=options.prefer_z3, compute_counterfactuals=False)
        incremental_times: list[float] = []
        full_times: list[float] = []
        parity_results: list[bool] = []
        last_incremental: dict[str, Any] | None = None
        for _ in range(repetitions):
            inc_started = perf_counter()
            inc = verify_case_incremental(
                case, updated, previous_result,
                prefer_z3=options.prefer_z3,
                validate_against_full=False,
            )
            incremental_times.append((perf_counter() - inc_started) * 1000)
            full_started = perf_counter()
            full = verify_case(updated, prefer_z3=options.prefer_z3, compute_counterfactuals=False)
            full_times.append((perf_counter() - full_started) * 1000)
            parity_results.append(_status_signature(inc) == _status_signature(full))
            last_incremental = inc
        assert last_incremental is not None
        inc_mean = _mean(incremental_times)
        full_mean = _mean(full_times)
        speedup = full_mean / inc_mean if inc_mean > 0 else None
        reduction = (full_mean - inc_mean) / full_mean * 100 if full_mean > 0 else None
        meta = last_incremental.get("incremental") or {}
        rows.append({
            "scenario_id": str(scenario.get("id") or "scenario"),
            "description": str(scenario.get("description") or ""),
            "changed_node_id": str(scenario["edit"].get("node_id")),
            "mode": meta.get("mode"),
            "parity_all_repetitions": all(parity_results),
            "repetitions": repetitions,
            "reused_reasoning_count": meta.get("reused_reasoning_count"),
            "revalidated_reasoning_count": meta.get("revalidated_reasoning_count"),
            "final_reused": meta.get("final_reused"),
            "candidate_suffix_count": meta.get("candidate_suffix_count"),
            "graph_local_count": meta.get("graph_local_count"),
            "scope_reduction_percent": meta.get("scope_reduction_percent"),
            "incremental_mean_ms": round(inc_mean, 3),
            "incremental_median_ms": round(median(incremental_times), 3),
            "full_mean_ms": round(full_mean, 3),
            "full_median_ms": round(median(full_times), 3),
            "speedup_ratio": round(speedup, 3) if speedup is not None else None,
            "runtime_reduction_percent": round(reduction, 2) if reduction is not None else None,
            "solver_stats": meta.get("solver_stats") or {},
            "final_proof_status": last_incremental.get("summary", {}).get("final_proof_status"),
            "final_chain_status": last_incremental.get("summary", {}).get("final_chain_status"),
        })
    total_ms = (perf_counter() - total_started) * 1000
    valid_speedups = [row["speedup_ratio"] for row in rows if row.get("speedup_ratio") is not None]
    return {
        "schema_version": "0.15.0",
        "summary": {
            "scenario_count": len(rows),
            "repetitions_per_scenario": repetitions,
            "parity_pass_count": sum(bool(row["parity_all_repetitions"]) for row in rows),
            "graph_local_scenario_count": sum(row.get("mode") == "graph_local_incremental" for row in rows),
            "mean_speedup_ratio": round(_mean(valid_speedups), 3) if valid_speedups else None,
            "mean_scope_reduction_percent": round(_mean([float(row.get("scope_reduction_percent") or 0) for row in rows]), 2),
            "total_benchmark_runtime_ms": round(total_ms, 3),
            "engine": rows[0].get("solver_stats", {}).get("strategy") if rows else None,
        },
        "scenarios": rows,
    }


def load_builtin_scenarios(data_dir: Path) -> list[dict[str, Any]]:
    def load(name: str) -> dict[str, Any]:
        return json.loads((data_dir / name).read_text(encoding="utf-8"))
    return [
        {
            "id": "long_chain_middle_edit",
            "description": "Linear 11-step chain; edit s8 so graph-local equals the remaining suffix.",
            "case": load("sample_long_chain_incremental.json"),
            "edit": {"node_id": "s8", "new_value": "Bob is not confident."},
        },
        {
            "id": "branched_unrelated_edit",
            "description": "Edit the non-answer branch; only s2-s4 should be revalidated and Final can be reused.",
            "case": load("sample_branched_graph_local.json"),
            "edit": {"node_id": "s2", "new_value": "Bob is not focused."},
        },
        {
            "id": "branched_answer_branch_edit",
            "description": "Edit the branch that supports Final; s6-s8 and Final should be revalidated.",
            "case": load("sample_branched_graph_local.json"),
            "edit": {"node_id": "s6", "new_value": "Bob is not helpful."},
        },
        {
            "id": "final_only_edit",
            "description": "Change Yes to No; reuse all reasoning and revalidate only Final.",
            "case": load("sample_valid.json"),
            "edit": {"node_id": "final", "new_value": "No"},
        },
    ]
