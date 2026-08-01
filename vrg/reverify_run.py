from __future__ import annotations

import csv
import json
import shutil
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .hybrid_runner import _analyze_transformed, _needs_repair
from .universal_graph import graph_diff
from .operational_batch import (
    _atomic_write_json,
    _compact_result,
    _safe_component,
    _utc_now,
    _write_summary_files,
    canonical_jsonl,
    dataset_fingerprint,
    parse_records_input,
)


def _load_case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_from_case(case: dict[str, Any], dataset_record: dict[str, Any] | None = None) -> dict[str, Any]:
    if dataset_record is not None:
        return dataset_record
    for attempt in case.get("attempts") or []:
        record = (((attempt.get("analysis") or {}).get("record_formalization") or {}).get("record"))
        if isinstance(record, dict):
            return record
    raise ValueError("Could not recover the original ProofWriter record from stored case JSON")


def _attempt_output(attempt: dict[str, Any]) -> dict[str, Any]:
    output = attempt.get("llm_output")
    if isinstance(output, dict):
        return deepcopy(output)
    raise ValueError("Stored attempt does not contain llm_output")


def reverify_case(
    legacy_case: dict[str, Any],
    record: dict[str, Any],
    *,
    prefer_z3: bool = True,
) -> dict[str, Any]:
    corrected_attempts: list[dict[str, Any]] = []
    prior_graph: dict[str, Any] | None = None
    for iteration, legacy_attempt in enumerate(legacy_case.get("attempts") or []):
        output = _attempt_output(legacy_attempt)
        analysis = _analyze_transformed(
            record,
            output,
            use_llm_formalizer=False,
            model=str((legacy_case.get("settings") or {}).get("model") or "stored-output"),
            reasoning_effort=str((legacy_case.get("settings") or {}).get("reasoning_effort") or "low"),
            prefer_z3=prefer_z3,
            client=None,
        )
        passed = not _needs_repair(analysis)
        attempt = {
            "iteration": iteration,
            "kind": "initial" if iteration == 0 else "stored_repair",
            "llm_output": output,
            "analysis": analysis,
            "passed": passed,
            "reverification_source": "stored_llm_output_no_api_call",
            "legacy_passed": bool(legacy_attempt.get("passed")),
            "legacy_verification_summary": deepcopy(((legacy_attempt.get("analysis") or {}).get("verified_graph") or {}).get("summary") or {}),
        }
        if prior_graph is not None:
            attempt["graph_diff_from_previous"] = graph_diff(prior_graph, analysis["universal_graph"])
        corrected_attempts.append(attempt)
        prior_graph = analysis["universal_graph"]

    if not corrected_attempts:
        raise ValueError("Stored case has no attempts")
    # Corrected verifier policy: never retain a repair that was triggered only by the old parser.
    # If the initial output now passes, it is the final output. Otherwise take the first stored
    # repair that passes; if none passes, retain the last stored attempt for inspection.
    if corrected_attempts[0]["passed"]:
        selected_index = 0
        selection_reason = "initial_output_passes_corrected_verifier"
    else:
        passing_repairs = [i for i, x in enumerate(corrected_attempts[1:], 1) if x["passed"]]
        if passing_repairs:
            selected_index = passing_repairs[0]
            selection_reason = "first_stored_repair_passing_corrected_verifier"
        else:
            selected_index = len(corrected_attempts) - 1
            selection_reason = "no_stored_attempt_passed_corrected_verifier"

    selected = corrected_attempts[selected_index]
    final_graph = deepcopy(selected["analysis"]["universal_graph"])
    if selected_index > 0:
        diff = graph_diff(corrected_attempts[0]["analysis"]["universal_graph"], final_graph)
        changed = {x["node_id"] for x in diff.get("changed_nodes") or []}
        added = set(diff.get("added_nodes") or [])
        for node in final_graph.get("nodes") or []:
            nid = str(node.get("id"))
            node["repair_status"] = "added" if nid in added else ("modified" if nid in changed else "unchanged")
        final_graph["repair_history"] = diff
    else:
        for node in final_graph.get("nodes") or []:
            node["repair_status"] = "original"

    initial = corrected_attempts[0]
    legacy_summary = legacy_case.get("summary") or {}
    usage = deepcopy(legacy_summary.get("total_usage") or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    summary = {
        "attempt_count": len(corrected_attempts),
        "stored_repair_attempt_count": max(0, len(corrected_attempts) - 1),
        "selected_attempt_index": selected_index,
        "effective_repair_count": selected_index,
        "repair_count": selected_index,
        "initial_pass": bool(initial["passed"]),
        "final_pass": bool(selected["passed"]),
        "initial_answer": initial["analysis"]["classification"]["predicted_label"],
        "final_answer": selected["analysis"]["classification"]["predicted_label"],
        "context_label": selected["analysis"]["classification"]["label"],
        "gold_label": selected["analysis"]["classification"]["gold_label"],
        "final_answer_correct": selected["analysis"]["classification"]["answer_correct"],
        "final_context_match": selected["analysis"]["classification"]["prediction_matches_context"],
        "selection_reason": selection_reason,
        "total_usage": usage,
        "new_api_calls": 0,
        "legacy_initial_pass": legacy_summary.get("initial_pass"),
        "legacy_final_pass": legacy_summary.get("final_pass"),
        "legacy_final_answer": legacy_summary.get("final_answer"),
    }
    return {
        "schema_version": "0.19.0",
        "record_id": str(record.get("id") or legacy_case.get("record_id") or "proofwriter_case"),
        "architecture": "Offline v019 re-verification using ProofWriter context/raw-logic cross-validation; stored LLM outputs reused",
        "settings": {
            **deepcopy(legacy_case.get("settings") or {}),
            "reverified_offline": True,
            "new_api_calls": 0,
            "prefer_z3": prefer_z3,
        },
        "record": deepcopy(record),
        "initial_generation": deepcopy(legacy_case.get("initial_generation") or {}),
        "attempts": corrected_attempts,
        "summary": summary,
        "final_universal_graph": final_graph,
        "legacy_v018": {
            "schema_version": legacy_case.get("schema_version"),
            "summary": deepcopy(legacy_summary),
        },
        "reverification": {
            "performed_at": _utc_now(),
            "new_api_calls": 0,
            "selection_policy": "initial-if-corrected-pass-else-first-passing-stored-repair-else-last",
        },
    }


def reverify_run_directory(
    source_run_dir: Path,
    destination_runs_root: Path | None = None,
    *,
    new_run_id: str | None = None,
    prefer_z3: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    if not source_run_dir.exists():
        raise ValueError(f"Run directory does not exist: {source_run_dir}")
    case_files = sorted((source_run_dir / "cases").glob("*.json"))
    if not case_files:
        raise ValueError("Run directory contains no cases/*.json files")
    dataset_path = source_run_dir / "dataset.jsonl"
    dataset_records = parse_records_input(dataset_path.read_text(encoding="utf-8-sig")) if dataset_path.exists() else []
    by_id = {str(x.get("id")): x for x in dataset_records}

    destination_runs_root = Path(destination_runs_root or source_run_dir.parent)
    destination_runs_root.mkdir(parents=True, exist_ok=True)
    run_id = _safe_component(new_run_id or f"{source_run_dir.name}_v019_reverified", "v019_reverified")
    dest = destination_runs_root / run_id
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "cases").mkdir(parents=True)
    (dest / "errors").mkdir(parents=True)

    records_out: list[dict[str, Any]] = []
    rows: dict[int, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    total = len(case_files)
    for ordinal, case_path in enumerate(case_files, 1):
        started = time.perf_counter()
        legacy = _load_case(case_path)
        rid = str(legacy.get("record_id") or "")
        record = _record_from_case(legacy, by_id.get(rid))
        records_out.append(record)
        try:
            corrected = reverify_case(legacy, record, prefer_z3=prefer_z3)
            filename = f"{ordinal:06d}_{_safe_component(record.get('id'), f'record_{ordinal}')}.json"
            _atomic_write_json(dest / "cases" / filename, corrected)
            row = _compact_result(ordinal, record, corrected, f"cases/{filename}", (time.perf_counter() - started) * 1000)
            row["api_calls"] = 0
            row["new_api_calls"] = 0
            row["selected_attempt_index"] = corrected["summary"]["selected_attempt_index"]
            row["selection_reason"] = corrected["summary"]["selection_reason"]
            rows[ordinal] = row
        except Exception as exc:
            error = {"index": ordinal, "record_id": record.get("id"), "error_type": type(exc).__name__, "error": str(exc)}
            errors.append(error)
            _atomic_write_json(dest / "errors" / f"{ordinal:06d}_{_safe_component(record.get('id'))}.json", error)
        if progress:
            progress(ordinal, total, str(record.get("id") or ""))

    (dest / "dataset.jsonl").write_text(canonical_jsonl(records_out), encoding="utf-8")
    fingerprint = dataset_fingerprint(records_out)
    state = {
        "schema_version": "0.19.0",
        "run_id": run_id,
        "status": "completed_with_errors" if errors else "completed",
        "total_records": len(records_out),
        "dataset_fingerprint": fingerprint,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "active_mode": "offline_reverify",
        "selected_records": len(records_out),
        "message": "v019 offline re-verification completed",
        "source_run_id": source_run_dir.name,
        "new_api_calls": 0,
        "phase_history": [{"mode": "offline_reverify", "status": "completed", "completed_at": _utc_now(), "selected_records": len(records_out)}],
    }
    _atomic_write_json(dest / "state.json", state)
    _atomic_write_json(dest / "settings.json", {
        "mode": "offline_reverify",
        "source_run_id": source_run_dir.name,
        "prefer_z3": prefer_z3,
        "new_api_calls": 0,
        "canonicalization": "context/raw_logic_cross_validation",
    })
    # Use the operational writer, then extend CSV with v019 selection fields.
    _write_summary_files(dest, state, rows)
    with (dest / "index.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for index in sorted(rows):
            handle.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
    if errors:
        _atomic_write_json(dest / "reverification_errors.json", errors)
    _atomic_write_json(dest / "reverification_manifest.json", {
        "schema_version": "0.19.0",
        "source_run": str(source_run_dir),
        "destination_run": str(dest),
        "records": len(records_out),
        "completed": len(rows),
        "errors": len(errors),
        "new_api_calls": 0,
        "created_at": _utc_now(),
    })
    return dest
