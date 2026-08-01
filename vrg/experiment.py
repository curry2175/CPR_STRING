from __future__ import annotations

import csv
import io
import json
import os
import random
import shutil
import time
import zipfile
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .logic import Atom, is_variable
from .engine import ChainSupportEngine, KnowledgeItem
from .mutation import atom_to_controlled_english, _lock_baseline_chain
from .parser import parse_statement
from .verifier import verify_case, verify_case_incremental
from .hybrid_runner import (
    _analyze_transformed,
    _needs_repair,
    _repair_call,
    build_repair_packet,
)

SCHEMA_VERSION = "0.23.0"
FAULT_TYPES = (
    "polarity_flip",
    "entity_swap",
    "argument_swap",
    "predicate_swap",
    "parent_deletion",
    "wrong_parent",
    "step_deletion",
    "answer_flip",
)
DIFFICULTIES = ("local", "upstream")
STRUCTURAL_FAULTS = {"step_deletion"}
EDGE_FAULTS = {"parent_deletion", "wrong_parent", "step_deletion"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _truth(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes"}


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _case_files(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "cases").glob("*.json"))


def _load_case(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_adapted_case(stored: dict[str, Any]) -> dict[str, Any]:
    attempts = stored.get("attempts") or []
    if not attempts:
        raise ValueError("Stored case has no attempts")
    case = deepcopy(attempts[0]["analysis"]["adapted_case"])
    case["verification_policy"] = {
        "require_declared_reasoning_dependencies": True,
        "require_declared_answer_dependencies": True,
    }
    return case


def _initial_output(stored: dict[str, Any]) -> dict[str, Any]:
    attempts = stored.get("attempts") or []
    if not attempts:
        raise ValueError("Stored case has no attempts")
    return deepcopy(attempts[0]["llm_output"])


def _step_parts(item: Any, index: int) -> tuple[str, str, list[str]]:
    if isinstance(item, str):
        return f"s{index}", item, []
    return str(item.get("id") or f"s{index}"), str(item.get("text") or ""), list(item.get("depends_on") or [])


def _steps(case: dict[str, Any]) -> list[Any]:
    return list((case.get("llm_output") or {}).get("reasoning_steps") or [])


def _set_steps(case: dict[str, Any], steps: list[Any]) -> dict[str, Any]:
    out = deepcopy(case)
    out.setdefault("llm_output", {})["reasoning_steps"] = deepcopy(steps)
    return out


def _replace_step(
    case: dict[str, Any],
    node_id: str,
    *,
    text: str | None = None,
    depends_on: list[str] | None = None,
) -> dict[str, Any]:
    out = deepcopy(case)
    new_steps = []
    found = False
    for i, item in enumerate(_steps(out), 1):
        sid, old_text, old_deps = _step_parts(item, i)
        payload = {"id": sid, "text": old_text, "depends_on": old_deps}
        if isinstance(item, dict):
            payload = {**item, **payload}
        if sid == node_id:
            found = True
            if text is not None:
                payload["text"] = text
            if depends_on is not None:
                payload["depends_on"] = list(depends_on)
        new_steps.append(payload)
    if not found:
        raise ValueError(f"Reasoning node not found: {node_id}")
    out.setdefault("llm_output", {})["reasoning_steps"] = new_steps
    return out


def _error_nodes(result: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    for node in result.get("nodes") or []:
        if node.get("kind") != "reasoning" and str(node.get("id")) != "final":
            continue
        if node.get("proof_status") != "valid" or node.get("chain_status") != "valid":
            bad.append(str(node.get("id")))
    return sorted(set(bad))


def _predicted_roots(result: dict[str, Any]) -> list[str]:
    roots = [str(x) for x in (result.get("summary") or {}).get("root_error_nodes") or []]
    final = next((n for n in result.get("nodes") or [] if str(n.get("id")) == "final"), None)
    if final and (final.get("proof_status") != "valid" or final.get("chain_status") not in {"valid", "given"}):
        if not final.get("blocking_parent_nodes"):
            roots.append("final")
    return sorted(set(roots))


def _overall_invalid(result: dict[str, Any]) -> bool:
    s = result.get("summary") or {}
    return bool(
        s.get("final_proof_status") != "valid"
        or s.get("final_chain_status") != "valid"
        or int(s.get("invalid_reasoning_count") or 0) > 0
        or int(s.get("chain_error_count") or 0) > 0
    )


def _candidate_nodes(locked_case: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    lookup = {str(n.get("id")): n for n in baseline.get("nodes") or []}
    rows = []
    for i, item in enumerate(_steps(locked_case), 1):
        sid, text, deps = _step_parts(item, i)
        node = lookup.get(sid, {})
        parsed = parse_statement(text)
        if node.get("proof_status") != "valid" or node.get("chain_status") != "valid" or not isinstance(parsed.formula, Atom):
            continue
        if any(is_variable(a) for a in parsed.formula.args):
            continue
        rows.append({"id": sid, "text": text, "deps": deps, "atom": parsed.formula, "node": node, "index": i - 1})
    return rows


def _difficulty_targets(candidates: list[dict[str, Any]], requested: Iterable[str]) -> list[tuple[str, dict[str, Any]]]:
    reaching = [x for x in candidates if x["node"].get("chain_reaches_final")]
    pool = reaching or candidates
    if not pool:
        return []
    ordered = sorted(pool, key=lambda x: x["index"])
    if len(ordered) == 1:
        return [("single_step", ordered[0])]
    result: list[tuple[str, dict[str, Any]]] = []
    if "upstream" in requested:
        result.append(("upstream", ordered[0]))
    if "local" in requested:
        local = ordered[-1]
        if not result or result[-1][1]["id"] != local["id"]:
            result.append(("local", local))
    return result


def _known_predicates(case: dict[str, Any]) -> list[str]:
    preds: set[str] = set()
    for premise in case.get("premises") or []:
        parsed = parse_statement(str(premise.get("text") or premise))
        formula = parsed.formula
        if isinstance(formula, Atom):
            preds.add(formula.predicate)
        elif hasattr(formula, "conclusion") and isinstance(formula.conclusion, Atom):
            preds.add(formula.conclusion.predicate)
    return sorted(preds)


def _apply_fault(
    locked: dict[str, Any],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    fault_type: str,
    ordinal: int,
    attempt: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    sid, atom, deps = candidate["id"], candidate["atom"], list(candidate["deps"])
    expected_node = sid
    meta: dict[str, Any] = {
        "mutated_node_id": sid,
        "original_text": candidate["text"],
        "original_depends_on": deps,
        "resample_attempt": attempt,
    }
    if fault_type == "polarity_flip":
        mutated = atom.complement()
        out = _replace_step(locked, sid, text=atom_to_controlled_english(mutated))
        meta["mutated_text"] = atom_to_controlled_english(mutated)
    elif fault_type == "entity_swap":
        args = list(atom.args)
        args[0] = f"fault_entity_{ordinal}_{attempt}"
        mutated = Atom(atom.predicate, tuple(args), atom.negated)
        out = _replace_step(locked, sid, text=atom_to_controlled_english(mutated))
        meta["mutated_text"] = atom_to_controlled_english(mutated)
    elif fault_type == "argument_swap":
        if len(atom.args) != 2 or atom.args[0] == atom.args[1]:
            return None
        mutated = Atom(atom.predicate, (atom.args[1], atom.args[0]), atom.negated)
        out = _replace_step(locked, sid, text=atom_to_controlled_english(mutated))
        meta["mutated_text"] = atom_to_controlled_english(mutated)
    elif fault_type == "predicate_swap":
        known = [p for p in _known_predicates(locked) if p != atom.predicate]
        if attempt < len(known):
            replacement = known[(ordinal + attempt) % len(known)]
            replacement_source = "known_predicate"
        else:
            replacement = f"fault_predicate_{ordinal}_{attempt}"
            replacement_source = "novel_fallback"
        mutated = Atom(replacement, atom.args, atom.negated)
        out = _replace_step(locked, sid, text=atom_to_controlled_english(mutated))
        meta.update({"mutated_text": atom_to_controlled_english(mutated), "replacement_predicate": replacement, "replacement_source": replacement_source})
    elif fault_type == "parent_deletion":
        if not deps:
            return None
        if len(deps) == 1:
            remaining: list[str] = []
        else:
            remove_index = attempt % len(deps)
            remaining = [dep for i, dep in enumerate(deps) if i != remove_index]
        out = _replace_step(locked, sid, depends_on=remaining)
        meta["mutated_depends_on"] = remaining
    elif fault_type == "wrong_parent":
        premise_ids = [str(n.get("id")) for n in baseline.get("nodes") or [] if n.get("kind") == "premise"]
        alternatives = [p for p in premise_ids if p not in deps]
        if not alternatives:
            return None
        wrong = alternatives[(ordinal + attempt) % len(alternatives)]
        out = _replace_step(locked, sid, depends_on=[wrong])
        meta["mutated_depends_on"] = [wrong]
    elif fault_type == "step_deletion":
        steps = []
        for i, item in enumerate(_steps(locked), 1):
            x, _, _ = _step_parts(item, i)
            if x != sid:
                steps.append(item)
        out = _set_steps(locked, steps)
        meta["deleted_node_id"] = sid
        expected_node = sid
    elif fault_type == "answer_flip":
        out = deepcopy(locked)
        current = str((out.get("llm_output") or {}).get("answer") or "Yes")
        flipped = "No" if current == "Yes" else "Yes"
        out.setdefault("llm_output", {})["answer"] = flipped
        expected_node = "final"
        meta.update({"original_answer": current, "mutated_answer": flipped, "mutated_node_id": "final"})
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")
    meta["expected_error_node_id"] = expected_node
    return out, meta


def _formula_map_and_order(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], dict[str, str]]:
    formulas: dict[str, Any] = {}
    orders: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for i, premise in enumerate(case.get("premises") or []):
        pid = str(premise.get("id") or f"p{i+1}") if isinstance(premise, dict) else f"p{i+1}"
        text = str(premise.get("text") or "") if isinstance(premise, dict) else str(premise)
        parsed = parse_statement(text)
        if parsed.formula is not None:
            formulas[pid] = parsed.formula
            orders[pid] = i
            kinds[pid] = "premise"
    offset = len(orders)
    for i, item in enumerate(_steps(case), 1):
        sid, text, _ = _step_parts(item, i)
        parsed = parse_statement(text)
        if parsed.formula is not None:
            formulas[sid] = parsed.formula
            orders[sid] = offset + i
            kinds[sid] = "reasoning"
    return formulas, orders, kinds


def _oracle_mutation_valid(
    locked: dict[str, Any],
    baseline: dict[str, Any],
    mutated: dict[str, Any],
    meta: dict[str, Any],
    fault_type: str,
) -> tuple[bool, str]:
    """Independent mutation oracle based on authored local support.

    This gate does not inspect the verifier result being evaluated. It uses a
    separate Horn support calculation over the model-visible premises and prior
    baseline reasoning formulas.
    """
    expected = str(meta.get("expected_error_node_id"))
    if fault_type == "answer_flip":
        return True, "binary final answer flipped from a clean valid baseline"
    if fault_type == "step_deletion":
        deleted = str(meta.get("deleted_node_id") or expected)
        referenced = False
        for i, item in enumerate(_steps(locked), 1):
            sid, _, deps = _step_parts(item, i)
            if sid != deleted and deleted in deps:
                referenced = True
                break
        if deleted in list((locked.get("llm_output") or {}).get("answer_depends_on") or []):
            referenced = True
        return referenced, "deleted step is referenced downstream" if referenced else "deleted step was not on the authored route"

    formulas, orders, kinds = _formula_map_and_order(mutated)
    target = formulas.get(expected)
    if target is None:
        return False, "mutated target could not be independently parsed"
    target_order = orders.get(expected, 10**9)
    prior_items = [
        KnowledgeItem(node_id, formula)
        for node_id, formula in formulas.items()
        if orders.get(node_id, 10**9) < target_order
    ]
    engine = ChainSupportEngine(kinds, orders)
    global_support = engine.support(prior_items, target).found
    step = next((_step_parts(item, i) for i, item in enumerate(_steps(mutated), 1) if _step_parts(item, i)[0] == expected), None)
    declared = step[2] if step else []
    local_items = [KnowledgeItem(dep, formulas[dep]) for dep in declared if dep in formulas]
    local_support = engine.support(local_items, target).found if local_items else False
    is_fault = not (global_support and local_support)
    return is_fault, f"independent_horn_global={global_support};independent_horn_local={local_support};declared={declared}"


def _dependency_closure(baseline: dict[str, Any], root_id: str) -> set[str]:
    dependents: dict[str, set[str]] = defaultdict(set)
    for node in baseline.get("nodes") or []:
        node_id = str(node.get("id"))
        deps = list(node.get("declared_reasoning_dependencies") or [])
        if node_id == "final" and not deps:
            deps = list(node.get("reasoning_dependencies") or [])
        for dep in deps:
            dependents[str(dep)].add(node_id)
    closure = {root_id}
    frontier = [root_id]
    while frontier:
        current = frontier.pop()
        for child in dependents.get(current, set()):
            if child not in closure:
                closure.add(child)
                frontier.append(child)
    return closure


def _set_metrics(predicted: set[str], expected: set[str]) -> tuple[int, int, int]:
    return len(predicted & expected), len(predicted - expected), len(expected - predicted)


def _aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    result = []
    for name, items in sorted(groups.items()):
        n = len(items)
        detected = sum(_truth(x.get("detected")) for x in items)
        localized = sum(_truth(x.get("root_exact_localized")) for x in items)
        result.append({
            key: name,
            "n": n,
            "detected": detected,
            "detection_rate": round(detected / n * 100, 2) if n else None,
            "root_exact_localized": localized,
            "root_exact_localization_rate": round(localized / n * 100, 2) if n else None,
            "false_acceptance": n - detected,
            "mean_runtime_ms": round(sum(float(x.get("runtime_ms") or 0) for x in items) / n, 3) if n else None,
        })
    return result


def run_fault_injection_experiment(
    source_run_dir: Path,
    output_root: Path,
    *,
    sample_count: int = 100,
    seed: int = 2026,
    fault_types: Iterable[str] = FAULT_TYPES,
    difficulties: Iterable[str] = DIFFICULTIES,
    prefer_z3: bool = True,
    max_reasoning_steps: int = 8,
    max_resample_attempts: int = 12,
) -> dict[str, Any]:
    fault_types = tuple(x for x in fault_types if x in FAULT_TYPES)
    difficulties = tuple(x for x in difficulties if x in DIFFICULTIES)
    if not fault_types or not difficulties:
        raise ValueError("Select at least one fault type and difficulty")
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _case_files(source_run_dir):
        stored = _load_case(path)
        attempt = (stored.get("attempts") or [{}])[0]
        if not attempt.get("passed"):
            continue
        adapted = _initial_adapted_case(stored)
        step_count = len(_steps(adapted))
        if not step_count:
            continue
        if max_reasoning_steps > 0 and step_count > max_reasoning_steps:
            continue
        candidates.append((path, stored))
    random.Random(seed).shuffle(candidates)
    selected = candidates[: max(1, min(sample_count, len(candidates)))]
    exp_id = f"fault_v023_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"
    out_dir = output_root / exp_id
    out_dir.mkdir(parents=True, exist_ok=False)
    controls: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    detail_path = out_dir / "mutation_details.jsonl"
    affected_tp = affected_fp = affected_fn = 0
    root_tp = root_fp = root_fn = 0
    seen_signatures: set[str] = set()
    with detail_path.open("w", encoding="utf-8") as detail:
        for case_number, (_path, stored) in enumerate(selected, 1):
            adapted = _initial_adapted_case(stored)
            initial = verify_case(adapted, prefer_z3=prefer_z3, compute_counterfactuals=False)
            locked = _lock_baseline_chain(adapted, initial)
            locked["verification_policy"] = adapted["verification_policy"]
            baseline = verify_case(locked, prefer_z3=prefer_z3, compute_counterfactuals=False)
            clean_invalid = _overall_invalid(baseline)
            controls.append({
                "record_id": stored.get("record_id"),
                "clean_pass": not clean_invalid,
                "clean_false_rejection": clean_invalid,
                "reasoning_nodes": len(_steps(locked)),
                "engine": baseline.get("engine"),
            })
            node_candidates = _candidate_nodes(locked, baseline)
            targets = _difficulty_targets(node_candidates, difficulties)
            planned: list[tuple[str, dict[str, Any], str]] = []
            for difficulty, chosen in targets:
                for fault_type in fault_types:
                    if fault_type == "answer_flip":
                        continue
                    planned.append((difficulty, chosen, fault_type))
            if "answer_flip" in fault_types and node_candidates:
                planned.append(("final_only", node_candidates[-1], "answer_flip"))

            for plan_index, (difficulty, chosen, fault_type) in enumerate(planned, 1):
                accepted: tuple[dict[str, Any], dict[str, Any], dict[str, Any], float] | None = None
                structural_exception: Exception | None = None
                not_applicable = False
                attempts = 1 if fault_type in {"polarity_flip", "entity_swap", "argument_swap", "step_deletion", "answer_flip"} else max_resample_attempts
                for resample in range(attempts):
                    applied = _apply_fault(locked, baseline, chosen, fault_type, case_number * 1000 + plan_index, resample)
                    if applied is None:
                        not_applicable = True
                        break
                    mutated, meta = applied
                    signature = json.dumps({
                        "record_id": stored.get("record_id"),
                        "fault_type": fault_type,
                        "mutated_node_id": meta.get("mutated_node_id"),
                        "mutated_text": meta.get("mutated_text"),
                        "mutated_depends_on": meta.get("mutated_depends_on"),
                        "deleted_node_id": meta.get("deleted_node_id"),
                        "mutated_answer": meta.get("mutated_answer"),
                    }, sort_keys=True, ensure_ascii=False)
                    if signature in seen_signatures:
                        continue
                    started = time.perf_counter()
                    try:
                        result = verify_case_incremental(
                            locked, mutated, baseline, prefer_z3=prefer_z3, validate_against_full=False
                        )
                        runtime = (time.perf_counter() - started) * 1000
                        if fault_type in STRUCTURAL_FAULTS:
                            skipped.append({
                                "record_id": stored.get("record_id"), "difficulty": difficulty, "fault_type": fault_type,
                                "reason": "step_deletion_did_not_trigger_structural_rejection", **meta,
                            })
                            continue
                        gate_ok, gate_detail = _oracle_mutation_valid(locked, baseline, mutated, meta, fault_type)
                        meta["oracle_detail"] = gate_detail
                        if not gate_ok:
                            if fault_type in {"predicate_swap", "parent_deletion", "wrong_parent"}:
                                continue
                            skipped.append({"record_id": stored.get("record_id"), "difficulty": difficulty, "fault_type": fault_type, "reason": "independent_mutation_oracle_failed", "oracle_detail": gate_detail, **meta})
                            break
                        accepted = (mutated, meta, result, runtime)
                        break
                    except Exception as exc:
                        runtime = (time.perf_counter() - started) * 1000
                        if fault_type == "step_deletion":
                            structural_exception = exc
                            accepted = (mutated, meta, {}, runtime)
                            break
                        skipped.append({"record_id": stored.get("record_id"), "difficulty": difficulty, "fault_type": fault_type, "reason": f"unexpected_exception:{type(exc).__name__}:{exc}", **meta})
                        break
                if accepted is None:
                    reason = "not_applicable_to_target" if not_applicable else "no_valid_unique_mutation_after_resampling"
                    skipped.append({"record_id": stored.get("record_id"), "difficulty": difficulty, "fault_type": fault_type, "reason": reason, "mutated_node_id": chosen.get("id")})
                    continue
                mutated, meta, result, runtime = accepted
                signature = json.dumps({
                    "record_id": stored.get("record_id"), "fault_type": fault_type,
                    "mutated_node_id": meta.get("mutated_node_id"), "mutated_text": meta.get("mutated_text"),
                    "mutated_depends_on": meta.get("mutated_depends_on"), "deleted_node_id": meta.get("deleted_node_id"),
                    "mutated_answer": meta.get("mutated_answer"),
                }, sort_keys=True, ensure_ascii=False)
                seen_signatures.add(signature)
                expected_root = str(meta["expected_error_node_id"])
                expected_affected = _dependency_closure(baseline, expected_root)
                if fault_type == "step_deletion":
                    predicted_errors: set[str] = set()
                    predicted_roots: set[str] = {expected_root} if structural_exception and expected_root in str(structural_exception) else set()
                    detected = structural_exception is not None
                    root_exact = bool(predicted_roots)
                    rejection_channel = "structural_schema"
                    summary_data: dict[str, Any] = {}
                    error_message = f"{type(structural_exception).__name__}: {structural_exception}" if structural_exception else ""
                else:
                    predicted_errors = set(_error_nodes(result))
                    predicted_roots = set(_predicted_roots(result))
                    detected = _overall_invalid(result)
                    root_exact = expected_root in predicted_roots
                    rejection_channel = "logical_semantic"
                    summary_data = result.get("summary") or {}
                    error_message = ""
                atp, afp, afn = _set_metrics(predicted_errors, expected_affected)
                rtp, rfp, rfn = _set_metrics(predicted_roots, {expected_root})
                affected_tp += atp; affected_fp += afp; affected_fn += afn
                root_tp += rtp; root_fp += rfp; root_fn += rfn
                row = {
                    "record_id": stored.get("record_id"), "case_number": case_number,
                    "difficulty": difficulty, "fault_type": fault_type, "rejection_channel": rejection_channel,
                    "mutated_node_id": meta.get("mutated_node_id"), "expected_error_node_id": expected_root,
                    "expected_affected_nodes": ";".join(sorted(expected_affected)),
                    "detected": detected, "root_exact_localized": root_exact,
                    "predicted_root_nodes": ";".join(sorted(predicted_roots)),
                    "predicted_error_nodes": ";".join(sorted(predicted_errors)),
                    "affected_tp": atp, "affected_fp": afp, "affected_fn": afn,
                    "root_tp": rtp, "root_fp": rfp, "root_fn": rfn,
                    "final_proof_status": summary_data.get("final_proof_status", "structural_exception" if structural_exception else None),
                    "final_chain_status": summary_data.get("final_chain_status", "structural_exception" if structural_exception else None),
                    "invalid_reasoning_count": summary_data.get("invalid_reasoning_count"),
                    "chain_error_count": summary_data.get("chain_error_count"),
                    "runtime_ms": round(runtime, 3), "error": error_message,
                    **{k: v for k, v in meta.items() if k != "expected_error_node_id"},
                }
                rows.append(row)
                detail.write(json.dumps({"row": row, "graph_summary": summary_data}, ensure_ascii=False) + "\n")

    clean_n = len(controls)
    clean_pass = sum(_truth(r["clean_pass"]) for r in controls)
    logical_rows = [r for r in rows if r["rejection_channel"] == "logical_semantic"]
    structural_rows = [r for r in rows if r["rejection_channel"] == "structural_schema"]
    n = len(rows)
    detected = sum(_truth(r["detected"]) for r in rows)
    logical_detected = sum(_truth(r["detected"]) for r in logical_rows)
    structural_detected = sum(_truth(r["detected"]) for r in structural_rows)
    root_precision = root_tp / (root_tp + root_fp) if root_tp + root_fp else None
    root_recall = root_tp / (root_tp + root_fn) if root_tp + root_fn else None
    affected_precision = affected_tp / (affected_tp + affected_fp) if affected_tp + affected_fp else None
    affected_recall = affected_tp / (affected_tp + affected_fn) if affected_tp + affected_fn else None
    affected_f1 = 2 * affected_precision * affected_recall / (affected_precision + affected_recall) if affected_precision is not None and affected_recall is not None and affected_precision + affected_recall else None
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": exp_id,
        "experiment_type": "controlled_fault_injection_v023",
        "source_run_id": source_run_dir.name,
        "created_at": _now(),
        "sampled_clean_cases": clean_n,
        "clean_acceptance_count": clean_pass,
        "valid_graph_acceptance_rate": round(clean_pass / clean_n * 100, 2) if clean_n else None,
        "false_rejection_rate": round((clean_n - clean_pass) / clean_n * 100, 2) if clean_n else None,
        "mutation_count": n,
        "logical_semantic_mutation_count": len(logical_rows),
        "structural_mutation_count": len(structural_rows),
        "skipped_mutation_count": len(skipped),
        "invalid_graph_rejection_count": detected,
        "invalid_graph_rejection_rate": round(detected / n * 100, 2) if n else None,
        "logical_semantic_rejection_rate": round(logical_detected / len(logical_rows) * 100, 2) if logical_rows else None,
        "structural_schema_rejection_rate": round(structural_detected / len(structural_rows) * 100, 2) if structural_rows else None,
        "false_acceptance_rate": round((n - detected) / n * 100, 2) if n else None,
        "root_exact_localization_rate": round(sum(_truth(r["root_exact_localized"]) for r in rows) / n * 100, 2) if n else None,
        "root_set_precision": round(root_precision, 4) if root_precision is not None else None,
        "root_set_recall": round(root_recall, 4) if root_recall is not None else None,
        "affected_node_precision": round(affected_precision, 4) if affected_precision is not None else None,
        "affected_node_recall": round(affected_recall, 4) if affected_recall is not None else None,
        "affected_node_f1": round(affected_f1, 4) if affected_f1 is not None else None,
        "duplicate_mutations_removed": True,
        "strict_empty_parent_policy": True,
        "mutation_validity_gate": True,
        "mutation_oracle": "independent_horn_global_and_authored_local_support",
        "new_api_calls": 0,
        "seed": seed,
        "fault_types": list(fault_types),
        "difficulties_requested": list(difficulties),
        "max_reasoning_steps": max_reasoning_steps,
        "verification_backend": "z3" if prefer_z3 else "finite_horn",
    }
    by_type = _aggregate(rows, "fault_type")
    by_difficulty = _aggregate(rows, "difficulty")
    by_channel = _aggregate(rows, "rejection_channel")
    paper = []
    for x in by_type:
        paper.append({"analysis": "Fault type", "group": x["fault_type"], "n": x["n"], "detection_rate_percent": x["detection_rate"], "root_exact_localization_rate_percent": x["root_exact_localization_rate"]})
    for x in by_difficulty:
        paper.append({"analysis": "Difficulty", "group": x["difficulty"], "n": x["n"], "detection_rate_percent": x["detection_rate"], "root_exact_localization_rate_percent": x["root_exact_localization_rate"]})
    config = {
        "sample_count_requested": sample_count, "sample_count_used": clean_n, "seed": seed,
        "fault_types": list(fault_types), "difficulties": list(difficulties), "prefer_z3": prefer_z3,
        "max_reasoning_steps": max_reasoning_steps, "max_resample_attempts": max_resample_attempts,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({**summary, "by_fault_type": by_type, "by_difficulty": by_difficulty, "by_rejection_channel": by_channel}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "clean_controls.csv").write_text(_csv_text(controls), encoding="utf-8-sig")
    (out_dir / "fault_mutations.csv").write_text(_csv_text(rows), encoding="utf-8-sig")
    (out_dir / "skipped_mutations.csv").write_text(_csv_text(skipped), encoding="utf-8-sig")
    (out_dir / "metrics_by_fault_type.csv").write_text(_csv_text(by_type), encoding="utf-8-sig")
    (out_dir / "metrics_by_difficulty.csv").write_text(_csv_text(by_difficulty), encoding="utf-8-sig")
    (out_dir / "metrics_by_rejection_channel.csv").write_text(_csv_text(by_channel), encoding="utf-8-sig")
    (out_dir / "paper_table.csv").write_text(_csv_text(paper), encoding="utf-8-sig")
    report = f"""# v023 Controlled Fault Injection Report

- Source run: `{source_run_dir.name}`
- Clean cases sampled: {clean_n}
- Valid unique mutations: {n}
- Skipped/non-fault candidates: {len(skipped)}
- New API calls: 0
- Verification backend: {summary['verification_backend']}

## Corrected evaluation policy
- Empty authored parents are strict Chain failures.
- Mutations pass an independent Horn support oracle; non-fault predicate/parent replacements are resampled.
- Detection rates are conditional on oracle-confirmed valid mutations; skipped/non-applicable candidates are reported separately.
- Identical local/upstream mutations are deduplicated; one-step cases use `single_step`.
- Step deletion is reported as structural/schema rejection.
- Root localization and downstream affected-node propagation use separate metrics.

## Core metrics
- Valid graph acceptance rate: {summary['valid_graph_acceptance_rate']}%
- Overall invalid graph rejection rate: {summary['invalid_graph_rejection_rate']}%
- Logical/semantic rejection rate: {summary['logical_semantic_rejection_rate']}%
- Structural/schema rejection rate: {summary['structural_schema_rejection_rate']}%
- Root exact localization rate: {summary['root_exact_localization_rate']}%
- Affected-node precision / recall / F1: {summary['affected_node_precision']} / {summary['affected_node_recall']} / {summary['affected_node_f1']}

This report evaluates the verifier only. Blind/Guided repair should be run after reviewing this corrected benchmark.
"""
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "experiment_id": exp_id,
        "output_dir": str(out_dir),
        "summary": {**summary, "by_fault_type": by_type, "by_difficulty": by_difficulty, "by_rejection_channel": by_channel},
        "files": sorted(p.name for p in out_dir.iterdir()),
    }


def _usage_add(total: dict[str,int], usage: dict[str,Any]) -> None:
    for key in ("input_tokens","output_tokens","total_tokens"):
        total[key]+=int(usage.get(key) or 0)


def run_natural_repair_experiment(
    source_run_dir: Path,
    output_root: Path,
    *, modes: Iterable[str] = ("no_repair","blind","guided","cascade"),
    max_cases: int = 0,
    model: str = "gpt-5.6",
    reasoning_effort: str = "low",
    max_output_tokens: int = 5000,
    prefer_z3: bool = True,
    client: Any = None,
) -> dict[str, Any]:
    modes=tuple(x for x in modes if x in {"no_repair","blind","guided","cascade"})
    if not modes: raise ValueError("Select at least one repair condition")
    failures=[]
    for path in _case_files(source_run_dir):
        stored=_load_case(path); attempt=(stored.get("attempts") or [{}])[0]
        if not attempt.get("passed"): failures.append(stored)
    if max_cases>0: failures=failures[:max_cases]
    exp_id=f"repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir=output_root/exp_id; out_dir.mkdir(parents=True,exist_ok=False)
    rows=[]; details=[]
    for stored in failures:
        record=stored["record"]; original=_initial_output(stored)
        initial_analysis=_analyze_transformed(record,original,use_llm_formalizer=False,model=model,reasoning_effort=reasoning_effort,prefer_z3=prefer_z3,client=client)
        initial_pass=not _needs_repair(initial_analysis); initial_correct=bool(initial_analysis["classification"].get("answer_correct"))
        for mode in modes:
            current_output=deepcopy(original); analysis=initial_analysis; calls=[]
            if mode in {"blind","guided"}:
                packet=build_repair_packet(analysis,mode=mode)
                call=_repair_call(record,current_output,packet,model=model,reasoning_effort=reasoning_effort,max_output_tokens=max_output_tokens,client=client)
                calls.append(call); current_output=call["llm_output"]
                analysis=_analyze_transformed(record,current_output,use_llm_formalizer=False,model=model,reasoning_effort=reasoning_effort,prefer_z3=prefer_z3,client=client)
            elif mode=="cascade":
                packet=build_repair_packet(analysis,mode="blind")
                call=_repair_call(record,current_output,packet,model=model,reasoning_effort=reasoning_effort,max_output_tokens=max_output_tokens,client=client)
                calls.append(call); current_output=call["llm_output"]
                analysis=_analyze_transformed(record,current_output,use_llm_formalizer=False,model=model,reasoning_effort=reasoning_effort,prefer_z3=prefer_z3,client=client)
                if _needs_repair(analysis):
                    packet=build_repair_packet(analysis,mode="guided")
                    call=_repair_call(record,current_output,packet,model=model,reasoning_effort=reasoning_effort,max_output_tokens=max_output_tokens,client=client)
                    calls.append(call); current_output=call["llm_output"]
                    analysis=_analyze_transformed(record,current_output,use_llm_formalizer=False,model=model,reasoning_effort=reasoning_effort,prefer_z3=prefer_z3,client=client)
            final_pass=not _needs_repair(analysis); final_correct=bool(analysis["classification"].get("answer_correct"))
            usage={"input_tokens":0,"output_tokens":0,"total_tokens":0}
            for c in calls: _usage_add(usage,c.get("usage") or {})
            row={"record_id":stored.get("record_id"),"condition":mode,"initial_pass":initial_pass,"final_pass":final_pass,
                 "graph_recovery":(not initial_pass and final_pass),"initial_answer_correct":initial_correct,"final_answer_correct":final_correct,
                 "answer_recovery":(not initial_correct and final_correct),"harmful_repair":(initial_correct and not final_correct),
                 "api_calls":len(calls),**usage,"final_answer":analysis["classification"].get("predicted_label"),"gold_answer":analysis["classification"].get("gold_label")}
            rows.append(row); details.append({"row":row,"initial_output":original,"final_output":current_output,"final_analysis":analysis,"repair_calls":calls})
    groups=[]
    for mode in modes:
        items=[r for r in rows if r["condition"]==mode]; n=len(items)
        groups.append({"condition":mode,"n":n,"graph_pass":sum(_truth(r["final_pass"]) for r in items),
                       "graph_recovery":sum(_truth(r["graph_recovery"]) for r in items),"answer_correct":sum(_truth(r["final_answer_correct"]) for r in items),
                       "answer_recovery":sum(_truth(r["answer_recovery"]) for r in items),"harmful_repair":sum(_truth(r["harmful_repair"]) for r in items),
                       "api_calls":sum(int(r["api_calls"]) for r in items),"input_tokens":sum(int(r["input_tokens"]) for r in items),
                       "output_tokens":sum(int(r["output_tokens"]) for r in items),"tokens_per_success":round(sum(int(r["total_tokens"]) for r in items)/max(1,sum(_truth(r["graph_recovery"]) for r in items)),2)})
    summary={"schema_version":SCHEMA_VERSION,"experiment_id":exp_id,"experiment_type":"natural_failure_repair","source_run_id":source_run_dir.name,
             "created_at":_now(),"natural_failure_cases":len(failures),"conditions":groups,"gold_answer_was_sent":False,
             "total_api_calls":sum(x["api_calls"] for x in groups),"total_input_tokens":sum(x["input_tokens"] for x in groups),"total_output_tokens":sum(x["output_tokens"] for x in groups)}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    (out_dir/"repair_cases.csv").write_text(_csv_text(rows),encoding="utf-8-sig")
    (out_dir/"repair_summary.csv").write_text(_csv_text(groups),encoding="utf-8-sig")
    (out_dir/"paper_table.csv").write_text(_csv_text(groups),encoding="utf-8-sig")
    with (out_dir/"repair_details.jsonl").open("w",encoding="utf-8") as f:
        for d in details: f.write(json.dumps(d,ensure_ascii=False)+"\n")
    report="# v023 Natural Failure Repair Comparison\n\n"+"\n".join(f"- {x['condition']}: graph recovery {x['graph_recovery']}/{x['n']}, harmful repair {x['harmful_repair']}, API calls {x['api_calls']}" for x in groups)+"\n\nGold labels were never sent to the repair model.\n"
    (out_dir/"report.md").write_text(report,encoding="utf-8")
    return {"experiment_id":exp_id,"output_dir":str(out_dir),"summary":summary,"files":sorted(p.name for p in out_dir.iterdir())}


def list_experiments(output_root: Path) -> list[dict[str, Any]]:
    rows=[]
    if not output_root.exists(): return rows
    for p in sorted(output_root.iterdir(),key=lambda x:x.stat().st_mtime,reverse=True):
        if not p.is_dir(): continue
        s=p/"summary.json"
        if not s.exists(): continue
        try: data=json.loads(s.read_text(encoding="utf-8"))
        except Exception: continue
        rows.append({"experiment_id":p.name,"experiment_type":data.get("experiment_type"),"created_at":data.get("created_at"),"source_run_id":data.get("source_run_id")})
    return rows


def archive_experiment(output_root: Path, experiment_id: str) -> Path:
    exp=(output_root/experiment_id).resolve(); root=output_root.resolve()
    if root not in exp.parents or not exp.exists(): raise ValueError("Unknown experiment")
    archive=output_root/f"{experiment_id}.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for p in exp.rglob("*"):
            if p.is_file(): z.write(p,p.relative_to(exp))
    return archive
