from __future__ import annotations

from copy import deepcopy
from typing import Any

from .logic import Atom, is_variable
from .parser import parse_statement
from .verifier import verify_case, verify_case_incremental


def _entity_text(value: str) -> str:
    return value.replace("_", " ")


def _predicate_text(value: str) -> str:
    return value.replace("_", " ")


def atom_to_controlled_english(atom: Atom) -> str:
    if any(is_variable(arg) for arg in atom.args):
        raise ValueError("Mutation testing only supports grounded reasoning atoms")
    if len(atom.args) == 1:
        subject = _entity_text(atom.args[0]).title()
        negation = "not " if atom.negated else ""
        return f"{subject} is {negation}{_predicate_text(atom.predicate)}."
    if len(atom.args) == 2:
        subject = _entity_text(atom.args[0])
        obj = _entity_text(atom.args[1])
        if atom.negated:
            return f"The {subject} does not {atom.predicate} the {obj}."
        # Controlled parser accepts the lemma directly; grammatical inflection is
        # intentionally avoided so every generated mutation round-trips.
        return f"The {subject} {atom.predicate} the {obj}."
    raise ValueError("Mutation testing supports unary and binary atoms only")


def _case_steps(case: dict[str, Any]) -> list[Any]:
    output = case.get("llm_output") or {}
    return output.get("reasoning_steps") or case.get("reasoning_steps") or []


def _step_id_text(item: Any, index: int) -> tuple[str, str]:
    if isinstance(item, str):
        return f"s{index}", item
    if isinstance(item, dict):
        return str(item.get("id") or f"s{index}"), str(item.get("text") or "")
    return f"s{index}", ""


def _replace_step_text(case: dict[str, Any], node_id: str, new_text: str) -> dict[str, Any]:
    updated = deepcopy(case)
    output = updated.setdefault("llm_output", {})
    steps = output.get("reasoning_steps")
    if steps is None:
        steps = updated.get("reasoning_steps") or []
        output["reasoning_steps"] = deepcopy(steps)
    else:
        output["reasoning_steps"] = deepcopy(steps)
    for index, item in enumerate(output["reasoning_steps"], 1):
        item_id, _ = _step_id_text(item, index)
        if item_id != node_id:
            continue
        if isinstance(item, str):
            output["reasoning_steps"][index - 1] = {"id": item_id, "text": new_text}
        else:
            output["reasoning_steps"][index - 1] = {**item, "id": item_id, "text": new_text}
        return updated
    raise ValueError(f"Reasoning node not found: {node_id}")



def _replace_step_dependencies(case: dict[str, Any], node_id: str, dependencies: list[str]) -> dict[str, Any]:
    updated = deepcopy(case)
    output = updated.setdefault("llm_output", {})
    raw_steps = output.get("reasoning_steps")
    if raw_steps is None:
        raw_steps = deepcopy(updated.get("reasoning_steps") or [])
    else:
        raw_steps = deepcopy(raw_steps)
    normalized = []
    found = False
    for index, item in enumerate(raw_steps, 1):
        item_id, text = _step_id_text(item, index)
        payload = {"id": item_id, "text": text}
        if isinstance(item, dict):
            payload = {**item, **payload}
        if item_id == node_id:
            payload["depends_on"] = list(dependencies)
            found = True
        normalized.append(payload)
    if not found:
        raise ValueError(f"Reasoning node not found: {node_id}")
    output["reasoning_steps"] = normalized
    return updated


def _lock_baseline_chain(case: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Freeze the baseline inferred LLM route as explicit depends_on provenance.

    Mutation testing must answer two different questions: can the claim be repaired
    through another proof, and did the originally inferred LLM route break?  By
    materializing baseline dependencies, downstream nodes cannot silently reroute
    around the injected error during this test.
    """
    locked = deepcopy(case)
    output = locked.setdefault("llm_output", {})
    raw_steps = output.get("reasoning_steps")
    if raw_steps is None:
        raw_steps = deepcopy(locked.get("reasoning_steps") or [])
    else:
        raw_steps = deepcopy(raw_steps)
    lookup = {str(node.get("id")): node for node in baseline.get("nodes", [])}
    normalized_steps: list[dict[str, Any]] = []
    for index, item in enumerate(raw_steps, 1):
        node_id, text = _step_id_text(item, index)
        node = lookup.get(node_id, {})
        dependencies = sorted(set((node.get("source_matches") or []) + (node.get("reasoning_dependencies") or [])))
        payload = {"id": node_id, "text": text, "depends_on": dependencies}
        if isinstance(item, dict):
            payload = {**item, **payload}
        normalized_steps.append(payload)
    output["reasoning_steps"] = normalized_steps
    final_node = lookup.get("final", {})
    output["answer_depends_on"] = sorted(set(final_node.get("reasoning_dependencies") or []))
    return locked


def mutation_test_case(
    case: dict[str, Any],
    *,
    prefer_z3: bool = True,
    max_nodes: int = 20,
) -> dict[str, Any]:
    initial_baseline = verify_case(case, prefer_z3=prefer_z3, compute_counterfactuals=False)
    locked_case = _lock_baseline_chain(case, initial_baseline)
    baseline = verify_case(locked_case, prefer_z3=prefer_z3, compute_counterfactuals=False)
    baseline_nodes = {node["id"]: node for node in baseline.get("nodes", [])}
    candidates: list[tuple[str, str, Atom]] = []
    for index, item in enumerate(_case_steps(locked_case), 1):
        node_id, text = _step_id_text(item, index)
        node = baseline_nodes.get(node_id)
        if not node or node.get("proof_status") != "valid":
            continue
        parsed = parse_statement(text)
        if not isinstance(parsed.formula, Atom) or any(is_variable(arg) for arg in parsed.formula.args):
            continue
        candidates.append((node_id, text, parsed.formula))
        if len(candidates) >= max_nodes:
            break

    rows: list[dict[str, Any]] = []
    for ordinal, (node_id, original_text, atom) in enumerate(candidates, 1):
        claim_mutations = [
            ("polarity_flip", atom.complement(), "contradiction"),
            (
                "novel_predicate",
                Atom(f"mutation_novel_{ordinal}_{atom.predicate}", atom.args, atom.negated),
                "ungrounded",
            ),
            (
                "entity_swap",
                Atom(atom.predicate, (f"mutation_entity_{ordinal}", *atom.args[1:]), atom.negated),
                "ungrounded",
            ),
        ]
        if len(atom.args) == 2 and atom.args[0] != atom.args[1]:
            claim_mutations.append(("argument_swap", Atom(atom.predicate, (atom.args[1], atom.args[0]), atom.negated), "ungrounded"))

        for mutation_type, mutated_atom, expected in claim_mutations:
            mutated_text = atom_to_controlled_english(mutated_atom)
            updated_case = _replace_step_text(locked_case, node_id, mutated_text)
            result = verify_case_incremental(
                locked_case, updated_case, baseline, prefer_z3=prefer_z3, validate_against_full=True
            )
            node_after = next(node for node in result["nodes"] if node["id"] == node_id)
            observed = node_after.get("proof_status")
            rows.append({
                "node_id": node_id, "mutation_type": mutation_type,
                "original_text": original_text, "mutated_text": mutated_text,
                "expected_proof_status": expected, "expected_chain_status": None,
                "observed_proof_status": observed,
                "detected_as_expected": observed == expected,
                "observed_chain_status": node_after.get("chain_status"),
                "final_proof_status": result.get("summary", {}).get("final_proof_status"),
                "final_chain_status": result.get("summary", {}).get("final_chain_status"),
                "strict_authored_path_locked": True,
                "mutated_node_reaches_final_in_baseline": bool(baseline_nodes.get(node_id, {}).get("chain_reaches_final")),
                "parity_match": result.get("incremental", {}).get("parity_validation", {}).get("matches_full_verification"),
                "revalidation_mode": result.get("incremental", {}).get("mode"),
                "runtime_ms": result.get("incremental", {}).get("incremental_runtime_ms"),
            })

        # Structural mutation: preserve the claim but replace its authored parents
        # with one premise that cannot derive it on its own.
        premise_ids = [str(node.get("id")) for node in baseline.get("nodes", []) if node.get("kind") == "premise"]
        wrong_parent = premise_ids[-1] if premise_ids else None
        if wrong_parent:
            updated_case = _replace_step_dependencies(locked_case, node_id, [wrong_parent])
            result = verify_case_incremental(
                locked_case, updated_case, baseline, prefer_z3=prefer_z3, validate_against_full=True
            )
            node_after = next(node for node in result["nodes"] if node["id"] == node_id)
            observed_chain = node_after.get("chain_status")
            rows.append({
                "node_id": node_id, "mutation_type": "wrong_declared_parent",
                "original_text": original_text, "mutated_text": original_text,
                "mutated_depends_on": [wrong_parent],
                "expected_proof_status": "valid",
                "expected_chain_status": "insufficient_declared_support",
                "observed_proof_status": node_after.get("proof_status"),
                "observed_chain_status": observed_chain,
                "detected_as_expected": node_after.get("proof_status") == "valid" and observed_chain == "insufficient_declared_support",
                "final_proof_status": result.get("summary", {}).get("final_proof_status"),
                "final_chain_status": result.get("summary", {}).get("final_chain_status"),
                "strict_authored_path_locked": True,
                "mutated_node_reaches_final_in_baseline": bool(baseline_nodes.get(node_id, {}).get("chain_reaches_final")),
                "parity_match": result.get("incremental", {}).get("parity_validation", {}).get("matches_full_verification"),
                "revalidation_mode": result.get("incremental", {}).get("mode"),
                "runtime_ms": result.get("incremental", {}).get("incremental_runtime_ms"),
            })

    polarity = [row for row in rows if row["mutation_type"] == "polarity_flip"]
    novel = [row for row in rows if row["mutation_type"] == "novel_predicate"]
    detected = sum(row["detected_as_expected"] for row in rows)
    mutation_types = sorted({row["mutation_type"] for row in rows})
    type_summary = {
        mutation_type: {
            "count": sum(row["mutation_type"] == mutation_type for row in rows),
            "detected_count": sum(row["mutation_type"] == mutation_type and row["detected_as_expected"] for row in rows),
        }
        for mutation_type in mutation_types
    }
    for values in type_summary.values():
        values["detection_percent"] = round(values["detected_count"] / values["count"] * 100, 2) if values["count"] else None
    return {
        "schema_version": "0.15.0",
        "case_id": str(case.get("id") or "case"),
        "engine": baseline.get("engine"),
        "baseline_summary": baseline.get("summary"),
        "summary": {
            "eligible_reasoning_nodes": len(candidates),
            "mutation_count": len(rows),
            "detected_as_expected_count": detected,
            "overall_detection_percent": round(detected / len(rows) * 100, 2) if rows else None,
            "polarity_flip_count": len(polarity),
            "polarity_flip_detected_count": sum(row["detected_as_expected"] for row in polarity),
            "polarity_flip_detection_percent": round(sum(row["detected_as_expected"] for row in polarity) / len(polarity) * 100, 2) if polarity else None,
            "novel_predicate_count": len(novel),
            "novel_predicate_detected_count": sum(row["detected_as_expected"] for row in novel),
            "novel_predicate_detection_percent": round(sum(row["detected_as_expected"] for row in novel) / len(novel) * 100, 2) if novel else None,
            "parity_pass_count": sum(row.get("parity_match") is True for row in rows),
            "strict_authored_path_mode": True,
            "final_chain_blocked_or_invalid_count": sum(row.get("final_chain_status") != "valid" for row in rows),
            "mutation_type_summary": type_summary,
        },
        "mutations": rows,
    }
