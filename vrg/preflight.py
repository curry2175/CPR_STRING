from __future__ import annotations

from typing import Any
import re

from .logic import formula_to_text, predicates_in_formula
from .parser import parse_question, parse_statement
from .semantic import SemanticLayer, parse_semantic_relations


def _coerce_items(raw: Any, prefix: str) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(raw, list):
        return [], [f"{prefix} must be a list"]
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, 1):
        if isinstance(item, str):
            node_id, text, depends_on = f"{prefix}{index}", item, []
        elif isinstance(item, dict):
            node_id = str(item.get("id") or f"{prefix}{index}")
            text = str(item.get("text") or "")
            depends_on = item.get("depends_on") or item.get("dependencies") or []
            if isinstance(depends_on, str):
                depends_on = [piece.strip() for piece in depends_on.split(",") if piece.strip()]
            if not isinstance(depends_on, list):
                errors.append(f"{node_id}: depends_on must be a list")
                depends_on = []
            depends_on = [str(value) for value in depends_on]
        else:
            errors.append(f"{prefix}{index}: unsupported item type {type(item).__name__}")
            continue
        if node_id in seen:
            errors.append(f"Duplicate node id: {node_id}")
        seen.add(node_id)
        if not text.strip():
            errors.append(f"{node_id}: empty text")
        items.append({"id": node_id, "text": text, "depends_on": depends_on})
    return items, errors


def _parse_row(node_id: str, kind: str, text: str, layer: SemanticLayer) -> dict[str, Any]:
    parsed = parse_statement(text)
    if parsed.formula is None:
        return {
            "id": node_id,
            "kind": kind,
            "text": text,
            "parse_status": "untranslatable",
            "raw_formal": None,
            "normalized_formal": None,
            "semantic_normalizations": [],
            "predicates": [],
            "error": parsed.error,
        }
    normalized, normalizations = layer.normalize_formula(parsed.formula)
    return {
        "id": node_id,
        "kind": kind,
        "text": text,
        "parse_status": "parseable",
        "raw_formal": formula_to_text(parsed.formula),
        "normalized_formal": formula_to_text(normalized),
        "semantic_normalizations": normalizations,
        "predicates": [f"{name}/{arity}" for name, arity in sorted(predicates_in_formula(normalized))],
        "error": None,
    }


def preflight_case(case: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(case, dict):
        raise ValueError("Case must be a JSON object")

    premises_raw = case.get("premises") or case.get("context") or []
    output = case.get("llm_output") or {}
    if not isinstance(output, dict):
        errors.append("llm_output must be an object")
        output = {}
    steps_raw = output.get("reasoning_steps") or case.get("reasoning_steps") or []
    premises, item_errors = _coerce_items(premises_raw, "p")
    errors.extend(item_errors)
    steps, item_errors = _coerce_items(steps_raw, "s")
    errors.extend(item_errors)

    try:
        relations = parse_semantic_relations(case.get("semantic_relations"))
    except Exception as exc:
        relations = []
        errors.append(f"semantic_relations: {exc}")
    layer = SemanticLayer(relations)

    all_ids = [relation.id for relation in relations] + [item["id"] for item in premises] + [item["id"] for item in steps]
    duplicate_cross_ids = sorted({node_id for node_id in all_ids if all_ids.count(node_id) > 1})
    for node_id in duplicate_cross_ids:
        message = f"Node id is reused across semantic/premise/reasoning sections: {node_id}"
        if message not in errors:
            errors.append(message)

    available_ids = {relation.id for relation in relations} | {item["id"] for item in premises}
    for item in steps:
        dependencies = item.get("depends_on") or []
        if len(dependencies) != len(set(dependencies)):
            warnings.append(f"{item['id']}: duplicate depends_on entries were provided")
        for dependency in dependencies:
            if dependency == item["id"]:
                errors.append(f"{item['id']}: a reasoning node cannot depend on itself")
            elif dependency not in available_ids:
                errors.append(f"{item['id']}: depends_on references unknown or future node {dependency}")
        available_ids.add(item["id"])
    answer_depends_on = output.get("answer_depends_on") or case.get("answer_depends_on") or []
    if isinstance(answer_depends_on, str):
        answer_depends_on = [piece.strip() for piece in answer_depends_on.split(",") if piece.strip()]
    if not isinstance(answer_depends_on, list):
        errors.append("answer_depends_on must be a list")
        answer_depends_on = []
    for dependency in answer_depends_on:
        if str(dependency) not in available_ids:
            errors.append(f"answer_depends_on references unknown node {dependency}")

    rows = [_parse_row(item["id"], "premise", item["text"], layer) for item in premises]
    reasoning_rows = [_parse_row(item["id"], "reasoning", item["text"], layer) for item in steps]
    for row, item in zip(reasoning_rows, steps):
        row["declared_dependencies"] = item.get("depends_on") or []
    rows.extend(reasoning_rows)

    question_text = str(case.get("question") or "")
    question_parsed = parse_question(question_text)
    if question_parsed.formula is None:
        question_row = {
            "id": "question",
            "kind": "question",
            "text": question_text,
            "parse_status": "untranslatable",
            "raw_formal": None,
            "normalized_formal": None,
            "semantic_normalizations": [],
            "predicates": [],
            "error": question_parsed.error,
        }
        errors.append(f"question: {question_parsed.error}")
    else:
        q_normalized, q_norms = layer.normalize_formula(question_parsed.formula)
        question_row = {
            "id": "question",
            "kind": "question",
            "text": question_text,
            "parse_status": "parseable",
            "raw_formal": formula_to_text(question_parsed.formula),
            "normalized_formal": formula_to_text(q_normalized),
            "semantic_normalizations": q_norms,
            "predicates": [f"{name}/{arity}" for name, arity in sorted(predicates_in_formula(q_normalized))],
            "error": None,
        }

    predicted = str(output.get("answer", case.get("predicted_answer", ""))).strip().lower()
    gold = str(case.get("gold_answer", "")).strip().lower()
    if predicted not in {"yes", "no"}:
        errors.append(f"LLM answer must be strictly Yes or No; received: {predicted!r}")
    if gold not in {"yes", "no"}:
        errors.append(f"gold_answer must be strictly Yes or No; received: {gold!r}")


    compound_steps = []
    for item in steps:
        pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\s*;\s*", item["text"].strip()) if piece.strip()]
        if len(pieces) > 1:
            compound_steps.append(item["id"])
    if compound_steps:
        warnings.append(
            f"{len(compound_steps)} reasoning step(s) contain multiple sentence-level claims: {', '.join(compound_steps)}"
        )

    untranslatable = [row for row in rows if row["parse_status"] != "parseable"]
    if untranslatable:
        warnings.append(
            f"{len(untranslatable)} premise/reasoning statement(s) are outside the controlled-English parser subset."
        )
    advisory = [relation for relation in relations if not relation.proof_usable]
    if advisory:
        warnings.append(
            f"{len(advisory)} semantic relation(s) are advisory only and cannot be used as proof."
        )

    total_statements = len(rows)
    parseable_count = sum(row["parse_status"] == "parseable" for row in rows)
    coverage = round(parseable_count / total_statements * 100, 2) if total_statements else 100.0
    semantic_rows = [
        {
            "id": relation.id,
            "type": relation.relation_type,
            "approved": relation.approved,
            "proof_usable": relation.proof_usable,
            "text": relation.display_text(),
            "description": relation.description,
        }
        for relation in relations
    ]
    return {
        "schema_version": "0.15.0",
        "case_id": str(case.get("id") or "case"),
        "ready_for_verification": not errors,
        "summary": {
            "premise_count": len(premises),
            "reasoning_step_count": len(steps),
            "statement_count": total_statements,
            "parseable_statement_count": parseable_count,
            "untranslatable_statement_count": len(untranslatable),
            "parser_coverage_percent": coverage,
            "question_parseable": question_row["parse_status"] == "parseable",
            "semantic_relation_count": len(relations),
            "proof_usable_semantic_relation_count": sum(r.proof_usable for r in relations),
            "advisory_semantic_relation_count": len(advisory),
            "error_count": len(errors),
            "warning_count": len(warnings),
            "declared_dependency_step_count": sum(bool(item.get("depends_on")) for item in steps),
            "answer_dependencies_declared": bool(answer_depends_on),
            "compound_reasoning_step_count": len(compound_steps),
        },
        "errors": errors,
        "warnings": warnings,
        "question": question_row,
        "statements": rows,
        "semantic_relations": semantic_rows,
    }
