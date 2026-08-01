from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from time import perf_counter
from typing import Any

from .engine import ChainSupportEngine, KnowledgeItem, LogicEngine
from .logic import Atom, Formula, Rule, constants_in_formula, formula_to_text, predicates_in_formula
from .parser import parse_question, parse_statement
from .semantic import SemanticLayer, SemanticRelation, parse_semantic_relations
from .diagnostics import enhance_reasoning_diagnostics


ALLOWED_ANSWERS = {"yes", "no"}
BASE_ERROR_STATUSES = {"contradiction", "ungrounded", "untranslatable"}
CHAIN_OK_STATUSES = {"given", "valid", "approved", "advisory"}
CHAIN_RELATIONS = {
    "source_match",
    "reasoning_dependency",
    "reasoning_conflict",
    "semantic_normalization",
    "semantic_bridge",
}
PROOF_RELATIONS = {"proof_support", "proof_conflict", "semantic_bridge"}
SEMANTIC_EDGE_RELATIONS = {"semantic_normalization", "semantic_bridge", "semantic_related"}


@dataclass
class NodeResult:
    id: str
    kind: str
    order: int
    text: str
    status: str
    proof_status: str
    chain_status: str
    formal: str | None
    parse_error: str | None
    dependencies: list[str]
    proof_dependencies: list[str]
    reasoning_dependencies: list[str]
    reasoning_conflict_dependencies: list[str]
    source_matches: list[str]
    blocking_parent_nodes: list[str]
    upstream_error_nodes: list[str]
    engine_detail: str
    chain_detail: str
    chain_direct_dependents: int = 0
    chain_descendant_count: int = 0
    chain_reaches_final: bool = False
    proof_direct_dependents: int = 0
    proof_descendant_count: int = 0
    proof_reaches_final: bool = False
    chain_impact_level: str = "low"
    logical_impact_level: str = "low"
    impact_level: str = "low"
    logical_final_changes_if_removed: bool | None = None
    logical_affected_if_removed: int | None = None
    chain_final_changes_if_removed: bool | None = None
    chain_affected_if_removed: int | None = None
    alternative_proof_exists: bool | None = None
    smtlib_formula: str | None = None
    consistency_check_result: str = "not_run"
    entailment_check_result: str = "not_run"
    consistency_query_smtlib: str | None = None
    entailment_query_smtlib: str | None = None
    proof_dependencies_raw: list[str] | None = None
    proof_core_minimized: bool = False
    alternative_proof_paths: list[list[str]] | None = None
    strict_chain_breaks_if_removed: bool | None = None
    strict_chain_affected_nodes: int | None = None
    chain_repairable: bool | None = None
    logical_answer_preserved_if_removed: bool | None = None
    # v005 semantic transparency
    raw_formal: str | None = None
    semantic_relation_type: str | None = None
    semantic_proof_usable: bool | None = None
    semantic_normalizations: list[str] | None = None
    semantic_proof_dependencies: list[str] | None = None
    semantic_hints: list[dict[str, Any]] | None = None
    # v012 chain provenance transparency
    chain_dependency_source: str = "inferred"
    declared_reasoning_dependencies: list[str] | None = None
    inferred_reasoning_dependencies: list[str] | None = None
    # v007 partial revalidation transparency
    verification_origin: str = "revalidated"


@dataclass
class EdgeResult:
    source: str
    target: str
    relation: str


def _normalise_answer(value: Any, field_name: str) -> str:
    answer = str(value or "").strip().lower()
    if answer not in ALLOWED_ANSWERS:
        raise ValueError(f"{field_name} must be strictly Yes or No; received: {value!r}")
    return answer


def _claim_for_answer(question_formula: Formula, answer: str) -> Formula:
    if not isinstance(question_formula, Atom):
        raise ValueError("MVP v005 supports atomic Yes/No questions only")
    return question_formula if answer == "yes" else question_formula.complement()


def _get_case_parts(case: dict[str, Any]) -> tuple[list[Any], list[Any], str, str, str]:
    premises = case.get("premises") or case.get("context") or []
    output = case.get("llm_output") or {}
    steps = output.get("reasoning_steps") or case.get("reasoning_steps") or []
    question = str(case.get("question") or "")
    predicted = _normalise_answer(output.get("answer", case.get("predicted_answer")), "LLM answer")
    gold = _normalise_answer(case.get("gold_answer"), "gold_answer")
    if not isinstance(premises, list) or not isinstance(steps, list):
        raise ValueError("premises and reasoning_steps must be lists")
    return premises, steps, question, predicted, gold


def _coerce_item(item: Any, prefix: str, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"id": f"{prefix}{index}", "text": item, "depends_on": []}
    if isinstance(item, dict):
        raw_dependencies = item.get("depends_on") or item.get("dependencies") or []
        if isinstance(raw_dependencies, str):
            raw_dependencies = [piece.strip() for piece in raw_dependencies.split(",") if piece.strip()]
        if not isinstance(raw_dependencies, list):
            raise ValueError(f"depends_on for {item.get('id') or f'{prefix}{index}'} must be a list")
        return {
            "id": str(item.get("id") or f"{prefix}{index}"),
            "text": str(item.get("text") or ""),
            "depends_on": [str(value) for value in raw_dependencies],
        }
    raise ValueError(f"Unsupported item at {prefix}{index}: {item!r}")


def _exact_source_matches(formula: Formula, premise_formulas: dict[str, Formula]) -> list[str]:
    return sorted(node_id for node_id, source_formula in premise_formulas.items() if source_formula == formula)


def _new_node(
    *,
    node_id: str,
    kind: str,
    order: int,
    text: str,
    proof_status: str,
    chain_status: str,
    formal: str | None,
    parse_error: str | None,
    proof_dependencies: list[str] | None = None,
    reasoning_dependencies: list[str] | None = None,
    reasoning_conflict_dependencies: list[str] | None = None,
    source_matches: list[str] | None = None,
    blocking_parent_nodes: list[str] | None = None,
    upstream_error_nodes: list[str] | None = None,
    engine_detail: str = "",
    chain_detail: str = "",
    smtlib_formula: str | None = None,
    consistency_check_result: str = "not_run",
    entailment_check_result: str = "not_run",
    consistency_query_smtlib: str | None = None,
    entailment_query_smtlib: str | None = None,
    proof_dependencies_raw: list[str] | None = None,
    proof_core_minimized: bool = False,
    alternative_proof_paths: list[list[str]] | None = None,
    raw_formal: str | None = None,
    semantic_relation_type: str | None = None,
    semantic_proof_usable: bool | None = None,
    semantic_normalizations: list[str] | None = None,
    semantic_proof_dependencies: list[str] | None = None,
    semantic_hints: list[dict[str, Any]] | None = None,
    chain_dependency_source: str = "inferred",
    declared_reasoning_dependencies: list[str] | None = None,
    inferred_reasoning_dependencies: list[str] | None = None,
    verification_origin: str = "revalidated",
) -> NodeResult:
    proof_dependencies = sorted(set(proof_dependencies or []))
    reasoning_dependencies = sorted(set(reasoning_dependencies or []))
    reasoning_conflict_dependencies = sorted(set(reasoning_conflict_dependencies or []))
    source_matches = sorted(set(source_matches or []))
    blocking_parent_nodes = sorted(set(blocking_parent_nodes or []))
    upstream_error_nodes = sorted(set(upstream_error_nodes or []))
    semantic_normalizations = sorted(set(semantic_normalizations or []))
    semantic_proof_dependencies = sorted(set(semantic_proof_dependencies or []))
    declared_reasoning_dependencies = sorted(set(declared_reasoning_dependencies or []))
    inferred_reasoning_dependencies = sorted(set(inferred_reasoning_dependencies or []))
    dependencies = sorted(
        set(
            source_matches
            + reasoning_dependencies
            + reasoning_conflict_dependencies
            + semantic_normalizations
            + semantic_proof_dependencies
        )
    )
    return NodeResult(
        id=node_id,
        kind=kind,
        order=order,
        text=text,
        status=proof_status,
        proof_status=proof_status,
        chain_status=chain_status,
        formal=formal,
        parse_error=parse_error,
        dependencies=dependencies,
        proof_dependencies=proof_dependencies,
        reasoning_dependencies=reasoning_dependencies,
        reasoning_conflict_dependencies=reasoning_conflict_dependencies,
        source_matches=source_matches,
        blocking_parent_nodes=blocking_parent_nodes,
        upstream_error_nodes=upstream_error_nodes,
        engine_detail=engine_detail,
        chain_detail=chain_detail,
        smtlib_formula=smtlib_formula,
        consistency_check_result=consistency_check_result,
        entailment_check_result=entailment_check_result,
        consistency_query_smtlib=consistency_query_smtlib,
        entailment_query_smtlib=entailment_query_smtlib,
        proof_dependencies_raw=sorted(set(proof_dependencies_raw or [])),
        proof_core_minimized=proof_core_minimized,
        alternative_proof_paths=alternative_proof_paths or [],
        raw_formal=raw_formal,
        semantic_relation_type=semantic_relation_type,
        semantic_proof_usable=semantic_proof_usable,
        semantic_normalizations=semantic_normalizations,
        semantic_proof_dependencies=semantic_proof_dependencies,
        semantic_hints=semantic_hints or [],
        chain_dependency_source=chain_dependency_source,
        declared_reasoning_dependencies=declared_reasoning_dependencies,
        inferred_reasoning_dependencies=inferred_reasoning_dependencies,
        verification_origin=verification_origin,
    )



def _node_from_result_dict(data: dict[str, Any], *, verification_origin: str = "reused_prefix") -> NodeResult:
    """Rehydrate a previous NodeResult while ignoring future/derived fields."""
    allowed = {field.name for field in fields(NodeResult)}
    payload = {key: value for key, value in data.items() if key in allowed}
    payload["verification_origin"] = verification_origin
    return NodeResult(**payload)


def _blocking_information(dependency_ids: list[str], node_lookup: dict[str, NodeResult]) -> tuple[list[str], list[str]]:
    blocking_parents: list[str] = []
    root_errors: set[str] = set()
    for dependency_id in dependency_ids:
        dependency = node_lookup.get(dependency_id)
        if dependency is None or dependency.chain_status in CHAIN_OK_STATUSES:
            continue
        blocking_parents.append(dependency_id)
        if dependency.chain_status == "blocked_by_upstream_error":
            root_errors.update(dependency.upstream_error_nodes or [dependency_id])
        else:
            root_errors.add(dependency_id)
    return sorted(set(blocking_parents)), sorted(root_errors)


def _chain_classification(
    proof_status: str,
    dependency_ids: list[str],
    node_lookup: dict[str, NodeResult],
) -> tuple[str, list[str], list[str], str]:
    blocking_parents, root_errors = _blocking_information(dependency_ids, node_lookup)
    if blocking_parents and proof_status != "contradiction":
        return (
            "blocked_by_upstream_error",
            blocking_parents,
            root_errors,
            f"LLM path depends on invalid upstream node(s): {', '.join(blocking_parents)}",
        )
    return proof_status, blocking_parents, root_errors, f"LLM-chain local status: {proof_status}"


def _relation_node(relation: SemanticRelation, order: int, engine: LogicEngine, disabled: bool) -> NodeResult:
    if disabled:
        return _new_node(
            node_id=relation.id,
            kind="semantic",
            order=order,
            text=relation.display_text(),
            proof_status="disabled",
            chain_status="disabled",
            formal=None,
            parse_error=None,
            engine_detail="Semantic relation disabled for counterfactual removal",
            chain_detail="Semantic relation disabled for counterfactual removal",
            semantic_relation_type=relation.relation_type,
            semantic_proof_usable=relation.proof_usable,
        )
    bridge = relation.bridge_rule()
    status = "approved" if relation.proof_usable else "advisory"
    detail = (
        "Approved semantic relation may participate in preprocessing/proof"
        if relation.proof_usable
        else "Advisory semantic relation is visible but never used as logical proof"
    )
    return _new_node(
        node_id=relation.id,
        kind="semantic",
        order=order,
        text=relation.display_text(),
        proof_status=status,
        chain_status=status,
        formal=formula_to_text(bridge) if bridge else relation.display_text(),
        raw_formal=formula_to_text(bridge) if bridge else relation.display_text(),
        parse_error=None,
        engine_detail=detail,
        chain_detail=detail,
        smtlib_formula=engine.formula_smtlib(bridge) if bridge else None,
        semantic_relation_type=relation.relation_type,
        semantic_proof_usable=relation.proof_usable,
    )


def _semantic_dependencies_from_selected_nodes(
    selected_node_ids: list[str],
    normalization_by_node: dict[str, list[str]],
) -> list[str]:
    result: set[str] = set()
    for node_id in selected_node_ids:
        result.update(normalization_by_node.get(node_id, []))
    return sorted(result)




def _record_engine_stats(execution_stats: dict[str, Any], check: Any) -> None:
    execution_stats["solver_instances"] = int(execution_stats.get("solver_instances") or 0) + int(getattr(check, "solver_instances", 0) or 0)
    execution_stats["backend_solver_checks"] = int(execution_stats.get("backend_solver_checks") or 0) + int(getattr(check, "solver_checks", 0) or 0)
    execution_stats["knowledge_assertions"] = int(execution_stats.get("knowledge_assertions") or 0) + int(getattr(check, "knowledge_assertions", 0) or 0)
    execution_stats["core_minimization_checks"] = int(execution_stats.get("core_minimization_checks") or 0) + int(getattr(check, "core_minimization_checks", 0) or 0)

def _verify_core(
    case: dict[str, Any],
    *,
    disabled_nodes: set[str] | None = None,
    prefer_z3: bool = True,
    previous_result: dict[str, Any] | None = None,
    reuse_reasoning_before: int = 0,
    reuse_reasoning_ids: set[str] | None = None,
    reuse_final: bool = False,
    execution_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    disabled_nodes = disabled_nodes or set()
    execution_stats = execution_stats if execution_stats is not None else {}
    execution_stats.setdefault("solver_checks", 0)
    execution_stats.setdefault("reused_reasoning_nodes", [])
    execution_stats.setdefault("revalidated_reasoning_nodes", [])
    execution_stats.setdefault("reused_final", False)
    execution_stats.setdefault("solver_instances", 0)
    execution_stats.setdefault("backend_solver_checks", 0)
    execution_stats.setdefault("knowledge_assertions", 0)
    execution_stats.setdefault("core_minimization_checks", 0)
    reuse_reasoning_ids = set(reuse_reasoning_ids or [])
    previous_node_lookup = {
        node.get("id"): node
        for node in (previous_result or {}).get("nodes", [])
        if isinstance(node, dict) and node.get("id")
    }
    premises_raw, steps_raw, question, predicted, gold = _get_case_parts(case)
    policy = case.get("verification_policy") or {}
    require_declared_reasoning = bool(policy.get("require_declared_reasoning_dependencies"))
    require_declared_answer = bool(policy.get("require_declared_answer_dependencies"))
    premises = [_coerce_item(item, "p", idx) for idx, item in enumerate(premises_raw, 1)]
    steps = [_coerce_item(item, "s", idx) for idx, item in enumerate(steps_raw, 1)]

    semantic_relations = parse_semantic_relations(case.get("semantic_relations"))
    semantic_layer = SemanticLayer(semantic_relations, disabled_ids=disabled_nodes)
    engine = LogicEngine(prefer_z3=prefer_z3)

    raw_question_result = parse_question(question)
    if raw_question_result.formula is None:
        raise ValueError(f"Question could not be translated: {raw_question_result.error}")
    question_formula, question_normalizations = semantic_layer.normalize_formula(raw_question_result.formula)
    final_claim = _claim_for_answer(question_formula, predicted)

    nodes: list[NodeResult] = []
    node_lookup: dict[str, NodeResult] = {}
    proof_knowledge: list[KnowledgeItem] = []
    chain_shadow_knowledge: list[KnowledgeItem] = []
    node_kinds: dict[str, str] = {}
    node_orders: dict[str, int] = {}
    premise_formulas: dict[str, Formula] = {}
    prior_predicates: dict[str, set[tuple[str, int]]] = {}
    normalization_by_node: dict[str, list[str]] = {}

    for index, relation in enumerate(semantic_relations):
        node_kinds[relation.id] = "semantic"
        node_orders[relation.id] = -len(semantic_relations) + index
        node = _relation_node(relation, index, engine, relation.id in disabled_nodes)
        nodes.append(node)
        node_lookup[relation.id] = node

    for relation, bridge in semantic_layer.bridge_rules():
        item = KnowledgeItem(relation.id, bridge)
        proof_knowledge.append(item)
        engine.add_knowledge(item)
        chain_shadow_knowledge.append(item)
        prior_predicates[relation.id] = predicates_in_formula(bridge)

    for order, premise in enumerate(premises):
        node_id = premise["id"]
        node_kinds[node_id] = "premise"
        node_orders[node_id] = order
        raw_result = parse_statement(premise["text"])
        if node_id in disabled_nodes:
            node = _new_node(
                node_id=node_id,
                kind="premise",
                order=order,
                text=premise["text"],
                proof_status="disabled",
                chain_status="disabled",
                formal=None,
                parse_error=None,
                engine_detail="Counterfactual removal",
                chain_detail="Counterfactual removal",
            )
        elif raw_result.formula is None:
            node = _new_node(
                node_id=node_id,
                kind="premise",
                order=order,
                text=premise["text"],
                proof_status="untranslatable",
                chain_status="untranslatable",
                formal=None,
                parse_error=raw_result.error,
                engine_detail=raw_result.error or "Translation failed",
                chain_detail=raw_result.error or "Translation failed",
            )
        else:
            normalized, normalizations = semantic_layer.normalize_formula(raw_result.formula)
            normalization_by_node[node_id] = normalizations
            premise_formulas[node_id] = normalized
            item = KnowledgeItem(node_id, normalized)
            proof_knowledge.append(item)
            engine.add_knowledge(item)
            chain_shadow_knowledge.append(item)
            prior_predicates[node_id] = predicates_in_formula(normalized)
            node = _new_node(
                node_id=node_id,
                kind="premise",
                order=order,
                text=premise["text"],
                proof_status="given",
                chain_status="given",
                formal=formula_to_text(normalized),
                raw_formal=formula_to_text(raw_result.formula),
                semantic_normalizations=normalizations,
                parse_error=None,
                engine_detail="Source premise",
                chain_detail="Source premise",
                smtlib_formula=engine.formula_smtlib(normalized),
            )
        nodes.append(node)
        node_lookup[node_id] = node

    for order, step in enumerate(steps, start=len(premises)):
        node_id = step["id"]
        node_kinds[node_id] = "reasoning"
        node_orders[node_id] = order
        raw_parsed = parse_statement(step["text"])
        step_index = order - len(premises)
        should_reuse = (step_index < reuse_reasoning_before or node_id in reuse_reasoning_ids) and node_id in previous_node_lookup
        if should_reuse:
            previous_node = previous_node_lookup[node_id]
            # Rebuild the trusted/shadow knowledge from the unchanged text, but skip
            # the expensive proof and chain checks for this prefix node.
            if raw_parsed.formula is not None:
                parsed_formula, normalizations = semantic_layer.normalize_formula(raw_parsed.formula)
                normalization_by_node[node_id] = normalizations
                if previous_node.get("proof_status") == "valid":
                    reused_item = KnowledgeItem(node_id, parsed_formula)
                    proof_knowledge.append(reused_item)
                    engine.add_knowledge(reused_item)
                chain_shadow_knowledge.append(KnowledgeItem(node_id, parsed_formula))
                prior_predicates[node_id] = predicates_in_formula(parsed_formula)
            origin = "reused_prefix" if step_index < reuse_reasoning_before else "reused_unaffected"
            node = _node_from_result_dict(previous_node, verification_origin=origin)
            nodes.append(node)
            node_lookup[node_id] = node
            execution_stats["reused_reasoning_nodes"].append(node_id)
            continue
        execution_stats["revalidated_reasoning_nodes"].append(node_id)
        if node_id in disabled_nodes:
            node = _new_node(
                node_id=node_id,
                kind="reasoning",
                order=order,
                text=step["text"],
                proof_status="disabled",
                chain_status="disabled",
                formal=None,
                parse_error=None,
                engine_detail="Counterfactual removal",
                chain_detail="Counterfactual removal",
            )
            nodes.append(node)
            node_lookup[node_id] = node
            continue
        if raw_parsed.formula is None:
            node = _new_node(
                node_id=node_id,
                kind="reasoning",
                order=order,
                text=step["text"],
                proof_status="untranslatable",
                chain_status="untranslatable",
                formal=None,
                parse_error=raw_parsed.error,
                engine_detail=raw_parsed.error or "Translation failed",
                chain_detail=raw_parsed.error or "Translation failed",
            )
            nodes.append(node)
            node_lookup[node_id] = node
            continue

        parsed_formula, normalizations = semantic_layer.normalize_formula(raw_parsed.formula)
        normalization_by_node[node_id] = normalizations
        semantic_hints = semantic_layer.related_hints(parsed_formula, prior_predicates)
        execution_stats["solver_checks"] += 2
        proof_check = engine.check(proof_knowledge, parsed_formula)
        _record_engine_stats(execution_stats, proof_check)
        support_target = (
            parsed_formula.complement()
            if proof_check.status == "contradiction" and isinstance(parsed_formula, Atom)
            else parsed_formula
        )
        support_engine = ChainSupportEngine(node_kinds, node_orders)
        chain_support = support_engine.support(chain_shadow_knowledge, support_target)
        all_proof_paths = support_engine.support_paths(proof_knowledge, support_target, max_paths=8)
        selected_proof_set = set(proof_check.dependencies)
        alternative_proof_paths = [path for path in all_proof_paths if set(path) != selected_proof_set][:3]
        source_matches = _exact_source_matches(parsed_formula, premise_formulas)
        inferred_chain_dependencies = [dependency for dependency in chain_support.dependencies if dependency not in source_matches]
        declared_chain_dependencies = sorted(set(step.get("depends_on") or []))
        if declared_chain_dependencies:
            unknown_declared = [dependency for dependency in declared_chain_dependencies if dependency not in node_lookup]
            if unknown_declared:
                raise ValueError(
                    f"Reasoning node {node_id} declares unknown or future depends_on node(s): {', '.join(unknown_declared)}"
                )
            chain_dependency_source = "declared"
            chain_dependencies = [dependency for dependency in declared_chain_dependencies if dependency not in source_matches]
            chain_dependency_ids = declared_chain_dependencies
        elif require_declared_reasoning:
            chain_dependency_source = "declared_missing"
            chain_dependencies = []
            chain_dependency_ids = []
        else:
            chain_dependency_source = "inferred"
            chain_dependencies = inferred_chain_dependencies
            chain_dependency_ids = sorted(set(source_matches + chain_dependencies))
        semantic_proof_dependencies = _semantic_dependencies_from_selected_nodes(
            proof_check.dependencies, normalization_by_node
        )
        semantic_proof_dependencies.extend(normalizations)
        semantic_proof_dependencies = sorted(set(semantic_proof_dependencies))

        if proof_check.status == "contradiction":
            reasoning_dependencies: list[str] = []
            reasoning_conflict_dependencies = chain_dependencies
        else:
            reasoning_dependencies = chain_dependencies
            reasoning_conflict_dependencies = []

        dependency_ids = sorted(
            set(chain_dependency_ids + normalizations + semantic_proof_dependencies)
        )
        chain_status, blocking_parents, root_errors, chain_detail = _chain_classification(
            proof_check.status, dependency_ids, node_lookup
        )
        if require_declared_reasoning and not declared_chain_dependencies and proof_check.status == "valid":
            chain_status = "insufficient_declared_support"
            chain_detail = "Reasoning step declared no direct parent IDs under strict authored-chain policy; inferred support is advisory only."
        provenance_note = (
            f"Declared chain dependencies used: {', '.join(declared_chain_dependencies)}; "
            f"inferred alternative: {', '.join(inferred_chain_dependencies) or '-'}"
            if declared_chain_dependencies
            else "Inferred CoT-aware chain dependencies used"
        )
        hint_note = ""
        if semantic_hints:
            hint_note = "; advisory semantic relation found but not used as proof"
        node = _new_node(
            node_id=node_id,
            kind="reasoning",
            order=order,
            text=step["text"],
            proof_status=proof_check.status,
            chain_status=chain_status,
            formal=formula_to_text(parsed_formula),
            raw_formal=formula_to_text(raw_parsed.formula),
            semantic_normalizations=normalizations,
            semantic_proof_dependencies=semantic_proof_dependencies,
            semantic_hints=semantic_hints,
            parse_error=None,
            proof_dependencies=proof_check.dependencies,
            reasoning_dependencies=reasoning_dependencies,
            reasoning_conflict_dependencies=reasoning_conflict_dependencies,
            source_matches=source_matches,
            blocking_parent_nodes=blocking_parents,
            upstream_error_nodes=root_errors,
            engine_detail=proof_check.detail + hint_note,
            chain_detail=f"{provenance_note}; {chain_support.detail}; {chain_detail}{hint_note}",
            chain_dependency_source=chain_dependency_source,
            declared_reasoning_dependencies=declared_chain_dependencies,
            inferred_reasoning_dependencies=inferred_chain_dependencies,
            smtlib_formula=proof_check.target_smtlib,
            consistency_check_result=proof_check.consistency_result,
            entailment_check_result=proof_check.entailment_result,
            consistency_query_smtlib=proof_check.consistency_query_smtlib,
            entailment_query_smtlib=proof_check.entailment_query_smtlib,
            proof_dependencies_raw=proof_check.raw_dependencies,
            proof_core_minimized=proof_check.core_minimized,
            alternative_proof_paths=alternative_proof_paths,
        )
        nodes.append(node)
        node_lookup[node_id] = node

        if proof_check.status == "valid":
            valid_item = KnowledgeItem(node_id, parsed_formula)
            proof_knowledge.append(valid_item)
            engine.add_knowledge(valid_item)
        chain_shadow_knowledge.append(KnowledgeItem(node_id, parsed_formula))
        prior_predicates[node_id] = predicates_in_formula(parsed_formula)

    final_id = "final"
    final_order = len(premises) + len(steps)
    node_kinds[final_id] = "answer"
    node_orders[final_id] = final_order
    if reuse_final and final_id in previous_node_lookup and final_id not in disabled_nodes:
        final_node = _node_from_result_dict(previous_node_lookup[final_id], verification_origin="reused_unaffected")
        final_node.text = f"Answer: {predicted.title()}"
        execution_stats["reused_final"] = True
    elif final_id in disabled_nodes:
        final_node = _new_node(
            node_id=final_id,
            kind="answer",
            order=final_order,
            text=f"Answer: {predicted.title()}",
            proof_status="disabled",
            chain_status="disabled",
            formal=formula_to_text(final_claim),
            raw_formal=formula_to_text(raw_question_result.formula),
            semantic_normalizations=question_normalizations,
            parse_error=None,
            engine_detail="Counterfactual removal",
            chain_detail="Counterfactual removal",
        )
    else:
        semantic_hints = semantic_layer.related_hints(final_claim, prior_predicates)
        execution_stats["solver_checks"] += 2
        proof_check = engine.check(proof_knowledge, final_claim)
        _record_engine_stats(execution_stats, proof_check)
        support_target = (
            final_claim.complement()
            if proof_check.status == "contradiction" and isinstance(final_claim, Atom)
            else final_claim
        )
        support_engine = ChainSupportEngine(node_kinds, node_orders)
        chain_support = support_engine.support(chain_shadow_knowledge, support_target)
        all_proof_paths = support_engine.support_paths(proof_knowledge, support_target, max_paths=8)
        selected_proof_set = set(proof_check.dependencies)
        alternative_proof_paths = [path for path in all_proof_paths if set(path) != selected_proof_set][:3]
        inferred_chain_dependencies = chain_support.dependencies
        output_payload = case.get("llm_output") or {}
        declared_chain_dependencies = output_payload.get("answer_depends_on") or case.get("answer_depends_on") or []
        if isinstance(declared_chain_dependencies, str):
            declared_chain_dependencies = [piece.strip() for piece in declared_chain_dependencies.split(",") if piece.strip()]
        if not isinstance(declared_chain_dependencies, list):
            raise ValueError("answer_depends_on must be a list")
        declared_chain_dependencies = sorted(set(str(value) for value in declared_chain_dependencies))
        if declared_chain_dependencies:
            unknown_declared = [dependency for dependency in declared_chain_dependencies if dependency not in node_lookup]
            if unknown_declared:
                raise ValueError(
                    f"Final answer declares unknown depends_on node(s): {', '.join(unknown_declared)}"
                )
            chain_dependency_source = "declared"
            chain_dependencies = declared_chain_dependencies
        elif require_declared_answer and steps:
            chain_dependency_source = "declared_missing"
            chain_dependencies = []
        else:
            chain_dependency_source = "inferred"
            chain_dependencies = inferred_chain_dependencies
        semantic_proof_dependencies = _semantic_dependencies_from_selected_nodes(
            proof_check.dependencies, normalization_by_node
        )
        semantic_proof_dependencies.extend(question_normalizations)
        semantic_proof_dependencies = sorted(set(semantic_proof_dependencies))
        if proof_check.status == "contradiction":
            reasoning_dependencies = []
            reasoning_conflict_dependencies = chain_dependencies
        else:
            reasoning_dependencies = chain_dependencies
            reasoning_conflict_dependencies = []
        dependency_ids = sorted(set(chain_dependencies + question_normalizations + semantic_proof_dependencies))
        chain_status, blocking_parents, root_errors, chain_detail = _chain_classification(
            proof_check.status, dependency_ids, node_lookup
        )
        if require_declared_answer and steps and not declared_chain_dependencies and proof_check.status == "valid":
            chain_status = "insufficient_declared_support"
            chain_detail = "Final answer declared no direct parent IDs under strict authored-chain policy; inferred support is advisory only."
        provenance_note = (
            f"Declared final dependencies used: {', '.join(declared_chain_dependencies)}; "
            f"inferred alternative: {', '.join(inferred_chain_dependencies) or '-'}"
            if declared_chain_dependencies
            else "Inferred CoT-aware final dependencies used"
        )
        hint_note = "; advisory semantic relation found but not used as proof" if semantic_hints else ""
        final_node = _new_node(
            node_id=final_id,
            kind="answer",
            order=final_order,
            text=f"Answer: {predicted.title()}",
            proof_status=proof_check.status,
            chain_status=chain_status,
            formal=formula_to_text(final_claim),
            raw_formal=formula_to_text(raw_question_result.formula),
            semantic_normalizations=question_normalizations,
            semantic_proof_dependencies=semantic_proof_dependencies,
            semantic_hints=semantic_hints,
            parse_error=None,
            proof_dependencies=proof_check.dependencies,
            reasoning_dependencies=reasoning_dependencies,
            reasoning_conflict_dependencies=reasoning_conflict_dependencies,
            source_matches=[],
            blocking_parent_nodes=blocking_parents,
            upstream_error_nodes=root_errors,
            engine_detail=proof_check.detail + hint_note,
            chain_detail=f"{provenance_note}; {chain_support.detail}; {chain_detail}{hint_note}",
            chain_dependency_source=chain_dependency_source,
            declared_reasoning_dependencies=declared_chain_dependencies,
            inferred_reasoning_dependencies=inferred_chain_dependencies,
            smtlib_formula=proof_check.target_smtlib,
            consistency_check_result=proof_check.consistency_result,
            entailment_check_result=proof_check.entailment_result,
            consistency_query_smtlib=proof_check.consistency_query_smtlib,
            entailment_query_smtlib=proof_check.entailment_query_smtlib,
            proof_dependencies_raw=proof_check.raw_dependencies,
            proof_core_minimized=proof_check.core_minimized,
            alternative_proof_paths=alternative_proof_paths,
        )
    nodes.append(final_node)
    node_lookup[final_id] = final_node

    edges: list[EdgeResult] = []
    node_ids = {node.id for node in nodes}
    for node in nodes:
        for source in node.source_matches:
            if source in node_ids and source != node.id:
                edges.append(EdgeResult(source, node.id, "source_match"))
        for source in node.reasoning_dependencies:
            if source in node_ids and source != node.id:
                relation = "semantic_bridge" if node_lookup[source].kind == "semantic" else "reasoning_dependency"
                edges.append(EdgeResult(source, node.id, relation))
        for source in node.reasoning_conflict_dependencies:
            if source in node_ids and source != node.id:
                edges.append(EdgeResult(source, node.id, "reasoning_conflict"))
        proof_relation = "proof_conflict" if node.proof_status == "contradiction" else "proof_support"
        for source in node.proof_dependencies:
            if source in node_ids and source != node.id:
                relation = "semantic_bridge" if node_lookup[source].kind == "semantic" else proof_relation
                edges.append(EdgeResult(source, node.id, relation))
        for source in node.semantic_normalizations or []:
            if source in node_ids and source != node.id:
                edges.append(EdgeResult(source, node.id, "semantic_normalization"))
        for source in node.semantic_proof_dependencies or []:
            if source in node_ids and source != node.id:
                edges.append(EdgeResult(source, node.id, "semantic_normalization"))
        for hint in node.semantic_hints or []:
            relation_id = hint.get("relation_id")
            if relation_id in node_ids and relation_id != node.id:
                edges.append(EdgeResult(relation_id, node.id, "semantic_related"))
        for source in node.blocking_parent_nodes:
            if source in node_ids and source != node.id:
                edges.append(EdgeResult(source, node.id, "error_propagation"))

    unique_edges = {(edge.source, edge.target, edge.relation): edge for edge in edges}
    edges = [unique_edges[key] for key in sorted(unique_edges)]

    reasoning_nodes = [node for node in nodes if node.kind == "reasoning"]
    semantic_nodes = [node for node in nodes if node.kind == "semantic"]
    return {
        "schema_version": "0.15.0",
        "case_id": str(case.get("id") or "case"),
        "question": question,
        "predicted_answer": predicted.title(),
        "gold_answer": gold.title(),
        "answer_correct": predicted == gold,
        "engine": engine.name,
        "semantic_policy": {
            "same_as": "Approved relations canonicalize predicates before SMT/Horn verification.",
            "implies": "Approved relations become explicit bridge rules usable by Z3 and Horn.",
            "related_to": "Advisory only; shown as a hint and never used to prove a claim.",
        },
        "nodes": [asdict(node) for node in nodes],
        "edges": [asdict(edge) for edge in edges],
        "summary": {
            "final_status": final_node.proof_status,
            "final_proof_status": final_node.proof_status,
            "final_chain_status": final_node.chain_status,
            "all_reasoning_valid": all(node.proof_status == "valid" for node in reasoning_nodes),
            "all_reasoning_proof_valid": all(node.proof_status == "valid" for node in reasoning_nodes),
            "all_reasoning_chain_valid": all(node.chain_status == "valid" for node in reasoning_nodes),
            "invalid_reasoning_count": sum(node.proof_status in BASE_ERROR_STATUSES for node in reasoning_nodes),
            "blocked_reasoning_count": sum(node.chain_status == "blocked_by_upstream_error" for node in reasoning_nodes),
            "chain_error_count": sum(node.chain_status not in CHAIN_OK_STATUSES for node in reasoning_nodes),
            "root_error_nodes": [node.id for node in reasoning_nodes if node.chain_status in BASE_ERROR_STATUSES],
            "valid_answer_but_invalid_reasoning": (
                final_node.proof_status == "valid" and any(node.chain_status != "valid" for node in reasoning_nodes)
            ),
            "source_match_edge_count": sum(edge.relation == "source_match" for edge in edges),
            "reasoning_dependency_edge_count": sum(edge.relation == "reasoning_dependency" for edge in edges),
            "reasoning_conflict_edge_count": sum(edge.relation == "reasoning_conflict" for edge in edges),
            "error_propagation_edge_count": sum(edge.relation == "error_propagation" for edge in edges),
            "proof_edge_count": sum(edge.relation in PROOF_RELATIONS for edge in edges),
            "semantic_relation_count": len(semantic_nodes),
            "proof_usable_semantic_relations": sum(node.semantic_proof_usable is True for node in semantic_nodes),
            "advisory_semantic_relations": sum(node.semantic_relation_type == "related_to" for node in semantic_nodes),
            "semantic_hint_count": sum(len(node.semantic_hints or []) for node in nodes),
            "semantic_edge_count": sum(edge.relation in SEMANTIC_EDGE_RELATIONS for edge in edges),
        },
    }


def _edges_for_relations(edges: list[dict[str, Any]], relations: set[str]) -> list[dict[str, Any]]:
    return [edge for edge in edges if edge["relation"] in relations]


def _descendants(edges: list[dict[str, Any]], start: str) -> set[str]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["source"], set()).add(edge["target"])
    seen: set[str] = set()
    stack = list(adjacency.get(start, set()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, set()))
    return seen


def _decorate_view_metrics(result: dict[str, Any], *, relations: set[str], prefix: str) -> None:
    edges = _edges_for_relations(result["edges"], relations)
    direct: dict[str, set[str]] = {}
    for edge in edges:
        direct.setdefault(edge["source"], set()).add(edge["target"])
    for node in result["nodes"]:
        descendants = _descendants(edges, node["id"])
        node[f"{prefix}_direct_dependents"] = len(direct.get(node["id"], set()))
        node[f"{prefix}_descendant_count"] = len(descendants)
        node[f"{prefix}_reaches_final"] = "final" in descendants or node["id"] == "final"


def _impact_rank(level: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3, "final": 4}.get(level, 0)


def _max_impact(*levels: str) -> str:
    return max(levels, key=_impact_rank)


def _chain_impact(node: dict[str, Any]) -> str:
    if node["kind"] == "answer":
        return "final"
    if node.get("chain_reaches_final"):
        return "critical"
    descendants = int(node.get("chain_descendant_count") or 0)
    if descendants >= 3:
        return "high"
    if descendants > 0:
        return "medium"
    return "low"


def verify_case(
    case: dict[str, Any],
    *,
    prefer_z3: bool = True,
    compute_counterfactuals: bool = True,
) -> dict[str, Any]:
    started = perf_counter()
    execution_stats: dict[str, Any] = {}
    result = _verify_core(case, prefer_z3=prefer_z3, execution_stats=execution_stats)
    _decorate_view_metrics(result, relations=CHAIN_RELATIONS, prefix="chain")
    _decorate_view_metrics(result, relations=PROOF_RELATIONS, prefix="proof")

    for node in result["nodes"]:
        node["chain_impact_level"] = _chain_impact(node)

    candidate_nodes = [node for node in result["nodes"] if node["kind"] != "answer"]
    if compute_counterfactuals and len(candidate_nodes) <= 20:
        baseline_proof = {node["id"]: node["proof_status"] for node in result["nodes"]}
        baseline_chain = {node["id"]: node["chain_status"] for node in result["nodes"]}
        for node in result["nodes"]:
            if node["kind"] == "answer":
                node["logical_impact_level"] = "final"
                node["impact_level"] = "final"
                continue
            counterfactual = _verify_core(deepcopy(case), disabled_nodes={node["id"]}, prefer_z3=prefer_z3)
            counter_proof = {item["id"]: item["proof_status"] for item in counterfactual["nodes"]}
            counter_chain = {item["id"]: item["chain_status"] for item in counterfactual["nodes"]}
            proof_changed = sum(
                baseline_proof.get(node_id) != status
                for node_id, status in counter_proof.items()
                if node_id != node["id"]
            )
            chain_changed = sum(
                baseline_chain.get(node_id) != status
                for node_id, status in counter_chain.items()
                if node_id != node["id"]
            )
            final_proof_changed = counter_proof.get("final") != baseline_proof.get("final")
            final_chain_changed = counter_chain.get("final") != baseline_chain.get("final")
            node["logical_affected_if_removed"] = proof_changed
            node["logical_final_changes_if_removed"] = final_proof_changed
            node["chain_affected_if_removed"] = chain_changed
            node["chain_final_changes_if_removed"] = final_chain_changed
            node["alternative_proof_exists"] = bool(node.get("chain_reaches_final") and not final_proof_changed)
            node["strict_chain_breaks_if_removed"] = bool(node.get("chain_reaches_final"))
            node["strict_chain_affected_nodes"] = int(node.get("chain_descendant_count") or 0)
            node["chain_repairable"] = bool(node["strict_chain_breaks_if_removed"] and not final_chain_changed)
            node["logical_answer_preserved_if_removed"] = not final_proof_changed
            if final_proof_changed:
                logical_level = "critical"
            elif proof_changed >= 3:
                logical_level = "high"
            elif proof_changed > 0:
                logical_level = "medium"
            else:
                logical_level = "low"
            node["logical_impact_level"] = logical_level
            node["impact_level"] = _max_impact(node["chain_impact_level"], logical_level)
    else:
        for node in result["nodes"]:
            if node["kind"] == "answer":
                node["logical_impact_level"] = "final"
                node["impact_level"] = "final"
                continue
            node["logical_impact_level"] = "not_computed"
            node["impact_level"] = node["chain_impact_level"]
            node["strict_chain_breaks_if_removed"] = bool(node.get("chain_reaches_final"))
            node["strict_chain_affected_nodes"] = int(node.get("chain_descendant_count") or 0)
            node["chain_repairable"] = None
            node["logical_answer_preserved_if_removed"] = None

    result["summary"]["critical_chain_nodes"] = sum(
        node["chain_impact_level"] == "critical"
        for node in result["nodes"]
        if node["kind"] != "answer"
    )
    result["summary"]["logically_necessary_nodes"] = sum(
        node.get("logical_final_changes_if_removed") is True
        for node in result["nodes"]
        if node["kind"] != "answer"
    )
    result["summary"]["alternative_proof_nodes"] = sum(
        node.get("alternative_proof_exists") is True
        for node in result["nodes"]
        if node["kind"] != "answer"
    )
    result["summary"]["nodes_with_displayed_alternative_paths"] = sum(
        bool(node.get("alternative_proof_paths")) for node in result["nodes"]
    )
    result["summary"]["minimized_z3_cores"] = sum(
        node.get("proof_core_minimized") is True for node in result["nodes"]
    )
    total_runtime_ms = (perf_counter() - started) * 1000
    result = enhance_reasoning_diagnostics(case, result)
    result["verification"] = {
        "runtime_ms": round(total_runtime_ms, 3),
        "compute_counterfactuals": compute_counterfactuals,
        "solver_stats": {
            "logical_claim_checks_estimate": int(execution_stats.get("solver_checks") or 0),
            "backend_solver_instances": int(execution_stats.get("solver_instances") or 0),
            "backend_solver_checks": int(execution_stats.get("backend_solver_checks") or 0),
            "knowledge_assertions": int(execution_stats.get("knowledge_assertions") or 0),
            "core_minimization_checks": int(execution_stats.get("core_minimization_checks") or 0),
            "strategy": (
                "Persistent per-case Z3 session: trusted knowledge is asserted once; K∧F and K∧¬F use assumption labels on shared tracked/display solvers."
                if result.get("engine") == "z3"
                else "Finite Horn fallback; no SMT solver instances were created."
            ),
        },
        "note": "Solver statistics cover the baseline verification pass; counterfactual removal reruns are excluded from these counters.",
    }
    return result


def _normalised_items_for_diff(items: Any, prefix: str) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []
    return [_coerce_item(item, prefix, index) for index, item in enumerate(items, 1)]


def _normalised_formula_for_text(case: dict[str, Any], text: str) -> Formula | None:
    parsed = parse_statement(text)
    if parsed.formula is None:
        return None
    layer = SemanticLayer(parse_semantic_relations(case.get("semantic_relations")))
    formula, _ = layer.normalize_formula(parsed.formula)
    return formula


def _predicate_flow_graph(case: dict[str, Any]) -> dict[tuple[str, int], set[tuple[str, int]]]:
    """Conservative predicate reachability induced by premises and semantic bridges."""
    graph: dict[tuple[str, int], set[tuple[str, int]]] = {}
    layer = SemanticLayer(parse_semantic_relations(case.get("semantic_relations")))
    formulas: list[Formula] = []
    for raw in case.get("premises") or case.get("context") or []:
        item = _coerce_item(raw, "p", len(formulas) + 1)
        parsed = parse_statement(item["text"])
        if parsed.formula is not None:
            normalized, _ = layer.normalize_formula(parsed.formula)
            formulas.append(normalized)
    formulas.extend(bridge for _, bridge in layer.bridge_rules())
    for formula in formulas:
        if not isinstance(formula, Rule):
            continue
        consequent = (formula.consequent.predicate, len(formula.consequent.args))
        for antecedent in formula.antecedents:
            source = (antecedent.predicate, len(antecedent.args))
            graph.setdefault(source, set()).add(consequent)
    return graph


def _reachable_predicates(
    seeds: set[tuple[str, int]],
    flow: dict[tuple[str, int], set[tuple[str, int]]],
) -> set[tuple[str, int]]:
    reached = set(seeds)
    frontier = list(seeds)
    while frontier:
        current = frontier.pop()
        for nxt in flow.get(current, set()):
            if nxt not in reached:
                reached.add(nxt)
                frontier.append(nxt)
    return reached


def _result_descendants(previous_result: dict[str, Any], starts: set[str]) -> set[str]:
    relevant = {
        "source_match", "reasoning_dependency", "reasoning_conflict", "error_propagation",
        "proof_support", "proof_conflict", "semantic_bridge", "semantic_normalization",
    }
    adjacency: dict[str, set[str]] = {}
    for edge in previous_result.get("edges") or []:
        if edge.get("relation") in relevant:
            adjacency.setdefault(str(edge.get("source")), set()).add(str(edge.get("target")))
    reached: set[str] = set()
    frontier = list(starts)
    while frontier:
        current = frontier.pop()
        for nxt in adjacency.get(current, set()):
            if nxt not in reached and nxt not in starts:
                reached.add(nxt)
                frontier.append(nxt)
    return reached


def plan_incremental_revalidation(
    previous_case: dict[str, Any],
    updated_case: dict[str, Any],
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Plan safe graph-local revalidation with conservative fallbacks.

    v008 starts from the previous dependency graph, expands through predicate
    reachability induced by Horn rules, and revalidates only affected later
    reasoning nodes. If the graph is unavailable, IDs changed, or a changed
    sentence cannot be translated, it falls back to the v007 suffix policy.
    """
    previous_output = previous_case.get("llm_output") or {}
    updated_output = updated_case.get("llm_output") or {}
    previous_steps = _normalised_items_for_diff(
        previous_output.get("reasoning_steps") or previous_case.get("reasoning_steps") or [], "s"
    )
    updated_steps = _normalised_items_for_diff(
        updated_output.get("reasoning_steps") or updated_case.get("reasoning_steps") or [], "s"
    )
    previous_premises = _normalised_items_for_diff(
        previous_case.get("premises") or previous_case.get("context") or [], "p"
    )
    updated_premises = _normalised_items_for_diff(
        updated_case.get("premises") or updated_case.get("context") or [], "p"
    )

    global_reasons: list[str] = []
    if previous_premises != updated_premises:
        global_reasons.append("premise_changed")
    if (previous_case.get("semantic_relations") or []) != (updated_case.get("semantic_relations") or []):
        global_reasons.append("semantic_relation_changed")
    if str(previous_case.get("question") or "") != str(updated_case.get("question") or ""):
        global_reasons.append("question_changed")

    previous_answer = str(previous_output.get("answer", previous_case.get("predicted_answer")) or "").strip()
    updated_answer = str(updated_output.get("answer", updated_case.get("predicted_answer")) or "").strip()
    answer_changed = previous_answer != updated_answer
    previous_answer_dependencies = previous_output.get("answer_depends_on") or previous_case.get("answer_depends_on") or []
    updated_answer_dependencies = updated_output.get("answer_depends_on") or updated_case.get("answer_depends_on") or []
    answer_dependency_changed = previous_answer_dependencies != updated_answer_dependencies
    gold_changed = str(previous_case.get("gold_answer") or "").strip() != str(updated_case.get("gold_answer") or "").strip()

    max_len = max(len(previous_steps), len(updated_steps))
    earliest_step_change: int | None = None
    changed_node_ids: list[str] = []
    changed_indexes: list[int] = []
    id_shape_changed = len(previous_steps) != len(updated_steps)
    for index in range(max_len):
        previous = previous_steps[index] if index < len(previous_steps) else None
        updated = updated_steps[index] if index < len(updated_steps) else None
        if previous and updated and previous["id"] != updated["id"]:
            id_shape_changed = True
        if previous != updated:
            changed_indexes.append(index)
            if earliest_step_change is None:
                earliest_step_change = index
            for item in (previous, updated):
                if item and item["id"] not in changed_node_ids:
                    changed_node_ids.append(item["id"])

    base = {
        "changed_node_ids": changed_node_ids,
        "global_change_reasons": global_reasons,
        "answer_changed": answer_changed,
        "answer_dependency_changed": answer_dependency_changed,
        "gold_changed": gold_changed,
    }
    if global_reasons:
        return {
            **base, "mode": "full_fallback", "reason": ", ".join(global_reasons),
            "reuse_reasoning_before": 0, "reuse_reasoning_ids": [], "revalidate_reasoning_ids": [],
            "reuse_final": False,
        }

    if earliest_step_change is not None:
        suffix_ids = [item["id"] for item in updated_steps[earliest_step_change:]]
        fallback = {
            **base,
            "mode": "suffix_incremental",
            "reason": "Changed reasoning could not be safely localized; reuse only the unchanged prefix.",
            "reuse_reasoning_before": earliest_step_change,
            "reuse_reasoning_ids": [item["id"] for item in updated_steps[:earliest_step_change]],
            "revalidate_reasoning_ids": suffix_ids,
            "reuse_final": False,
            "candidate_suffix_count": len(suffix_ids),
            "graph_local_count": len(suffix_ids),
            "scope_reduction_percent": 0.0,
        }
        if previous_result is None or id_shape_changed:
            fallback["reason"] = "Previous graph unavailable or reasoning IDs/length changed; using conservative suffix."
            return fallback

        old_by_id = {item["id"]: item for item in previous_steps}
        new_by_id = {item["id"]: item for item in updated_steps}
        changed_existing = {node_id for node_id in changed_node_ids if node_id in new_by_id}
        seed_predicates: set[tuple[str, int]] = set()
        for node_id in changed_existing:
            old_formula = _normalised_formula_for_text(previous_case, old_by_id[node_id]["text"])
            new_formula = _normalised_formula_for_text(updated_case, new_by_id[node_id]["text"])
            if old_formula is None or new_formula is None:
                fallback["reason"] = "A changed reasoning sentence is untranslatable; using conservative suffix."
                return fallback
            seed_predicates.update(predicates_in_formula(old_formula))
            seed_predicates.update(predicates_in_formula(new_formula))

        flow = _predicate_flow_graph(updated_case)
        reachable = _reachable_predicates(seed_predicates, flow)
        affected = set(changed_existing)
        affected.update(
            node_id for node_id in _result_descendants(previous_result, changed_existing)
            if node_id.startswith("s")
        )
        for index, item in enumerate(updated_steps):
            if index < earliest_step_change or item["id"] in affected:
                continue
            formula = _normalised_formula_for_text(updated_case, item["text"])
            if formula is None:
                # Existing untranslatable nodes remain safely reused unless they
                # were already dependency descendants.
                continue
            if predicates_in_formula(formula) & reachable:
                affected.add(item["id"])
        affected.update(
            node_id for node_id in _result_descendants(previous_result, affected)
            if node_id.startswith("s")
        )

        ordered_affected = [item["id"] for item in updated_steps if item["id"] in affected]
        reuse_ids = [item["id"] for item in updated_steps if item["id"] not in affected]

        # Final is affected if it was a graph descendant, the answer changed, or
        # the edited predicate can reach the final claim predicate.
        final_affected = answer_changed or "final" in _result_descendants(previous_result, affected)
        try:
            q = parse_question(str(updated_case.get("question") or ""))
            if q.formula is not None:
                layer = SemanticLayer(parse_semantic_relations(updated_case.get("semantic_relations")))
                q_formula, _ = layer.normalize_formula(q.formula)
                final_claim = _claim_for_answer(q_formula, _normalise_answer(updated_answer, "LLM answer"))
                final_affected = final_affected or bool(predicates_in_formula(final_claim) & reachable)
        except Exception:
            final_affected = True

        suffix_count = len(suffix_ids)
        graph_count = len(ordered_affected)
        reduction = ((suffix_count - graph_count) / suffix_count * 100) if suffix_count else 0.0
        mode = "graph_local_incremental" if graph_count < suffix_count else "suffix_incremental"
        reason = (
            "Dependency descendants plus predicate-flow relevance selected only affected branches."
            if mode == "graph_local_incremental"
            else "The affected set spans the full suffix; using suffix-equivalent incremental verification."
        )
        return {
            **base,
            "mode": mode,
            "reason": reason,
            "reuse_reasoning_before": 0,
            "reuse_reasoning_ids": reuse_ids,
            "revalidate_reasoning_ids": ordered_affected,
            "reuse_final": not final_affected,
            "candidate_suffix_count": suffix_count,
            "graph_local_count": graph_count,
            "scope_reduction_percent": round(reduction, 2),
            "predicate_seed_count": len(seed_predicates),
            "reachable_predicate_count": len(reachable),
            "affected_set_policy": "Previous dependency descendants ∪ later claims reachable through premise/semantic predicate flow; parity mismatch triggers full fallback.",
        }

    all_ids = [item["id"] for item in updated_steps]
    if answer_changed or answer_dependency_changed:
        return {
            **base, "mode": "final_only",
            "reason": "Only the predicted answer or its declared dependency changed; reuse all reasoning nodes and revalidate Final.",
            "reuse_reasoning_before": 0, "reuse_reasoning_ids": all_ids,
            "revalidate_reasoning_ids": [], "reuse_final": False,
        }
    if gold_changed:
        return {
            **base, "mode": "metadata_only",
            "reason": "Only gold_answer changed; logical statuses and Final are reusable.",
            "reuse_reasoning_before": 0, "reuse_reasoning_ids": all_ids,
            "revalidate_reasoning_ids": [], "reuse_final": True,
        }
    return {
        **base, "mode": "no_change", "reason": "No verification-relevant change detected.",
        "reuse_reasoning_before": 0, "reuse_reasoning_ids": all_ids,
        "revalidate_reasoning_ids": [], "reuse_final": True,
    }

def _status_signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "nodes": {
            node["id"]: {
                "proof_status": node.get("proof_status"),
                "chain_status": node.get("chain_status"),
                "formal": node.get("formal"),
                "proof_dependencies": sorted(node.get("proof_dependencies") or []),
                "reasoning_dependencies": sorted(node.get("reasoning_dependencies") or []),
                "declared_reasoning_dependencies": sorted(node.get("declared_reasoning_dependencies") or []),
                "inferred_reasoning_dependencies": sorted(node.get("inferred_reasoning_dependencies") or []),
                "chain_dependency_source": node.get("chain_dependency_source"),
                "blocking_parent_nodes": sorted(node.get("blocking_parent_nodes") or []),
                "local_support_status": node.get("local_support_status"),
                "dependency_confidence": node.get("dependency_confidence"),
                "declared_dependency_sufficient": node.get("declared_dependency_sufficient"),
                "minimal_proof_paths": node.get("minimal_proof_paths") or [],
            }
            for node in result.get("nodes", [])
        },
        "edges": sorted(
            (edge.get("source"), edge.get("target"), edge.get("relation"))
            for edge in result.get("edges", [])
        ),
        "final_proof_status": result.get("summary", {}).get("final_proof_status"),
        "final_chain_status": result.get("summary", {}).get("final_chain_status"),
    }


def _decorate_incremental_metrics_without_counterfactuals(result: dict[str, Any]) -> None:
    _decorate_view_metrics(result, relations=CHAIN_RELATIONS, prefix="chain")
    _decorate_view_metrics(result, relations=PROOF_RELATIONS, prefix="proof")
    for node in result["nodes"]:
        node["chain_impact_level"] = _chain_impact(node)
        if node["kind"] == "answer":
            node["logical_impact_level"] = "final"
            node["impact_level"] = "final"
            continue
        node["logical_impact_level"] = "not_computed"
        node["impact_level"] = node["chain_impact_level"]
        node["logical_final_changes_if_removed"] = None
        node["logical_affected_if_removed"] = None
        node["chain_final_changes_if_removed"] = None
        node["chain_affected_if_removed"] = None
        node["alternative_proof_exists"] = bool(node.get("alternative_proof_paths"))
        node["strict_chain_breaks_if_removed"] = bool(node.get("chain_reaches_final"))
        node["strict_chain_affected_nodes"] = int(node.get("chain_descendant_count") or 0)
        node["chain_repairable"] = None
        node["logical_answer_preserved_if_removed"] = None
    result["summary"]["critical_chain_nodes"] = sum(
        node["chain_impact_level"] == "critical"
        for node in result["nodes"]
        if node["kind"] != "answer"
    )
    result["summary"]["logically_necessary_nodes"] = None
    result["summary"]["alternative_proof_nodes"] = sum(
        node.get("alternative_proof_exists") is True
        for node in result["nodes"]
        if node["kind"] != "answer"
    )
    result["summary"]["nodes_with_displayed_alternative_paths"] = sum(
        bool(node.get("alternative_proof_paths")) for node in result["nodes"]
    )
    result["summary"]["minimized_z3_cores"] = sum(
        node.get("proof_core_minimized") is True for node in result["nodes"]
    )


def _build_edit_diff(
    previous_result: dict[str, Any],
    current_result: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    before = {node.get("id"): node for node in previous_result.get("nodes", [])}
    after = {node.get("id"): node for node in current_result.get("nodes", [])}
    status_changes: list[dict[str, Any]] = []
    dependency_changes: list[dict[str, Any]] = []
    text_changes: list[dict[str, Any]] = []
    for node_id in sorted(set(before) | set(after)):
        old = before.get(node_id)
        new = after.get(node_id)
        if old is None or new is None:
            status_changes.append({"node_id": node_id, "before": "missing" if old is None else "present", "after": "missing" if new is None else "present"})
            continue
        if old.get("text") != new.get("text"):
            text_changes.append({"node_id": node_id, "before": old.get("text"), "after": new.get("text")})
        if (old.get("proof_status"), old.get("chain_status")) != (new.get("proof_status"), new.get("chain_status")):
            status_changes.append({
                "node_id": node_id,
                "proof_before": old.get("proof_status"), "proof_after": new.get("proof_status"),
                "chain_before": old.get("chain_status"), "chain_after": new.get("chain_status"),
            })
        old_deps = {
            "proof": sorted(old.get("proof_dependencies") or []),
            "reasoning": sorted(old.get("reasoning_dependencies") or []),
            "conflict": sorted(old.get("reasoning_conflict_dependencies") or []),
            "blocking": sorted(old.get("blocking_parent_nodes") or []),
        }
        new_deps = {
            "proof": sorted(new.get("proof_dependencies") or []),
            "reasoning": sorted(new.get("reasoning_dependencies") or []),
            "conflict": sorted(new.get("reasoning_conflict_dependencies") or []),
            "blocking": sorted(new.get("blocking_parent_nodes") or []),
        }
        if old_deps != new_deps:
            dependency_changes.append({"node_id": node_id, "before": old_deps, "after": new_deps})
    old_roots = set(previous_result.get("summary", {}).get("root_error_nodes") or [])
    new_roots = set(current_result.get("summary", {}).get("root_error_nodes") or [])
    return {
        "requested_changed_node_ids": list(plan.get("changed_node_ids") or []),
        "text_changes": text_changes,
        "status_changes": status_changes,
        "dependency_changes": dependency_changes,
        "new_root_error_nodes": sorted(new_roots - old_roots),
        "resolved_root_error_nodes": sorted(old_roots - new_roots),
        "final_before": {
            "proof": previous_result.get("summary", {}).get("final_proof_status"),
            "chain": previous_result.get("summary", {}).get("final_chain_status"),
        },
        "final_after": {
            "proof": current_result.get("summary", {}).get("final_proof_status"),
            "chain": current_result.get("summary", {}).get("final_chain_status"),
        },
        "final_changed": (
            previous_result.get("summary", {}).get("final_proof_status") != current_result.get("summary", {}).get("final_proof_status")
            or previous_result.get("summary", {}).get("final_chain_status") != current_result.get("summary", {}).get("final_chain_status")
        ),
    }


def verify_case_incremental(
    previous_case: dict[str, Any],
    updated_case: dict[str, Any],
    previous_result: dict[str, Any],
    *,
    prefer_z3: bool = True,
    validate_against_full: bool = True,
) -> dict[str, Any]:
    """Graph-local incremental verification with parity-safe fallback."""
    plan = plan_incremental_revalidation(previous_case, updated_case, previous_result)
    execution_stats: dict[str, Any] = {}
    started = perf_counter()
    reusable_modes = {
        "graph_local_incremental", "suffix_incremental", "final_only",
        "metadata_only", "no_change",
    }
    use_reuse = plan["mode"] in reusable_modes
    result = _verify_core(
        updated_case,
        prefer_z3=prefer_z3,
        previous_result=previous_result if use_reuse else None,
        reuse_reasoning_before=int(plan.get("reuse_reasoning_before") or 0) if use_reuse else 0,
        reuse_reasoning_ids=set(plan.get("reuse_reasoning_ids") or []) if use_reuse else None,
        reuse_final=bool(plan.get("reuse_final")) if use_reuse else False,
        execution_stats=execution_stats,
    )
    _decorate_incremental_metrics_without_counterfactuals(result)
    result = enhance_reasoning_diagnostics(updated_case, result)
    incremental_runtime_ms = (perf_counter() - started) * 1000

    reused_ids = list(execution_stats.get("reused_reasoning_nodes") or [])
    revalidated_ids = list(execution_stats.get("revalidated_reasoning_nodes") or [])
    final_reused = bool(execution_stats.get("reused_final"))
    revalidated_with_final = revalidated_ids + ([] if final_reused else ["final"])
    total_reasoning = len(reused_ids) + len(revalidated_ids)
    parity: dict[str, Any] = {
        "checked": False,
        "matches_full_verification": None,
        "full_runtime_ms": None,
        "difference_note": None,
        "speedup_ratio": None,
        "runtime_reduction_percent": None,
    }
    if validate_against_full:
        full_started = perf_counter()
        full_result = verify_case(updated_case, prefer_z3=prefer_z3, compute_counterfactuals=False)
        full_runtime_ms = (perf_counter() - full_started) * 1000
        matches = _status_signature(result) == _status_signature(full_result)
        speedup = (full_runtime_ms / incremental_runtime_ms) if incremental_runtime_ms > 0 else None
        reduction = ((full_runtime_ms - incremental_runtime_ms) / full_runtime_ms * 100) if full_runtime_ms > 0 else None
        parity = {
            "checked": True,
            "matches_full_verification": matches,
            "full_runtime_ms": round(full_runtime_ms, 3),
            "difference_note": None if matches else "Incremental and full signatures differ; the full result was returned for safety.",
            "speedup_ratio": round(speedup, 3) if speedup is not None else None,
            "runtime_reduction_percent": round(reduction, 2) if reduction is not None else None,
        }
        if not matches:
            result = full_result
            for node in result.get("nodes", []):
                node["verification_origin"] = "full_parity_fallback"
            plan = {**plan, "mode": "full_parity_fallback", "reason": parity["difference_note"]}

    reuse_percent = (len(reused_ids) / total_reasoning * 100) if total_reasoning else 0.0
    result["schema_version"] = "0.15.0"
    result["edit_diff"] = _build_edit_diff(previous_result, result, plan)
    result["incremental"] = {
        **plan,
        "reused_reasoning_node_ids": reused_ids,
        "revalidated_reasoning_node_ids": revalidated_ids,
        "revalidated_node_ids": revalidated_with_final,
        "final_reused": final_reused,
        "reused_reasoning_count": len(reused_ids),
        "revalidated_reasoning_count": len(revalidated_ids),
        "total_reasoning_nodes": total_reasoning,
        "reuse_percent": round(reuse_percent, 2),
        "incremental_runtime_ms": round(incremental_runtime_ms, 3),
        "counterfactuals_recomputed": False,
        "solver_stats": {
            "logical_claim_checks_estimate": int(execution_stats.get("solver_checks") or 0),
            "backend_solver_instances": int(execution_stats.get("solver_instances") or 0),
            "backend_solver_checks": int(execution_stats.get("backend_solver_checks") or 0),
            "knowledge_assertions": int(execution_stats.get("knowledge_assertions") or 0),
            "core_minimization_checks": int(execution_stats.get("core_minimization_checks") or 0),
            "strategy": (
                "Persistent per-case Z3 session: trusted knowledge is asserted once; K∧F and K∧¬F use assumption labels on shared tracked/display solvers."
                if result.get("engine") == "z3"
                else "Finite Horn fallback; no SMT solver instances were created."
            ),
        },
        "affected_set_policy": plan.get("affected_set_policy") or "Conservative fallback policy.",
        "parity_validation": parity,
    }
    result["summary"]["last_revalidation_mode"] = result["incremental"]["mode"]
    result["summary"]["last_reused_reasoning_count"] = len(reused_ids)
    result["summary"]["last_revalidated_reasoning_count"] = len(revalidated_ids)
    result["summary"]["last_incremental_runtime_ms"] = round(incremental_runtime_ms, 3)
    result["summary"]["last_scope_reduction_percent"] = plan.get("scope_reduction_percent")
    result["summary"]["last_parity_match"] = parity.get("matches_full_verification")
    return result

