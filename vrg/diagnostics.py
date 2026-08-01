from __future__ import annotations

import re
from typing import Any

from .engine import ChainSupportEngine, KnowledgeItem
from .logic import Atom, Formula, formula_to_text
from .parser import parse_question, parse_statement
from .semantic import SemanticLayer, parse_semantic_relations

CHAIN_OK = {"given", "valid", "approved", "advisory"}
ROOT_CHAIN_ERRORS = {
    "contradiction",
    "ungrounded",
    "untranslatable",
    "insufficient_declared_support",
    "invalid_dependency_structure",
}


def _coerce(item: Any, prefix: str, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"id": f"{prefix}{index}", "text": item, "depends_on": []}
    if isinstance(item, dict):
        deps = item.get("depends_on") or item.get("dependencies") or []
        if isinstance(deps, str):
            deps = [x.strip() for x in deps.split(",") if x.strip()]
        return {
            "id": str(item.get("id") or f"{prefix}{index}"),
            "text": str(item.get("text") or ""),
            "depends_on": [str(x) for x in deps] if isinstance(deps, list) else [],
        }
    return {"id": f"{prefix}{index}", "text": "", "depends_on": []}


def _minimal_sets(paths: list[list[str]], cap: int = 20) -> tuple[list[list[str]], bool]:
    unique = sorted({frozenset(path) for path in paths}, key=lambda p: (len(p), sorted(p)))
    minimal: list[frozenset[str]] = []
    for candidate in unique:
        if any(existing <= candidate for existing in minimal):
            continue
        minimal.append(candidate)
    truncated = len(minimal) > cap
    return [sorted(path) for path in minimal[:cap]], truncated


def _atomicity(text: str) -> tuple[str, int, str | None]:
    cleaned = text.strip()
    if not cleaned:
        return "empty", 0, "Empty reasoning step"
    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\s*;\s*", cleaned) if p.strip()]
    count = max(1, len(pieces))
    if count > 1:
        return "compound", count, "This step contains multiple sentence-level claims; atomic steps are easier to verify faithfully."
    # A single sentence with multiple explicit conclusion markers is also suspicious.
    marker_count = len(re.findall(r"\b(therefore|thus|hence|consequently|so)\b", cleaned, flags=re.I))
    if marker_count >= 2:
        return "possibly_compound", 2, "Multiple inference markers suggest more than one hidden reasoning move."
    return "atomic", 1, None


def _support_paths(engine: ChainSupportEngine, items: list[KnowledgeItem], target: Formula, max_paths: int = 30) -> list[list[str]]:
    return engine.support_paths(items, target, max_paths=max_paths)


def _role(node: dict[str, Any]) -> tuple[str, str | None]:
    proof = node.get("proof_status")
    chain = node.get("chain_status")
    if proof == "untranslatable":
        return "untranslatable_step", "parser_failure"
    if proof == "contradiction":
        return "contradictory_step", "contradiction"
    if proof == "ungrounded":
        return "unsupported_claim", "ungrounded"
    if chain == "insufficient_declared_support":
        return "globally_valid_but_declared_path_insufficient", "insufficient_declared_support"
    if chain == "blocked_by_upstream_error":
        return "globally_valid_but_upstream_contaminated", "upstream_error"
    if node.get("source_matches"):
        return "premise_restatement", None
    if proof == "valid" and node.get("proof_reaches_final"):
        return "valid_final_support", None
    if proof == "valid":
        return "valid_but_not_used_in_selected_final_proof", None
    return "other", None


def _refresh_error_edges_and_summary(result: dict[str, Any]) -> None:
    nodes = result.get("nodes") or []
    lookup = {str(node.get("id")): node for node in nodes}
    # Re-propagate chain errors in authored/inferred order after local-support diagnostics.
    for node in sorted(nodes, key=lambda n: int(n.get("order") or 0)):
        if node.get("kind") not in {"reasoning", "answer"}:
            continue
        if node.get("chain_status") in ROOT_CHAIN_ERRORS:
            node["blocking_parent_nodes"] = []
            node["upstream_error_nodes"] = [node.get("id")]
            continue
        deps = sorted(set((node.get("reasoning_dependencies") or []) + (node.get("reasoning_conflict_dependencies") or [])))
        blocking: list[str] = []
        roots: set[str] = set()
        for dep_id in deps:
            parent = lookup.get(dep_id)
            if not parent or parent.get("chain_status") in CHAIN_OK:
                continue
            blocking.append(dep_id)
            roots.update(parent.get("upstream_error_nodes") or [dep_id])
        if blocking and node.get("proof_status") != "contradiction":
            node["chain_status"] = "blocked_by_upstream_error"
            node["blocking_parent_nodes"] = sorted(set(blocking))
            node["upstream_error_nodes"] = sorted(roots)
            node["chain_detail"] = (str(node.get("chain_detail") or "") + "; blocked by diagnosed upstream path error").strip("; ")

    edges = [edge for edge in result.get("edges") or [] if edge.get("relation") != "error_propagation"]
    edge_keys = {(e.get("source"), e.get("target"), e.get("relation")) for e in edges}
    for node in nodes:
        for source in node.get("blocking_parent_nodes") or []:
            key = (source, node.get("id"), "error_propagation")
            if key not in edge_keys:
                edges.append({"source": source, "target": node.get("id"), "relation": "error_propagation"})
                edge_keys.add(key)
    result["edges"] = sorted(edges, key=lambda e: (str(e.get("source")), str(e.get("target")), str(e.get("relation"))))

    reasoning = [n for n in nodes if n.get("kind") == "reasoning"]
    final = next((n for n in nodes if n.get("id") == "final"), {})
    summary = result.setdefault("summary", {})
    summary["final_chain_status"] = final.get("chain_status")
    summary["all_reasoning_chain_valid"] = all(n.get("chain_status") == "valid" for n in reasoning)
    summary["blocked_reasoning_count"] = sum(n.get("chain_status") == "blocked_by_upstream_error" for n in reasoning)
    summary["chain_error_count"] = sum(n.get("chain_status") not in CHAIN_OK for n in reasoning)
    summary["root_error_nodes"] = [n.get("id") for n in reasoning if n.get("chain_status") in ROOT_CHAIN_ERRORS]
    summary["valid_answer_but_invalid_reasoning"] = bool(
        final.get("proof_status") == "valid" and any(n.get("chain_status") != "valid" for n in reasoning)
    )
    summary["error_propagation_edge_count"] = sum(e.get("relation") == "error_propagation" for e in result["edges"])


def enhance_reasoning_diagnostics(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Add premise-given reasoning diagnostics without changing Proof semantics.

    The diagnostic layer focuses on whether the AI's *used local route* is
    sufficient, how ambiguous an inferred route is, and whether steps are
    indispensable, optional, redundant, compound, or erroneous.
    """
    premises_raw = case.get("premises") or case.get("context") or []
    output = case.get("llm_output") or {}
    steps_raw = output.get("reasoning_steps") or case.get("reasoning_steps") or []
    premises = [_coerce(item, "p", i) for i, item in enumerate(premises_raw, 1)]
    steps = [_coerce(item, "s", i) for i, item in enumerate(steps_raw, 1)]
    relations = parse_semantic_relations(case.get("semantic_relations"))
    layer = SemanticLayer(relations)
    nodes = result.get("nodes") or []
    lookup = {str(node.get("id")): node for node in nodes}
    node_kinds = {str(node.get("id")): str(node.get("kind")) for node in nodes}
    node_orders = {str(node.get("id")): int(node.get("order") or 0) for node in nodes}
    path_engine = ChainSupportEngine(node_kinds, node_orders)

    formula_by_id: dict[str, Formula] = {}
    chain_items: list[KnowledgeItem] = []
    proof_items: list[KnowledgeItem] = []
    for relation in relations:
        bridge = relation.bridge_rule()
        if bridge is not None and relation.proof_usable:
            formula_by_id[relation.id] = bridge
            item = KnowledgeItem(relation.id, bridge)
            chain_items.append(item)
            proof_items.append(item)
    for premise in premises:
        parsed = parse_statement(premise["text"])
        if parsed.formula is None:
            continue
        formula, _ = layer.normalize_formula(parsed.formula)
        formula_by_id[premise["id"]] = formula
        item = KnowledgeItem(premise["id"], formula)
        chain_items.append(item)
        proof_items.append(item)

    diagnostic_nodes: list[dict[str, Any]] = []
    for step in steps:
        node = lookup.get(step["id"])
        if not node:
            continue
        parsed = parse_statement(step["text"])
        atomicity_status, claim_count, atomicity_warning = _atomicity(step["text"])
        node["atomicity_status"] = atomicity_status
        node["atomic_claim_count_estimate"] = claim_count
        node["atomicity_warning"] = atomicity_warning
        if parsed.formula is None:
            node["dependency_confidence"] = "not_applicable"
            node["dependency_candidate_count"] = 0
            node["candidate_reasoning_paths"] = []
            node["minimal_proof_paths"] = []
            node["minimal_proof_count"] = 0
            diagnostic_nodes.append(node)
            continue
        formula, _ = layer.normalize_formula(parsed.formula)
        formula_by_id[step["id"]] = formula
        target: Formula = formula.complement() if node.get("proof_status") == "contradiction" and isinstance(formula, Atom) else formula

        chain_paths = _support_paths(path_engine, chain_items, target, max_paths=20)
        proof_paths = _support_paths(path_engine, proof_items, target, max_paths=40)
        minimal_paths, minimal_truncated = _minimal_sets(proof_paths, cap=20)
        node["candidate_reasoning_paths"] = chain_paths[:10]
        node["dependency_candidate_count"] = len(chain_paths)
        declared = [str(x) for x in node.get("declared_reasoning_dependencies") or []]
        if declared:
            parent_items = [KnowledgeItem(dep, formula_by_id[dep]) for dep in declared if dep in formula_by_id]
            local = path_engine.support(parent_items, target)
            sufficient = bool(local.found)
            node["declared_dependency_sufficient"] = sufficient
            node["local_support_status"] = "sufficient" if sufficient else "insufficient"
            node["dependency_confidence"] = "declared_and_verified" if sufficient else "declared_but_insufficient"
            if not sufficient and node.get("proof_status") not in {"contradiction", "ungrounded", "untranslatable"}:
                node["chain_status"] = "insufficient_declared_support"
                node["chain_detail"] = (str(node.get("chain_detail") or "") + "; declared parents do not locally derive this claim").strip("; ")
                node["blocking_parent_nodes"] = []
                node["upstream_error_nodes"] = [step["id"]]
        else:
            node["declared_dependency_sufficient"] = None
            if len(chain_paths) == 0:
                node["local_support_status"] = "no_support"
                node["dependency_confidence"] = "no_support"
            elif len(chain_paths) == 1:
                node["local_support_status"] = "inferred_unique"
                node["dependency_confidence"] = "inferred_unique"
            else:
                node["local_support_status"] = "inferred_ambiguous"
                node["dependency_confidence"] = "inferred_ambiguous"
        node["minimal_proof_paths"] = minimal_paths
        node["minimal_proof_count"] = len(minimal_paths)
        node["minimal_proof_paths_truncated"] = minimal_truncated
        used = set((node.get("source_matches") or []) + (node.get("reasoning_dependencies") or []) + (node.get("reasoning_conflict_dependencies") or []))
        selected = set(node.get("proof_dependencies") or [])
        union = used | selected
        node["proof_chain_dependency_overlap_percent"] = round(len(used & selected) / len(union) * 100, 2) if union else 100.0

        chain_items.append(KnowledgeItem(step["id"], formula))
        if node.get("proof_status") == "valid":
            proof_items.append(KnowledgeItem(step["id"], formula))
        diagnostic_nodes.append(node)

    # Final diagnostics.
    final_node = lookup.get("final")
    if final_node:
        q = parse_question(str(case.get("question") or ""))
        if q.formula is not None:
            q_formula, _ = layer.normalize_formula(q.formula)
            answer = str(output.get("answer", case.get("predicted_answer")) or "").strip().lower()
            final_formula = q_formula if answer == "yes" else q_formula.complement()
            target = final_formula.complement() if final_node.get("proof_status") == "contradiction" else final_formula
            final_chain_paths = _support_paths(path_engine, chain_items, target, max_paths=30)
            final_proof_paths = _support_paths(path_engine, proof_items, target, max_paths=60)
            final_minimal_paths, final_truncated = _minimal_sets(final_proof_paths, cap=30)
            final_node["candidate_reasoning_paths"] = final_chain_paths[:10]
            final_node["dependency_candidate_count"] = len(final_chain_paths)
            final_node["minimal_proof_paths"] = final_minimal_paths
            final_node["minimal_proof_count"] = len(final_minimal_paths)
            final_node["minimal_proof_paths_truncated"] = final_truncated
            declared_final = output.get("answer_depends_on") or case.get("answer_depends_on") or []
            if isinstance(declared_final, str):
                declared_final = [x.strip() for x in declared_final.split(",") if x.strip()]
            if declared_final:
                parent_items = [KnowledgeItem(str(dep), formula_by_id[str(dep)]) for dep in declared_final if str(dep) in formula_by_id]
                sufficient = path_engine.support(parent_items, target).found
                final_node["declared_dependency_sufficient"] = sufficient
                final_node["local_support_status"] = "sufficient" if sufficient else "insufficient"
                final_node["dependency_confidence"] = "declared_and_verified" if sufficient else "declared_but_insufficient"
                if not sufficient and final_node.get("proof_status") not in {"contradiction", "ungrounded", "untranslatable"}:
                    final_node["chain_status"] = "insufficient_declared_support"
                    final_node["upstream_error_nodes"] = ["final"]
            else:
                final_node["declared_dependency_sufficient"] = None
                final_node["dependency_confidence"] = (
                    "no_support" if not final_chain_paths else "inferred_unique" if len(final_chain_paths) == 1 else "inferred_ambiguous"
                )
                final_node["local_support_status"] = final_node["dependency_confidence"]

            final_sets = [set(path) for path in final_minimal_paths]
            for node in nodes:
                if node.get("id") == "final":
                    node["final_proof_necessity"] = "final"
                    continue
                memberships = sum(str(node.get("id")) in path for path in final_sets)
                if not final_sets:
                    necessity = "unknown"
                elif memberships == len(final_sets):
                    necessity = "indispensable_across_minimal_proofs"
                elif memberships > 0:
                    necessity = "optional_across_minimal_proofs"
                else:
                    necessity = "not_in_minimal_final_proof"
                node["final_proof_necessity"] = necessity

    _refresh_error_edges_and_summary(result)

    reasoning = [n for n in nodes if n.get("kind") == "reasoning"]
    for node in reasoning:
        role, error_type = _role(node)
        node["reasoning_role"] = role
        node["reasoning_error_type"] = error_type

    count = len(reasoning)
    proof_valid = sum(n.get("proof_status") == "valid" for n in reasoning)
    chain_valid = sum(n.get("chain_status") == "valid" for n in reasoning)
    restatements = sum(n.get("reasoning_role") == "premise_restatement" for n in reasoning)
    redundant = sum(n.get("reasoning_role") == "valid_but_not_used_in_selected_final_proof" for n in reasoning)
    compound = sum(n.get("atomicity_status") != "atomic" for n in reasoning)
    ambiguous = sum(n.get("dependency_confidence") == "inferred_ambiguous" for n in reasoning)
    no_support = sum(n.get("dependency_confidence") == "no_support" for n in reasoning)
    declared = [n for n in reasoning if n.get("declared_reasoning_dependencies")]
    declared_sufficient = sum(n.get("declared_dependency_sufficient") is True for n in declared)

    def pct(num: int, den: int) -> float | None:
        return round(num / den * 100, 2) if den else None

    confidence_points = 0.0
    for node in reasoning:
        value = node.get("dependency_confidence")
        confidence_points += 1.0 if value in {"declared_and_verified", "inferred_unique"} else 0.5 if value == "inferred_ambiguous" else 0.0
    proof_rate = proof_valid / count if count else 1.0
    chain_rate = chain_valid / count if count else 1.0
    confidence_rate = confidence_points / count if count else 1.0
    integrity_score = round((0.45 * proof_rate + 0.35 * chain_rate + 0.20 * confidence_rate) * 100, 2)

    profile = {
        "reasoning_step_count": count,
        "proof_valid_percent": pct(proof_valid, count),
        "chain_valid_percent": pct(chain_valid, count),
        "premise_restatement_percent": pct(restatements, count),
        "selected_proof_redundancy_percent": pct(redundant, count),
        "compound_step_percent": pct(compound, count),
        "ambiguous_inferred_dependency_percent": pct(ambiguous, count),
        "no_local_support_percent": pct(no_support, count),
        "declared_dependency_sufficiency_percent": pct(declared_sufficient, len(declared)),
        "reasoning_integrity_score": integrity_score,
        "score_formula": "45% step Proof validity + 35% Chain validity + 20% dependency-confidence; redundancy is reported separately, not penalized.",
    }
    result["reasoning_quality_profile"] = profile
    result.setdefault("summary", {}).update({
        "reasoning_integrity_score": integrity_score,
        "compound_reasoning_step_count": compound,
        "ambiguous_dependency_step_count": ambiguous,
        "no_local_support_step_count": no_support,
        "insufficient_declared_support_count": sum(n.get("chain_status") == "insufficient_declared_support" for n in reasoning),
        "premise_restatement_count": restatements,
        "selected_proof_redundant_step_count": redundant,
        "final_minimal_proof_count": int(final_node.get("minimal_proof_count") or 0) if final_node else 0,
    })
    return result
