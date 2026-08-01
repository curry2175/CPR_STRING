from __future__ import annotations

from copy import deepcopy
from typing import Any


STATUS_ORDER = [
    "given", "valid", "contradiction", "ungrounded", "untranslatable",
    "blocked_by_upstream_error", "insufficient_declared_support", "approved", "advisory", "disabled",
]

EDGE_LAYERS = {
    "source_match": "source",
    "authored_dependency": "reasoning",
    "inferred_dependency": "reasoning",
    "reasoning_dependency": "reasoning",
    "reasoning_conflict": "conflict",
    "proof_support": "proof",
    "proof_conflict": "proof",
    "alternative_proof": "proof",
    "error_propagation": "error",
    "semantic_normalization": "semantic",
    "semantic_bridge": "semantic",
    "semantic_related": "semantic",
    "premise_candidate": "premise_grounding",
    "grounding_suggestion": "premise_grounding",
}

VIEW_RELATIONS = {
    "simple": ["authored_dependency", "reasoning_dependency", "source_match"],
    "reasoning": ["authored_dependency", "inferred_dependency", "reasoning_dependency", "reasoning_conflict", "source_match"],
    "proof": ["proof_support", "proof_conflict", "alternative_proof"],
    "error": ["reasoning_conflict", "proof_conflict", "error_propagation"],
    "semantic": ["semantic_normalization", "semantic_bridge", "semantic_related", "premise_candidate"],
    "all": list(EDGE_LAYERS),
}


def _edge(source: str, target: str, relation: str, **extra: Any) -> dict[str, Any]:
    return {"source": source, "target": target, "relation": relation, "layer": EDGE_LAYERS.get(relation, "other"), **extra}


def build_universal_graph(
    verified_graph: dict[str, Any],
    *,
    formalization_metadata: dict[str, dict[str, Any]] | None = None,
    inferred_premise_candidates: list[dict[str, Any]] | None = None,
    grounding_proposals: list[dict[str, Any]] | None = None,
    repair_iteration: int = 0,
) -> dict[str, Any]:
    graph = deepcopy(verified_graph)
    metadata = formalization_metadata or {}
    nodes = graph.get("nodes") or []
    node_ids = {str(n.get("id")) for n in nodes}
    for node in nodes:
        node_id = str(node.get("id"))
        info = metadata.get(node_id) or (metadata.get("question") if node.get("kind") == "answer" else {})
        if info:
            node["original_text"] = info.get("original_text", node.get("text"))
            node["formalized_text"] = info.get("formalized_text", node.get("text"))
            node["formalization_source"] = info.get("formalization_source")
            node["formalization_confidence"] = info.get("formalization_confidence")
            node["formalization_notes"] = info.get("formalization_notes")
            node["new_vocabulary"] = info.get("new_vocabulary", [])
            node["premise_provenance"] = info.get("premise_provenance")
            node["text"] = info.get("original_text", node.get("text"))
        else:
            node.setdefault("formalization_source", "system_generated" if node.get("kind") in {"semantic", "answer"} else "deterministic_parser")
            node.setdefault("formalization_confidence", "high")
            node.setdefault("original_text", node.get("text"))
            node.setdefault("formalized_text", node.get("text"))
        node["repair_iteration"] = repair_iteration
        node["display_status"] = node.get("chain_status") if node.get("chain_status") not in {"given", "valid", "approved", "advisory"} else node.get("proof_status")
        if node.get("kind") == "semantic":
            node["display_status"] = "advisory" if node.get("semantic_relation_type") == "related_to" else "approved"

    edges: list[dict[str, Any]] = []
    for raw in graph.get("edges") or []:
        edges.append(_edge(str(raw["source"]), str(raw["target"]), str(raw["relation"])))
    for node in nodes:
        target = str(node.get("id"))
        for source in node.get("declared_reasoning_dependencies") or []:
            if source in node_ids and source != target:
                edges.append(_edge(source, target, "authored_dependency", provenance="model_declared"))
        for source in node.get("inferred_reasoning_dependencies") or []:
            if source in node_ids and source != target:
                edges.append(_edge(source, target, "inferred_dependency", provenance="system_inferred"))
        for path in node.get("alternative_proof_paths") or []:
            for source in path:
                if source in node_ids and source != target:
                    edges.append(_edge(source, target, "alternative_proof", path=path))


    for proposal in grounding_proposals or []:
        target = str(proposal.get("node_id") or "")
        if target not in node_ids:
            continue
        for source in proposal.get("suggested_parents") or []:
            if source in node_ids and source != target:
                edges.append(_edge(source, target, "grounding_suggestion", advisory=True, verdict=proposal.get("verdict")))

    for index, candidate in enumerate(inferred_premise_candidates or [], 1):
        cid = str(candidate.get("id") or f"ip_candidate_{index}")
        if cid in node_ids:
            continue
        nodes.append({
            "id": cid,
            "kind": "premise_candidate",
            "order": -1,
            "text": candidate.get("text", ""),
            "original_text": candidate.get("text", ""),
            "formalized_text": candidate.get("controlled_english", candidate.get("text", "")),
            "status": "advisory",
            "proof_status": "advisory",
            "chain_status": "advisory",
            "display_status": "advisory",
            "premise_provenance": candidate.get("provenance", "llm_inferred_assumption"),
            "formalization_source": candidate.get("formalization_source", "llm_premise_grounder"),
            "formalization_confidence": candidate.get("confidence", "low"),
            "approved_for_proof": bool(candidate.get("approved_for_proof", False)),
            "repair_iteration": repair_iteration,
        })
        target = str(candidate.get("supports_node") or "")
        if target in node_ids:
            edges.append(_edge(cid, target, "premise_candidate", approved=bool(candidate.get("approved_for_proof", False))))
        node_ids.add(cid)

    unique = {}
    for e in edges:
        key = (e["source"], e["target"], e["relation"], str(e.get("path", "")))
        unique[key] = e
    edges = list(unique.values())
    status_counts = {status: 0 for status in STATUS_ORDER}
    for node in nodes:
        status = str(node.get("display_status") or node.get("proof_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    relation_counts = {relation: 0 for relation in EDGE_LAYERS}
    for edge in edges:
        relation_counts[edge["relation"]] = relation_counts.get(edge["relation"], 0) + 1

    graph["schema_version"] = "0.19.0"
    graph["nodes"] = nodes
    graph["edges"] = edges
    graph["universal_viewer"] = {
        "status_counts": status_counts,
        "relation_counts": relation_counts,
        "edge_layers": EDGE_LAYERS,
        "views": VIEW_RELATIONS,
        "legend_always_visible": True,
    }
    return graph


def graph_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    bnodes = {str(x.get("id")): x for x in before.get("nodes") or []}
    anodes = {str(x.get("id")): x for x in after.get("nodes") or []}
    added_nodes = sorted(set(anodes) - set(bnodes))
    removed_nodes = sorted(set(bnodes) - set(anodes))
    changed_nodes = []
    for node_id in sorted(set(bnodes) & set(anodes)):
        b, a = bnodes[node_id], anodes[node_id]
        changes = {}
        for field in ("text", "proof_status", "chain_status", "display_status", "declared_reasoning_dependencies", "reasoning_dependencies", "proof_dependencies", "reasoning_error_type"):
            if b.get(field) != a.get(field):
                changes[field] = {"before": b.get(field), "after": a.get(field)}
        if changes:
            changed_nodes.append({"node_id": node_id, "changes": changes})
    def ekey(e: dict[str, Any]) -> tuple[str, str, str]:
        return str(e.get("source")), str(e.get("target")), str(e.get("relation"))
    be = {ekey(x) for x in before.get("edges") or []}
    ae = {ekey(x) for x in after.get("edges") or []}
    return {
        "added_nodes": added_nodes,
        "removed_nodes": removed_nodes,
        "changed_nodes": changed_nodes,
        "added_edges": [dict(zip(("source", "target", "relation"), x)) for x in sorted(ae - be)],
        "removed_edges": [dict(zip(("source", "target", "relation"), x)) for x in sorted(be - ae)],
        "before_summary": before.get("summary", {}),
        "after_summary": after.get("summary", {}),
    }
