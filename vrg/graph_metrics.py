from __future__ import annotations

from collections import Counter, deque
from statistics import mean, median
from typing import Any


def _safe_round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _weak_components(node_ids: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source in adjacency and target in adjacency:
            adjacency[source].add(target)
            adjacency[target].add(source)
    seen: set[str] = set()
    components: list[list[str]] = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        stack = [node_id]
        component: list[str] = []
        seen.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _tarjan_scc(node_ids: list[str], adjacency: dict[str, list[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in adjacency.get(node, []):
            if target not in indices:
                strongconnect(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in on_stack:
                lowlink[node] = min(lowlink[node], indices[target])
        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                current = stack.pop()
                on_stack.remove(current)
                component.append(current)
                if current == node:
                    break
            components.append(component)

    for node_id in node_ids:
        if node_id not in indices:
            strongconnect(node_id)
    return components


def _condensed_levels(node_ids: list[str], edges: list[dict[str, Any]]) -> tuple[dict[str, int], int, bool]:
    adjacency = {node_id: [] for node_id in node_ids}
    for edge in edges:
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source in adjacency and target in adjacency and source != target:
            adjacency[source].append(target)
    sccs = _tarjan_scc(node_ids, adjacency)
    component_of = {node: idx for idx, component in enumerate(sccs) for node in component}
    dag_out = {idx: set() for idx in range(len(sccs))}
    indegree = {idx: 0 for idx in range(len(sccs))}
    for source, targets in adjacency.items():
        a = component_of[source]
        for target in targets:
            b = component_of[target]
            if a != b and b not in dag_out[a]:
                dag_out[a].add(b)
                indegree[b] += 1
    queue = deque(sorted(idx for idx, degree in indegree.items() if degree == 0))
    component_level = {idx: 0 for idx in queue}
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in sorted(dag_out[current]):
            component_level[target] = max(component_level.get(target, 0), component_level[current] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    for idx in range(len(sccs)):
        component_level.setdefault(idx, 0)
    levels = {node: component_level[component_of[node]] for node in node_ids}
    has_cycle = any(len(component) > 1 for component in sccs) or any(
        str(edge.get("source")) == str(edge.get("target")) for edge in edges
    )
    return levels, max(levels.values(), default=0), has_cycle or visited != len(sccs)


def calculate_graph_metrics(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate descriptive and quality metrics for a directed reasoning graph.

    Depth is computed on the SCC-condensed DAG, so a cyclic graph remains measurable.
    Width uses longest-path levels from directed roots in that condensed graph.
    These metrics describe structure; they do not by themselves establish graph quality.
    """
    issues = issues or []
    node_ids = [str(node.get("id")) for node in nodes if str(node.get("id", "")).strip()]
    node_set = set(node_ids)
    valid_edges = [
        edge for edge in edges
        if str(edge.get("source")) in node_set and str(edge.get("target")) in node_set
    ]
    # Some verifier graphs keep dependency information on nodes rather than in a
    # top-level edge list. Reconstruct those edges for metric calculation only.
    existing_pairs = {(str(edge.get("source")), str(edge.get("target"))) for edge in valid_edges}
    for node in nodes:
        target = str(node.get("id"))
        dependencies = (
            node.get("reasoning_dependencies") or node.get("dependencies")
            or node.get("proof_dependencies") or node.get("declared_reasoning_dependencies") or []
        )
        for parent in dependencies:
            source = str(parent)
            if source in node_set and target in node_set and source != target and (source, target) not in existing_pairs:
                valid_edges.append({"source": source, "target": target, "relation": "supports", "confidence": 1.0, "derived_for_metrics": True})
                existing_pairs.add((source, target))
    indegree = Counter({node_id: 0 for node_id in node_ids})
    outdegree = Counter({node_id: 0 for node_id in node_ids})
    for edge in valid_edges:
        indegree[str(edge["target"])] += 1
        outdegree[str(edge["source"])] += 1
    roots = [node_id for node_id in node_ids if indegree[node_id] == 0]
    leaves = [node_id for node_id in node_ids if outdegree[node_id] == 0]
    levels, maximum_depth, has_cycle = _condensed_levels(node_ids, valid_edges) if node_ids else ({}, 0, False)
    width_counts = Counter(levels.values())
    width_profile = [width_counts.get(level, 0) for level in range(maximum_depth + 1)] if node_ids else []
    conclusions = [node for node in nodes if node.get("role") == "conclusion" or node.get("kind") == "answer"]
    conclusion_depths = [levels.get(str(node.get("id")), 0) for node in conclusions]
    components = _weak_components(node_ids, valid_edges) if node_ids else []
    possible_edges = len(node_ids) * max(0, len(node_ids) - 1)
    density = len(valid_edges) / possible_edges if possible_edges else 0.0

    evidence_nodes = [node for node in nodes if node.get("role") in {"evidence", "observation"} or node.get("kind") == "premise"]
    limitation_nodes = [node for node in nodes if node.get("role") in {"limitation", "study_design", "analysis_method", "eligibility_criterion", "selection_criterion", "exposure_definition"}]
    issue_node_ids = {str(node_id) for issue in issues for node_id in issue.get("node_ids", [])}
    conclusions_with_issue = [node for node in conclusions if str(node.get("id")) in issue_node_ids]
    exact_source_nodes = [node for node in nodes if node.get("source_fidelity_status") in {"exact", "not_checked", None}]
    source_checked_nodes = [node for node in nodes if node.get("source_fidelity_status") not in {None, "not_checked"}]
    numeric_nodes = [node for node in nodes if node.get("numeric_mentions")]
    numeric_exact_nodes = [node for node in numeric_nodes if node.get("source_span_exact") is True]
    grounded_relations = {"supports", "evidence_for", "causes", "mediates", "precedes", "necessary_for", "defines_population", "defines_time_zero", "defines_exposure", "adjusts_for", "conditions_on", "competes_with"}
    grounded_edges = [edge for edge in valid_edges if edge.get("relation") in grounded_relations and float(edge.get("confidence", 0.0) or 0.0) >= 0.65]
    weak_or_model_edges = [edge for edge in valid_edges if edge.get("relation") in {"supports_weakly", "qualifies", "does_not_establish"} or float(edge.get("confidence", 1.0) or 1.0) < 0.65]

    edge_node_ratio = len(valid_edges) / len(node_ids) if node_ids else 0.0
    complexity_score = _clamp(
        22 + min(28, maximum_depth * 7) + min(22, max(width_profile, default=0) * 3)
        + min(18, edge_node_ratio * 8) + min(10, len(set(edge.get("relation") for edge in valid_edges)) * 1.5)
    ) if node_ids else 0.0
    grounding_rate = len(grounded_edges) / len(valid_edges) if valid_edges else (1.0 if len(node_ids) <= 1 else 0.0)
    connected_rate = 1.0 - ((len(components) - 1) / max(1, len(node_ids) - 1)) if node_ids else 0.0
    grounding_score = _clamp(100 * (0.62 * grounding_rate + 0.23 * connected_rate + 0.15 * (1 - len(weak_or_model_edges) / max(1, len(valid_edges)))))
    formal_issue_count = sum(1 for issue in issues if issue.get("verification_level") == "formal_conflict")
    unsupported_count = sum(1 for issue in issues if issue.get("verification_level") == "rule_confirmed_unsupported")
    method_count = sum(1 for issue in issues if issue.get("verification_level") == "structural_methodological_risk")
    suggested_count = sum(1 for issue in issues if issue.get("verification_level") == "model_suggested_concern")
    integrity_penalty = 20 * formal_issue_count + 12 * unsupported_count + 8 * method_count + 3 * suggested_count
    integrity_score = _clamp(100 - integrity_penalty / max(1.0, len(conclusions) or 1))
    source_alignment_rate = len(exact_source_nodes) / len(source_checked_nodes) if source_checked_nodes else 1.0
    numeric_preservation_rate = len(numeric_exact_nodes) / len(numeric_nodes) if numeric_nodes else 1.0
    inferred_penalty = sum(len(node.get("inferred_details") or []) for node in nodes) / max(1, len(nodes))
    fidelity_score = _clamp(100 * (0.72 * source_alignment_rate + 0.28 * numeric_preservation_rate) - min(20, inferred_penalty * 4))

    return {
        "definitions": {
            "depth": "Longest directed root-to-node level after collapsing strongly connected components.",
            "width": "Number of nodes at each longest-path level; maximum_width is the largest level size.",
            "quality_note": "Width and depth measure structural complexity, not correctness by themselves.",
        },
        "size": {
            "node_count": len(node_ids),
            "edge_count": len(valid_edges),
            "root_count": len(roots),
            "leaf_count": len(leaves),
            "conclusion_count": len(conclusions),
            "connected_component_count": len(components),
        },
        "structure": {
            "maximum_depth": maximum_depth,
            "mean_conclusion_depth": _safe_round(mean(conclusion_depths), 2) if conclusion_depths else 0.0,
            "median_conclusion_depth": _safe_round(median(conclusion_depths), 2) if conclusion_depths else 0.0,
            "maximum_width": max(width_profile, default=0),
            "width_profile": width_profile,
            "mean_branching_factor": _safe_round(mean(outdegree.values()), 3) if outdegree else 0.0,
            "edge_node_ratio": _safe_round(edge_node_ratio, 3),
            "density": _safe_round(density, 5),
            "has_cycle": has_cycle,
            "root_ids": roots,
            "leaf_ids": leaves,
            "node_levels": levels,
        },
        "reasoning_quality": {
            "evidence_to_conclusion_ratio": _safe_round(len(evidence_nodes) / max(1, len(conclusions)), 3),
            "limitation_to_conclusion_ratio": _safe_round(len(limitation_nodes) / max(1, len(conclusions)), 3),
            "conclusions_with_issue_ratio": _safe_round(len(conclusions_with_issue) / max(1, len(conclusions)), 3),
            "grounded_edge_ratio": _safe_round(grounding_rate, 3),
            "weak_or_model_edge_ratio": _safe_round(len(weak_or_model_edges) / max(1, len(valid_edges)), 3),
            "source_alignment_rate": _safe_round(source_alignment_rate, 3),
            "numeric_preservation_rate": _safe_round(numeric_preservation_rate, 3),
            "unique_relation_type_count": len({str(edge.get("relation")) for edge in valid_edges}),
        },
        "scores": {
            "complexity": round(complexity_score, 1),
            "grounding": round(grounding_score, 1),
            "integrity": round(integrity_score, 1),
            "fidelity": round(fidelity_score, 1),
        },
    }
