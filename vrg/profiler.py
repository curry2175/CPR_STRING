from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable


CHAIN_RELATIONS = {
    "source_match",
    "reasoning_dependency",
    "reasoning_conflict",
    "semantic_bridge",
    "semantic_normalization",
    "error_propagation",
}
PROOF_RELATIONS = {"proof_support", "proof_conflict", "semantic_bridge", "semantic_normalization"}
ERROR_STATUSES = {
    "contradiction",
    "ungrounded",
    "untranslatable",
    "insufficient_declared_support",
    "blocked_by_upstream_error",
}


def _edges(result: dict[str, Any], relations: set[str]) -> list[dict[str, Any]]:
    return [edge for edge in result.get("edges", []) if str(edge.get("relation")) in relations]


def _ancestor_ids(edges: Iterable[dict[str, Any]], target: str) -> set[str]:
    reverse: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        reverse[str(edge.get("target"))].add(str(edge.get("source")))
    seen: set[str] = set()
    stack = list(reverse.get(target, set()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(reverse.get(current, set()))
    return seen


def _depths(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, int]:
    parents: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        parents[str(edge.get("target"))].add(str(edge.get("source")))
    ordered = sorted(nodes, key=lambda n: int(n.get("order") or 0))
    depths: dict[str, int] = {}
    for node in ordered:
        node_id = str(node.get("id"))
        parent_depths = [depths[parent] for parent in parents.get(node_id, set()) if parent in depths]
        depths[node_id] = (max(parent_depths) + 1) if parent_depths else 0
    return depths


def _pct(numerator: int | float, denominator: int | float) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator) * 100, 2)


def _canonical_path(path: list[str], lookup: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    canonical: set[str] = set()
    for node_id in path:
        node = lookup.get(str(node_id), {})
        matches = [str(x) for x in node.get("source_matches") or []]
        if node.get("kind") == "reasoning" and len(matches) == 1:
            canonical.add(matches[0])
        else:
            canonical.add(str(node_id))
    return tuple(sorted(canonical))


def build_reasoning_fingerprint(
    result: dict[str, Any],
    *,
    context_proof_dependencies: list[str] | None = None,
) -> dict[str, Any]:
    nodes = list(result.get("nodes") or [])
    edges = list(result.get("edges") or [])
    lookup = {str(node.get("id")): node for node in nodes}
    premise_nodes = [node for node in nodes if node.get("kind") == "premise"]
    reasoning_nodes = [node for node in nodes if node.get("kind") == "reasoning"]
    semantic_nodes = [node for node in nodes if node.get("kind") == "semantic"]

    chain_edges = _edges(result, CHAIN_RELATIONS)
    proof_edges = _edges(result, PROOF_RELATIONS)
    chain_ancestors = _ancestor_ids(chain_edges, "final")
    proof_ancestors = _ancestor_ids(proof_edges, "final")
    depths = _depths(nodes, chain_edges)
    depth_counts = Counter(depths.values())

    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in chain_edges:
        source, target = str(edge.get("source")), str(edge.get("target"))
        outgoing[source].add(target)
        incoming[target].add(source)

    graph_node_count = len(nodes)
    graph_edge_count = len(edges)
    possible_edges = graph_node_count * max(0, graph_node_count - 1)
    branching_nodes = [node_id for node_id, targets in outgoing.items() if targets]
    convergence_nodes = [node_id for node_id, sources in incoming.items() if len(sources) > 1]
    relevant_reasoning = [node for node in reasoning_nodes if str(node.get("id")) in chain_ancestors]
    irrelevant_reasoning = [node for node in reasoning_nodes if str(node.get("id")) not in chain_ancestors]
    authored_used_premises = sorted(
        str(node.get("id")) for node in premise_nodes if str(node.get("id")) in chain_ancestors
    )
    logical_used_premises = sorted(set(str(x) for x in (context_proof_dependencies or [])))
    premise_ids = {str(node.get("id")) for node in premise_nodes}
    logical_used_premises = [node_id for node_id in logical_used_premises if node_id in premise_ids]
    distractor_premises = sorted(premise_ids - set(logical_used_premises or authored_used_premises))

    relevant_shape_nodes = [
        node for node in nodes
        if str(node.get("id")) in chain_ancestors or str(node.get("id")) == "final"
    ]
    linear_nodes = 0
    for node in relevant_shape_nodes:
        node_id = str(node.get("id"))
        if len(incoming.get(node_id, set())) <= 1 and len(outgoing.get(node_id, set())) <= 1:
            linear_nodes += 1

    final = lookup.get("final", {})
    minimal_paths = [list(map(str, path)) for path in final.get("minimal_proof_paths") or []]
    canonical_paths = [_canonical_path(path, lookup) for path in minimal_paths]
    canonical_unique = sorted(set(canonical_paths))
    benign_multiplicity = max(0, len(minimal_paths) - len(canonical_unique))
    genuine_alternatives = max(0, len(canonical_unique) - 1)

    path_sets = [set(path) for path in minimal_paths]
    indispensable: set[str] = set.intersection(*path_sets) if path_sets else set()
    bottleneck_rows: list[dict[str, Any]] = []
    for node in nodes:
        node_id = str(node.get("id"))
        if node.get("kind") == "answer":
            continue
        reaches = bool(node.get("chain_reaches_final"))
        blast = int(node.get("chain_descendant_count") or 0)
        path_coverage = (
            sum(node_id in path for path in path_sets) / len(path_sets)
            if path_sets else 0.0
        )
        score = round(0.45 * path_coverage + 0.35 * (1.0 if reaches else 0.0) + 0.20 * min(1.0, blast / max(1, len(nodes) - 1)), 3)
        if score > 0 or node_id in indispensable:
            bottleneck_rows.append({
                "node_id": node_id,
                "text": node.get("text"),
                "bottleneck_score": score,
                "minimal_proof_coverage_percent": round(path_coverage * 100, 2),
                "chain_descendant_count": blast,
                "indispensable_across_minimal_proofs": node_id in indispensable,
            })
    bottleneck_rows.sort(key=lambda row: (-float(row["bottleneck_score"]), str(row["node_id"])))

    root_errors = [
        node for node in reasoning_nodes
        if node.get("chain_status") in {"contradiction", "ungrounded", "untranslatable", "insufficient_declared_support"}
    ]
    affected = [node for node in nodes if node.get("chain_status") == "blocked_by_upstream_error"]
    first_error = min(root_errors, key=lambda n: int(n.get("order") or 0), default=None)

    restatements = [node for node in reasoning_nodes if node.get("reasoning_role") == "premise_restatement"]
    compound = [node for node in reasoning_nodes if node.get("atomicity_status") not in {None, "atomic"}]
    unsupported = [node for node in reasoning_nodes if node.get("proof_status") == "ungrounded"]
    contradictory = [node for node in reasoning_nodes if node.get("proof_status") == "contradiction"]

    shape = {
        "node_count": graph_node_count,
        "edge_count": graph_edge_count,
        "premise_node_count": len(premise_nodes),
        "reasoning_node_count": len(reasoning_nodes),
        "semantic_node_count": len(semantic_nodes),
        "max_chain_depth": max(depths.values(), default=0),
        "final_chain_depth": depths.get("final", 0),
        "max_width": max(depth_counts.values(), default=0),
        "mean_branching_factor": round(
            sum(len(outgoing[node_id]) for node_id in branching_nodes) / len(branching_nodes), 3
        ) if branching_nodes else 0.0,
        "mean_convergence_degree": round(
            sum(len(incoming[node_id]) for node_id in convergence_nodes) / len(convergence_nodes), 3
        ) if convergence_nodes else 0.0,
        "graph_density": round(graph_edge_count / possible_edges, 4) if possible_edges else 0.0,
        "linearity_percent": _pct(linear_nodes, len(relevant_shape_nodes)),
        "isolated_node_count": sum(not incoming.get(str(node.get("id"))) and not outgoing.get(str(node.get("id"))) for node in nodes),
    }
    grounding = {
        "context_premise_count": len(premise_nodes),
        "logical_used_premise_ids": logical_used_premises,
        "logical_premise_utilization_percent": _pct(len(logical_used_premises), len(premise_nodes)),
        "authored_chain_used_premise_ids": authored_used_premises,
        "authored_chain_premise_utilization_percent": _pct(len(authored_used_premises), len(premise_nodes)),
        "distractor_premise_ids": distractor_premises,
        "distractor_premise_count": len(distractor_premises),
        "unsupported_claim_count": len(unsupported),
        "contradictory_claim_count": len(contradictory),
    }
    efficiency = {
        "final_relevant_reasoning_ids": [str(node.get("id")) for node in relevant_reasoning],
        "final_irrelevant_reasoning_ids": [str(node.get("id")) for node in irrelevant_reasoning],
        "final_relevant_step_percent": _pct(len(relevant_reasoning), len(reasoning_nodes)),
        "final_irrelevant_step_percent": _pct(len(irrelevant_reasoning), len(reasoning_nodes)),
        "premise_restatement_percent": _pct(len(restatements), len(reasoning_nodes)),
        "compound_step_percent": _pct(len(compound), len(reasoning_nodes)),
        "selected_chain_compression_ratio": round(len(relevant_reasoning) / len(reasoning_nodes), 3) if reasoning_nodes else 1.0,
    }
    robustness = {
        "raw_minimal_proof_count": len(minimal_paths),
        "canonical_minimal_proof_count": len(canonical_unique),
        "benign_path_multiplicity_count": benign_multiplicity,
        "genuine_alternative_proof_count": genuine_alternatives,
        "indispensable_node_ids": sorted(indispensable),
        "top_bottleneck_nodes": bottleneck_rows[:10],
    }
    error_profile = {
        "root_error_count": len(root_errors),
        "root_error_node_ids": [str(node.get("id")) for node in root_errors],
        "first_error_node_id": str(first_error.get("id")) if first_error else None,
        "first_error_depth": depths.get(str(first_error.get("id"))) if first_error else None,
        "blocked_node_count": len(affected),
        "max_error_blast_radius": max((int(node.get("chain_descendant_count") or 0) for node in root_errors), default=0),
        "final_chain_affected": final.get("chain_status") not in {"valid", "given", "approved", "advisory"},
    }

    narrative: list[str] = []
    narrative.append(
        f"이 Output은 reasoning {len(reasoning_nodes)}개로 구성되며 Final까지의 chain 깊이는 {shape['final_chain_depth']}이다."
    )
    if grounding["logical_premise_utilization_percent"] is not None:
        narrative.append(
            f"Context premise {len(premise_nodes)}개 중 논리적 최소 근거에 사용된 premise는 {len(logical_used_premises)}개({grounding['logical_premise_utilization_percent']}%)다."
        )
    if irrelevant_reasoning:
        narrative.append(
            f"Reasoning {len(irrelevant_reasoning)}개는 선택된 Final chain에 도달하지 않는 부가 또는 우회 Step이다."
        )
    if genuine_alternatives:
        narrative.append(f"서로 독립적인 대체 증명 경로가 {genuine_alternatives}개 존재한다.")
    elif benign_multiplicity:
        narrative.append("표면상 여러 경로가 있으나 일부는 premise와 그 재진술의 차이에서 생긴 benign multiplicity다.")
    if root_errors:
        narrative.append(
            f"첫 근본 오류는 {error_profile['first_error_node_id']}이며 최대 {error_profile['max_error_blast_radius']}개 하위 Node에 영향을 줄 수 있다."
        )
    else:
        narrative.append("논리적 모순, 근거 없는 주장, 번역 실패 또는 불충분한 declared path는 발견되지 않았다.")

    return {
        "shape": shape,
        "grounding": grounding,
        "efficiency": efficiency,
        "robustness": robustness,
        "error_profile": error_profile,
        "narrative": narrative,
    }
