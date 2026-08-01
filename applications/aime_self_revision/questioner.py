"""Turn graph issues and deterministic defects into focused revision questions."""
from __future__ import annotations
from typing import Any

TEMPLATES={
 "certainty_escalation":"What exact premises justify the stronger certainty of {nodes}?",
 "assertion_type_jump":"Does {nodes} incorrectly turn an association or description into a causal or universal claim?",
 "scope_widening":"Is the population, range, or quantifier in {nodes} broader than the premises allow?",
 "unsupported_conclusion":"Which explicit equations or deductions support {nodes}?",
 "missing_premise":"What missing premise is required for {nodes}, and is it given by the problem?",
 "invalid_derivation":"Recompute the step involving {nodes}; does the conclusion follow algebraically?",
 "internal_contradiction":"Which of the conflicting claims involving {nodes} should be corrected?",
 "orphan_node":"How should {nodes} connect to the main proof, or should it be removed?",
 "shallow_justification":"What intermediate derivation between the premise and {nodes} is missing?",
}

def _nodes(item: dict[str,Any]) -> str:
    ids=item.get("node_ids") or []
    return ", ".join(map(str,ids)) or "the affected claim"

def build(graph: dict[str,Any], defects: list[Any], *, limit: int=8) -> list[str]:
    records=[]
    for issue in graph.get("issues",[]) or []:
        if isinstance(issue,dict): records.append(issue)
    for defect in defects:
        records.append(defect.as_dict() if hasattr(defect,"as_dict") else dict(defect))
    questions=[]; seen=set()
    for item in records:
        kind=str(item.get("kind") or item.get("issue_type") or item.get("category") or "").lower()
        q=TEMPLATES.get(kind)
        if q: q=q.format(nodes=_nodes(item))
        else:
            explanation=str(item.get("explanation") or item.get("summary") or item.get("title") or "").strip()
            if not explanation: continue
            q=f"Inspect {_nodes(item)} and repair this issue: {explanation}"
        key=q.lower()
        if key not in seen: seen.add(key); questions.append(q)
        if len(questions)>=limit: break
    if not questions:
        questions.append("Independently recompute the key equations and verify that the final integer follows from all stated cases.")
    return questions
