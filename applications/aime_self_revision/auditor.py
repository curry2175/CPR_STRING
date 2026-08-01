"""Deterministic structural checks for a STRING reasoning graph.

Adapted from GracieRho/solving-aime. These checks complement, rather than
replace, the six-agent graph findings.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

STRONG_CERTAINTY={"establishes","proves","certain","always"}
WEAK_CERTAINTY={"possible","suggests","may","uncertain","reported"}
STRONG_ASSERTIONS={"causal","universal","necessity"}
WEAK_ASSERTIONS={"association","descriptive","statistical","other"}

@dataclass(frozen=True)
class Defect:
    id: str
    kind: str
    severity: str
    title: str
    explanation: str
    node_ids: tuple[str,...]=()
    edge_ids: tuple[str,...]=()
    def as_dict(self) -> dict[str,Any]:
        return {"id":self.id,"kind":self.kind,"severity":self.severity,"title":self.title,"explanation":self.explanation,"node_ids":list(self.node_ids),"edge_ids":list(self.edge_ids)}

def _text(node: dict[str,Any]) -> str:
    return " ".join(str(node.get(k) or "") for k in ("source_text","plain_meaning","normalized_claim")).lower()

def _edge_ends(edge: dict[str,Any]) -> tuple[str,str]:
    return str(edge.get("source") or edge.get("from") or edge.get("source_id") or ""), str(edge.get("target") or edge.get("to") or edge.get("target_id") or "")

def audit(graph: dict[str,Any]) -> list[Defect]:
    nodes=[n for n in graph.get("nodes",[]) if isinstance(n,dict)]
    edges=[e for e in graph.get("edges",[]) if isinstance(e,dict)]
    by_id={str(n.get("id")):n for n in nodes}
    incoming={nid:[] for nid in by_id}; outgoing={nid:[] for nid in by_id}
    for e in edges:
        a,b=_edge_ends(e)
        if b in incoming: incoming[b].append(e)
        if a in outgoing: outgoing[a].append(e)
    defects=[]
    def add(kind,severity,title,explanation,node_ids=(),edge_ids=()):
        defects.append(Defect(f"D{len(defects)+1}",kind,severity,title,explanation,tuple(x for x in node_ids if x),tuple(x for x in edge_ids if x)))
    for nid,node in by_id.items():
        certainty=str(node.get("certainty") or "").lower(); assertion=str(node.get("assertion_type") or "").lower(); role=str(node.get("role") or "").lower(); text=_text(node)
        parents=[]
        for e in incoming[nid]:
            a,_=_edge_ends(e)
            if a in by_id: parents.append(by_id[a])
        if certainty in STRONG_CERTAINTY and parents and all(str(p.get("certainty") or "").lower() in WEAK_CERTAINTY for p in parents):
            add("certainty_escalation","high","Certainty escalation","A strong conclusion is derived only from weaker or uncertain premises.",[nid], [str(e.get("id") or "") for e in incoming[nid]])
        if assertion in STRONG_ASSERTIONS and parents and all(str(p.get("assertion_type") or "").lower() in WEAK_ASSERTIONS for p in parents):
            add("assertion_type_jump","high","Assertion-type jump","The graph strengthens descriptive or associative premises into a causal or universal claim.",[nid])
        if any(w in text for w in ("all ","always","every ","never")) and not any(w in " ".join(_text(p) for p in parents) for w in ("all ","always","every ","never")) and parents:
            add("scope_widening","high","Scope widening","The conclusion expands the scope beyond its supporting premises.",[nid])
        if any(w in text for w in ("therefore","proves","must be","hence")) and not incoming[nid] and role in {"conclusion","claim"}:
            add("proof_language_without_proof","medium","Proof language without support","The node uses conclusive language but has no incoming justification edge.",[nid])
        if role=="conclusion" and not incoming[nid]:
            add("unsupported_conclusion","high","Unsupported conclusion","A conclusion node has no represented supporting dependency.",[nid])
        if not incoming[nid] and not outgoing[nid] and len(nodes)>1:
            add("orphan_node","low","Orphan node","This claim is disconnected from the represented reasoning structure.",[nid])
        if role=="conclusion" and len(incoming[nid])==1:
            add("shallow_justification","medium","Shallow justification","The final conclusion rests on only one represented premise; inspect whether intermediate steps are missing.",[nid])
    # Self-defeating support and direct attack/support conflict.
    for e in edges:
        a,b=_edge_ends(e); rel=str(e.get("relation") or e.get("type") or "").lower()
        if a and a==b and rel in {"supports","derives","entails"}:
            add("self_defeating_support","high","Self-supporting claim","A node is used to support itself.",[a],[str(e.get("id") or "")])
    return defects

def summarize(defects: list[Defect]) -> str:
    if not defects: return "No deterministic structural defects detected."
    counts={}
    for d in defects: counts[d.severity]=counts.get(d.severity,0)+1
    return ", ".join(f"{k}={counts[k]}" for k in ("high","medium","low") if counts.get(k))
