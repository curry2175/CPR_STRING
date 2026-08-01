from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class GroundedNodeProposal(BaseModel):
    node_id: str
    suggested_parents: list[str] = Field(default_factory=list)
    verdict: Literal["supported_by_available_context", "unsupported", "needs_implicit_premise", "uncertain"]
    explanation: str = ""


class ImplicitPremiseCandidate(BaseModel):
    text: str
    supports_node: str
    provenance: Literal["context_recovery", "commonsense", "domain_knowledge", "model_assumption"] = "model_assumption"
    confidence: Literal["high", "medium", "low"] = "low"


class PremiseGroundingOutput(BaseModel):
    node_proposals: list[GroundedNodeProposal] = Field(default_factory=list)
    implicit_premises: list[ImplicitPremiseCandidate] = Field(default_factory=list)


def _load_env(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1]
    path = root / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in {"OPENAI_API_KEY", "OPENAI_MODEL"} and v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def _parsed(response: Any) -> PremiseGroundingOutput:
    value = getattr(response, "output_parsed", None)
    if value is not None:
        return value if isinstance(value, PremiseGroundingOutput) else PremiseGroundingOutput.model_validate(value)
    for output in getattr(response, "output", []) or []:
        for item in getattr(output, "content", []) or []:
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return parsed if isinstance(parsed, PremiseGroundingOutput) else PremiseGroundingOutput.model_validate(parsed)
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return PremiseGroundingOutput.model_validate_json(text)
    raise ValueError("Premise grounding returned no structured output")


def ground_failed_nodes(
    case: dict[str, Any],
    graph: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str = "low",
    allow_external_premises: bool = False,
    client: Any = None,
) -> dict[str, Any]:
    failures = [
        node for node in graph.get("nodes") or []
        if node.get("kind") == "reasoning" and (
            node.get("proof_status") in {"ungrounded", "untranslatable"}
            or node.get("chain_status") in {"insufficient_declared_support", "blocked_by_upstream_error"}
        )
    ]
    if not failures:
        return {"node_proposals": [], "implicit_premises": [], "api_call": None}
    _load_env()
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is required for premise grounding")
        from openai import OpenAI
        client = OpenAI()
    premises = case.get("premises") or []
    steps = (case.get("llm_output") or {}).get("reasoning_steps") or []
    system = (
        "You are the premise-grounding component of a neuro-symbolic verifier. "
        "For each diagnosed reasoning node, identify only DIRECT support IDs from the supplied context premises and earlier reasoning steps. "
        "Do not assume the previous final answer is correct. Do not rewrite the proof. "
        "Do not use later steps. If no available support exists, mark unsupported. "
        + ("You may propose clearly labelled implicit premises, but never claim they are context facts. " if allow_external_premises else "External knowledge and implicit premises are forbidden for this task. ")
    )
    user_parts = ["Context premises:"]
    user_parts += [f"{x.get('id')}: {x.get('text')}" for x in premises]
    user_parts.append("\nReasoning steps:")
    user_parts += [f"{x.get('id')}: {x.get('text')} | declared parents={x.get('depends_on', [])}" for x in steps]
    user_parts.append("\nDiagnosed nodes:")
    for node in failures:
        user_parts.append(
            f"{node.get('id')}: {node.get('text')} | proof={node.get('proof_status')} | chain={node.get('chain_status')} | "
            f"declared={node.get('declared_reasoning_dependencies', [])} | inferred={node.get('inferred_reasoning_dependencies', [])}"
        )
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=4000,
        store=False,
        input=[{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_parts)}],
        text_format=PremiseGroundingOutput,
    )
    result = _parsed(response)
    valid_ids = {str(x.get("id")) for x in premises}
    ordered_steps = [str(x.get("id")) for x in steps]
    node_order = {node_id: i for i, node_id in enumerate(ordered_steps)}
    normalized = []
    for proposal in result.node_proposals:
        if proposal.node_id not in node_order:
            continue
        allowed = valid_ids | {sid for sid, idx in node_order.items() if idx < node_order[proposal.node_id]}
        parents = [x for x in proposal.suggested_parents if x in allowed]
        normalized.append({**proposal.model_dump(), "suggested_parents": parents, "invalid_suggested_parents": [x for x in proposal.suggested_parents if x not in allowed]})
    candidates = []
    if allow_external_premises:
        for index, candidate in enumerate(result.implicit_premises, 1):
            candidates.append({
                "id": f"ip_candidate_{index}",
                **candidate.model_dump(),
                "approved_for_proof": False,
                "formalization_source": "llm_premise_grounder",
            })
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
    return {
        "node_proposals": normalized,
        "implicit_premises": candidates,
        "api_call": {
            "response_id": str(getattr(response, "id", "")),
            "model": str(getattr(response, "model", model)),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "usage": usage_dict,
            "allow_external_premises": allow_external_premises,
        },
    }
