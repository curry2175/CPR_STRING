from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .hybrid_formalizer import formalize_proofwriter_record, hybrid_formalize_case
from .openai_runner import (
    DEFAULT_MODEL,
    GeneratedProofWriterOutput,
    _parsed_from_response,
    _usage_dict,
    generate_proofwriter_output,
    normalize_generated_output,
)
from .parser import parse_statement
from .premise_grounding import ground_failed_nodes
from .profiler import build_reasoning_fingerprint
from .proofwriter import (
    _context_classification,
    _map_binary_label,
    _prefer_compact_inferred_paths,
    _rewrite_unknown_final,
    atom_to_question,
    extract_query_statement,
    normalize_proofwriter_label,
    split_context,
)
from .universal_graph import build_universal_graph, graph_diff
from .verifier import verify_case
from .logic import Atom, formula_to_text
from .proofwriter_logic import parse_raw_logic_program, normalize_reasoning_text


class RepairOutput(GeneratedProofWriterOutput):
    repair_summary: str = Field(default="", description="Brief summary of what changed, without hidden chain-of-thought")


def _load_env() -> None:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() in {"OPENAI_API_KEY", "OPENAI_MODEL"} and v.strip():
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))


def _analyze_transformed(
    record: dict[str, Any],
    generated_llm_output: dict[str, Any],
    *,
    use_llm_formalizer: bool,
    model: str,
    reasoning_effort: str,
    prefer_z3: bool,
    client: Any,
    record_formalized: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_formalized = record_formalized or formalize_proofwriter_record(
        record, use_llm_fallback=use_llm_formalizer, model=model, reasoning_effort=reasoning_effort, client=client
    )
    unresolved = list((record_formalized.get("summary") or {}).get("unresolved_ids") or [])
    if unresolved:
        details = {node_id: (record_formalized.get("metadata") or {}).get(node_id, {}).get("formalization_blockers", []) for node_id in unresolved}
        raise ValueError(f"Formalization blocked for {unresolved}: {details}")
    premises = record_formalized["premises"]
    query_statement = record_formalized["query_statement"]
    canonical_program = parse_raw_logic_program(record)
    if canonical_program is not None:
        query = canonical_program.query
    else:
        query_parsed = parse_statement(query_statement)
        if query_parsed.formula is None or not isinstance(query_parsed.formula, Atom):
            raise ValueError(f"Query autoformalization failed: {query_parsed.error}")
        query = query_parsed.formula
    normalized_output = deepcopy(generated_llm_output)
    reasoning_originals: dict[str, str] = {}
    for index, step in enumerate(normalized_output.get("reasoning_steps") or [], 1):
        step_id = str(step.get("id") or f"s{index}")
        original_text = str(step.get("text") or "")
        reasoning_originals[step_id] = original_text
        step["text"] = normalize_reasoning_text(original_text, canonical_program)
    predicted = normalize_proofwriter_label(normalized_output.get("answer"), "generated answer")
    gold = normalize_proofwriter_label(record.get("answer", record.get("label", record.get("gold_answer"))), "gold answer")
    core_case = {
        "id": str(record.get("id") or "proofwriter_case"),
        "premises": premises,
        "question": atom_to_question(query),
        "llm_output": {
            "reasoning_steps": deepcopy(normalized_output.get("reasoning_steps") or []),
            "answer": _map_binary_label(predicted),
            "answer_depends_on": list(normalized_output.get("answer_depends_on") or []),
        },
        "gold_answer": _map_binary_label(gold if gold != "Unknown" else predicted),
        "semantic_relations": deepcopy(record.get("semantic_relations") or []),
        "input_mode": str(record.get("input_mode") or (record_formalized.get("summary") or {}).get("input_mode") or "controlled"),
        "verification_policy": {
            "require_declared_reasoning_dependencies": True,
            "require_declared_answer_dependencies": True,
        },
    }
    case_formalized = hybrid_formalize_case(
        core_case, use_llm_fallback=use_llm_formalizer, model=model, reasoning_effort=reasoning_effort, client=client
    )
    # Restore model-authored text in the viewer while verifying the canonicalized form.
    for step in case_formalized["case"].get("llm_output", {}).get("reasoning_steps", []):
        step_id = str(step.get("id"))
        original_text = reasoning_originals.get(step_id, str(step.get("text") or ""))
        info = case_formalized["metadata"].setdefault(step_id, {})
        info["original_text"] = original_text
        info["formalized_text"] = str(step.get("text") or "")
        if original_text != str(step.get("text") or "") and info.get("formalization_source") == "deterministic_parser":
            info["formalization_source"] = "proofwriter_vocabulary_normalization"
            info["formalization_confidence"] = "high"
            info["formalization_notes"] = "Known ProofWriter entity/predicate vocabulary was canonicalized before parsing."
    if record_formalized.get("metadata", {}).get("query_statement"):
        case_formalized["metadata"]["question"] = deepcopy(record_formalized["metadata"]["query_statement"])
    context = _context_classification(case_formalized["case"]["premises"], query)
    graph = verify_case(case_formalized["case"], prefer_z3=prefer_z3, compute_counterfactuals=False)
    if predicted == "Unknown":
        _rewrite_unknown_final(graph, query, context)
    _prefer_compact_inferred_paths(graph)
    graph["predicted_answer"] = predicted
    graph["gold_answer"] = gold
    graph["answer_correct"] = predicted == gold
    graph["proofwriter"] = {
        "query_statement": str(record.get("question", record.get("query")) or ""),
        "formalized_query_statement": query_statement,
        "query_formal": formula_to_text(query),
        "three_way_predicted_label": predicted,
        "three_way_gold_label": gold,
        "context_derived_label": context["label"],
        "dataset_label_matches_context": gold == context["label"],
        "prediction_matches_context": predicted == context["label"],
        "open_world_policy": "True if query is derivable; False if its explicit opposite is derivable; Unknown if neither is derivable.",
    }
    fingerprint = build_reasoning_fingerprint(graph, context_proof_dependencies=context.get("selected_dependencies") or [])
    graph["reasoning_fingerprint"] = fingerprint
    combined_metadata = {**case_formalized["metadata"], **record_formalized["metadata"]}
    universal = build_universal_graph(graph, formalization_metadata=combined_metadata)
    return {
        "record_formalization": record_formalized,
        "case_formalization": case_formalized,
        "adapted_case": case_formalized["case"],
        "classification": {
            **context,
            "gold_label": gold,
            "predicted_label": predicted,
            "answer_correct": predicted == gold,
            "dataset_label_matches_context": gold == context["label"],
            "prediction_matches_context": predicted == context["label"],
        },
        "verified_graph": graph,
        "universal_graph": universal,
        "reasoning_fingerprint": fingerprint,
    }


def _needs_repair(analysis: dict[str, Any]) -> bool:
    graph = analysis["verified_graph"]
    summary = graph.get("summary") or {}
    return bool(
        not analysis["classification"].get("prediction_matches_context")
        or summary.get("final_proof_status") != "valid"
        or summary.get("final_chain_status") != "valid"
        or int(summary.get("chain_error_count") or 0) > 0
        or int(summary.get("invalid_reasoning_count") or 0) > 0
    )


def build_repair_packet(analysis: dict[str, Any], *, mode: str = "blind") -> dict[str, Any]:
    graph = analysis["verified_graph"]
    root_errors = []
    for node in graph.get("nodes") or []:
        if node.get("kind") != "reasoning":
            continue
        if node.get("proof_status") in {"contradiction", "ungrounded", "untranslatable"} or node.get("chain_status") in {"insufficient_declared_support", "blocked_by_upstream_error"}:
            root_errors.append({
                "node_id": node.get("id"),
                "claim": node.get("text"),
                "proof_status": node.get("proof_status"),
                "chain_status": node.get("chain_status"),
                "error_type": node.get("reasoning_error_type") or node.get("chain_status") or node.get("proof_status"),
                "declared_parents": node.get("declared_reasoning_dependencies") or [],
                "inferred_parents": node.get("inferred_reasoning_dependencies") or [],
                "blocking_parents": node.get("blocking_parent_nodes") or [],
                "affected_nodes": [x.get("id") for x in graph.get("nodes") or [] if node.get("id") in (x.get("upstream_error_nodes") or [])],
            })
    packet = {
        "query": graph.get("proofwriter", {}).get("query_statement"),
        "previous_answer": analysis["classification"].get("predicted_label"),
        "context_label_is_hidden": True,
        "root_errors": root_errors,
        "constraints": [
            "Use only supplied context premises and earlier revised steps.",
            "Do not assume the previous answer is correct.",
            "Each step must be one atomic claim.",
            "List direct parent IDs only.",
            "For Unknown, do not invent a proof.",
        ],
        "mode": mode,
    }
    if mode == "guided":
        packet["verified_context_support"] = analysis["classification"].get("selected_dependencies") or []
        packet["query_proof_paths"] = analysis["classification"].get("query_proof_paths") or []
        packet["opposite_proof_paths"] = analysis["classification"].get("opposite_proof_paths") or []
    return packet


def _repair_call(
    record: dict[str, Any],
    previous_output: dict[str, Any],
    repair_packet: dict[str, Any],
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any = None,
) -> dict[str, Any]:
    _load_env()
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is required for self-reflection")
        from openai import OpenAI
        client = OpenAI()
    premises = split_context(record.get("context", record.get("premises")))
    query = extract_query_statement(record.get("question", record.get("query")))
    premise_block = "\n".join(f"{x['id']}: {x['text']}" for x in premises)
    system = (
        "You repair an explicit stated proof after neuro-symbolic verification. "
        "Use open-world semantics and explicit negation. The gold answer is not provided. "
        "Use only the context. Revise from the earliest root error and reconsider the final label. "
        "Return a concise inspectable proof, not private hidden chain-of-thought. "
        "Each step must be one atomic claim with direct parent IDs."
    )
    user = (
        f"Context:\n{premise_block}\n\nQuery: {query}\n\n"
        f"Previous structured output:\n{json.dumps(previous_output, ensure_ascii=False)}\n\n"
        f"Verifier repair packet ({repair_packet['mode']}):\n{json.dumps(repair_packet, ensure_ascii=False)}\n\n"
        "Return a revised proof and a newly reconsidered True/False/Unknown answer."
    )
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=RepairOutput,
    )
    parsed = _parsed_from_response(response)
    normalized = normalize_generated_output(parsed, {x["id"] for x in premises})
    return {
        "llm_output": normalized["llm_output"],
        "raw_structured_output": normalized["raw_structured_output"],
        "warnings": normalized["warnings"],
        "repair_summary": getattr(parsed, "repair_summary", ""),
        "response_id": str(getattr(response, "id", "")),
        "model": str(getattr(response, "model", model)),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "usage": _usage_dict(response),
        "prompt": {"system": system, "user": user, "gold_answer_was_sent": False},
    }


def run_hybrid_proofwriter(payload: dict[str, Any], *, client: Any = None) -> dict[str, Any]:
    record = payload.get("record", payload.get("case", payload))
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    model = str(payload.get("model") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)
    effort = str(payload.get("reasoning_effort") or "low")
    max_tokens = int(payload.get("max_output_tokens") or 5000)
    max_repairs = max(0, min(3, int(payload.get("max_repair_iterations", 1))))
    repair_mode = str(payload.get("repair_mode") or "blind").lower()
    if repair_mode not in {"blind", "guided"}:
        raise ValueError("repair_mode must be blind or guided")
    use_llm_formalizer = bool(payload.get("use_llm_formalizer", True))
    use_premise_grounder = bool(payload.get("use_premise_grounder", True))
    allow_external = bool(payload.get("allow_external_premises", False))
    prefer_z3 = bool(payload.get("prefer_z3", True))

    # Formalize and validate the context/query before spending an API call on
    # answer generation. A sentence that parses syntactically but has suspicious
    # semantics is either normalized/fallback-formalized or blocked explicitly.
    record_formalized = formalize_proofwriter_record(
        record,
        use_llm_fallback=use_llm_formalizer,
        model=model,
        reasoning_effort=effort,
        client=client,
    )
    unresolved = list((record_formalized.get("summary") or {}).get("unresolved_ids") or [])
    if unresolved:
        details = {
            node_id: (record_formalized.get("metadata") or {}).get(node_id, {}).get("formalization_blockers", [])
            for node_id in unresolved
        }
        raise ValueError(
            "Input formalization could not be validated. "
            f"Blocked items: {unresolved}. Details: {details}. "
            "Rewrite the item in controlled English or enable the LLM formalizer."
        )

    input_mode = str(record.get("input_mode") or (record_formalized.get("summary") or {}).get("input_mode") or "controlled")
    generation_record = record_formalized.get("formalized_record") if input_mode == "general_science" else record
    if not isinstance(generation_record, dict):
        generation_record = record

    generation = generate_proofwriter_output(
        generation_record,
        model=model,
        reasoning_effort=effort,
        max_output_tokens=max_tokens,
        custom_instruction=str(payload.get("custom_instruction") or ""),
        client=client,
    )
    attempts = []
    current_output = generation["llm_output"]
    prior_universal = None
    for iteration in range(max_repairs + 1):
        analysis = _analyze_transformed(
            record,
            current_output,
            use_llm_formalizer=use_llm_formalizer,
            model=model,
            reasoning_effort=effort,
            prefer_z3=prefer_z3,
            client=client,
            record_formalized=record_formalized,
        )
        grounding = {"node_proposals": [], "implicit_premises": [], "api_call": None}
        if use_premise_grounder:
            grounding = ground_failed_nodes(
                analysis["adapted_case"], analysis["verified_graph"], model=model,
                reasoning_effort=effort, allow_external_premises=allow_external, client=client,
            )
            analysis["universal_graph"] = build_universal_graph(
                analysis["verified_graph"],
                formalization_metadata={**analysis["case_formalization"]["metadata"], **analysis["record_formalization"]["metadata"]},
                inferred_premise_candidates=grounding["implicit_premises"],
                grounding_proposals=grounding["node_proposals"],
                repair_iteration=iteration,
            )
        attempt = {
            "iteration": iteration,
            "kind": "initial" if iteration == 0 else "repair",
            "llm_output": deepcopy(current_output),
            "analysis": analysis,
            "premise_grounding": grounding,
            "passed": not _needs_repair(analysis),
        }
        if prior_universal is not None:
            attempt["graph_diff_from_previous"] = graph_diff(prior_universal, analysis["universal_graph"])
        attempts.append(attempt)
        prior_universal = analysis["universal_graph"]
        if attempt["passed"] or iteration >= max_repairs:
            break
        packet = build_repair_packet(analysis, mode=repair_mode)
        repair = _repair_call(
            generation_record, current_output, packet, model=model, reasoning_effort=effort,
            max_output_tokens=max_tokens, client=client,
        )
        attempt["repair_packet"] = packet
        attempt["repair_call"] = repair
        current_output = repair["llm_output"]

    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    calls = [generation]
    if record_formalized.get("api_call"):
        calls.append(record_formalized["api_call"])
    for attempt in attempts:
        if attempt.get("repair_call"):
            calls.append(attempt["repair_call"])
        for source in (
            attempt["analysis"].get("case_formalization", {}).get("api_call"),
            attempt.get("premise_grounding", {}).get("api_call"),
        ):
            if source:
                calls.append(source)
    # A preformalized record is reused across repair iterations. Deduplicate
    # usage entries so the same fallback call is not counted multiple times.
    unique_calls = []
    seen_calls = set()
    for call in calls:
        key = str(call.get("response_id") or "") or f"object:{id(call)}"
        if key in seen_calls:
            continue
        seen_calls.add(key)
        unique_calls.append(call)
    calls = unique_calls
    for call in calls:
        usage = call.get("usage") or {}
        for key in total_usage:
            total_usage[key] += int(usage.get(key) or 0)
    final_attempt = attempts[-1]
    final_graph = final_attempt["analysis"]["universal_graph"]
    if len(attempts) > 1:
        full_diff = graph_diff(attempts[0]["analysis"]["universal_graph"], final_graph)
        changed_ids = {row["node_id"] for row in full_diff.get("changed_nodes", [])}
        added_ids = set(full_diff.get("added_nodes", []))
        for node in final_graph.get("nodes", []):
            node_id = str(node.get("id"))
            node["repair_status"] = "added" if node_id in added_ids else ("modified" if node_id in changed_ids else "unchanged")
        final_graph["repair_history"] = full_diff
    else:
        for node in final_graph.get("nodes", []):
            node["repair_status"] = "original"
    return {
        "schema_version": "0.23.0",
        "record_id": str(record.get("id") or "proofwriter_case"),
        "architecture": "Global-vocabulary-conditioned scientific formalization + symbol-drift/connectivity preflight + Hybrid VeriCoT-style repair + VRG graph verification",
        "settings": {
            "model": model,
            "reasoning_effort": effort,
            "use_llm_formalizer": use_llm_formalizer,
            "use_premise_grounder": use_premise_grounder,
            "allow_external_premises": allow_external,
            "repair_mode": repair_mode,
            "max_repair_iterations": max_repairs,
            "gold_answer_sent_to_model": False,
            "input_mode": input_mode,
            "generation_used_formalized_context": input_mode == "general_science",
            "formalization_preflight_passed": True,
        },
        "formalization_preflight": record_formalized,
        "initial_generation": generation,
        "attempts": attempts,
        "summary": {
            "attempt_count": len(attempts),
            "repair_count": max(0, len(attempts) - 1),
            "initial_pass": attempts[0]["passed"],
            "final_pass": final_attempt["passed"],
            "initial_answer": attempts[0]["analysis"]["classification"]["predicted_label"],
            "final_answer": final_attempt["analysis"]["classification"]["predicted_label"],
            "context_label": final_attempt["analysis"]["classification"]["label"],
            "gold_label": final_attempt["analysis"]["classification"]["gold_label"],
            "final_answer_correct": final_attempt["analysis"]["classification"]["answer_correct"],
            "final_context_match": final_attempt["analysis"]["classification"]["prediction_matches_context"],
            "total_usage": total_usage,
            "api_call_count": len(calls),
            "formalization_fallback_used": bool(record_formalized.get("api_call")),
            "formalization_source": (record_formalized.get("summary") or {}).get("source"),
        },
        "final_universal_graph": final_graph,
    }


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(str(text or "").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_no} must be an object")
        records.append(value)
    return records


def run_hybrid_batch(payload: dict[str, Any], *, client: Any = None) -> dict[str, Any]:
    records = payload.get("records")
    if not isinstance(records, list):
        records = parse_jsonl(str(payload.get("jsonl") or ""))
    results, errors = [], []
    for index, record in enumerate(records, 1):
        try:
            results.append(run_hybrid_proofwriter({**payload, "record": record}, client=client))
        except Exception as exc:  # visible per-record failure
            errors.append({"index": index, "record_id": record.get("id"), "error_type": type(exc).__name__, "error": str(exc)})
    return {
        "schema_version": "0.23.0",
        "summary": {
            "input_records": len(records),
            "completed_records": len(results),
            "failed_records": len(errors),
            "final_pass_count": sum(bool(x["summary"]["final_pass"]) for x in results),
            "repaired_count": sum(bool(x["summary"]["repair_count"] and x["summary"]["final_pass"]) for x in results),
            "total_input_tokens": sum(x["summary"]["total_usage"]["input_tokens"] for x in results),
            "total_output_tokens": sum(x["summary"]["total_usage"]["output_tokens"] for x in results),
        },
        "results": results,
        "errors": errors,
    }
