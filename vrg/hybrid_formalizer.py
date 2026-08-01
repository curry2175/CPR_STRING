from __future__ import annotations

import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .parser import parse_question, parse_statement
from .scientific_text import derive_label_type_premises, preview_scientific_text
from .symbol_alignment import align_item_texts, build_global_symbol_table


class FormalizationItem(BaseModel):
    id: str
    kind: Literal["premise", "reasoning", "question"]
    controlled_english: str = Field(description="A single atomic controlled-English statement or Yes/No question preserving the original meaning")
    confidence: Literal["high", "medium", "low"] = "medium"
    notes: str = ""
    new_vocabulary: list[str] = Field(default_factory=list)


class FormalizationBatch(BaseModel):
    items: list[FormalizationItem]


def _load_env(root: Path | None = None) -> None:
    root = root or Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"\'')
        if key in {"OPENAI_API_KEY", "OPENAI_MODEL"} and value:
            os.environ.setdefault(key, value)


def _parse_ok(kind: str, text: str) -> bool:
    parsed = parse_question(text) if kind == "question" else parse_statement(text)
    return parsed.formula is not None


def _response_parsed(response: Any) -> FormalizationBatch:
    value = getattr(response, "output_parsed", None)
    if value is not None:
        return value if isinstance(value, FormalizationBatch) else FormalizationBatch.model_validate(value)
    for output in getattr(response, "output", []) or []:
        for item in getattr(output, "content", []) or []:
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return parsed if isinstance(parsed, FormalizationBatch) else FormalizationBatch.model_validate(parsed)
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return FormalizationBatch.model_validate_json(text)
    raise ValueError("LLM formalizer returned no structured output")


def _call_formalizer(items: list[dict[str, str]], *, model: str, reasoning_effort: str, client: Any = None, vocabulary: dict[str, Any] | None = None) -> tuple[FormalizationBatch, dict[str, Any]]:
    _load_env()
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is required for LLM fallback formalization")
        from openai import OpenAI
        client = OpenAI()
    system = (
        "You are the fallback autoformalization component of a neuro-symbolic verifier. "
        "Rewrite each supplied item into the narrow controlled-English grammar accepted by a deterministic parser. "
        "Preserve entities, polarity, quantification, conjunctions, relation argument order, and epistemic strength exactly. "
        "Do not add facts, world knowledge, causal claims, or missing premises. Do not solve the task. "
        "Keep labelled multiword entities as one underscore token, for example Treatment A -> Treatment_A. "
        "Keep multiword concepts as one argument concept, for example fibrosis progression -> fibrosis_progression when needed. "
        "Rewrite an explicit relative universal such as 'All treatments that reduce X reduce Y' as "
        "'If something is a treatment and it reduces X, then it reduces Y.' "
        "Premises and reasoning items must be one atomic fact or one Horn-style rule. "
        "Questions must be a single Yes/No question beginning with Is/Are/Does/Do. "
        "Expressions such as may, suggests, is associated with, supports, and proves must not be upgraded to causes. "
        "If faithful rewriting is uncertain, use low confidence and explain why. "
        "A GLOBAL SYMBOL TABLE may be supplied. Reuse its existing predicates exactly. "
        "Do not invent near-duplicates such as observational_study when observational and study already exist. "
        "Represent modifier+head phrases compositionally: observational studies -> study(x) AND observational(x)."
    )
    vocabulary_text = ""
    if vocabulary:
        symbols = ", ".join(
            f"{row.get('predicate')}/{row.get('arity')}" for row in vocabulary.get("symbols", [])
        )
        vocabulary_text = f"Global symbols to reuse: {symbols or '(none)'}\n"
    user = vocabulary_text + "Items:\n" + "\n".join(f"{x['id']} | {x['kind']} | {x['text']}" for x in items)
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max(2000, 500 * len(items)),
        store=False,
        input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        text_format=FormalizationBatch,
    )
    parsed = _response_parsed(response)
    usage = getattr(response, "usage", None)
    usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
    return parsed, {
        "response_id": str(getattr(response, "id", "")),
        "model": str(getattr(response, "model", model)),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "usage": usage_dict,
        "stage": "formalization_fallback",
    }


def _preview_metadata(node_id: str, text: str, *, kind: str, mode: str, provenance: str | None = None) -> tuple[dict[str, Any], str, bool]:
    preview_kind = "query_statement" if kind == "query_statement" else kind
    preview = preview_scientific_text(text, kind=preview_kind, mode=mode)
    clean = not preview.blocking_warnings and preview.parse_ok
    source = "scientific_normalizer" if clean and preview.normalized_text != text else ("deterministic_parser" if clean else "semantic_anomaly")
    info = {
        "original_text": text,
        "normalized_text": preview.normalized_text,
        "formalized_text": preview.normalized_text if clean else text,
        "formalization_source": source,
        "formalization_confidence": "high" if clean else "none",
        "formalization_notes": "; ".join(preview.warnings + preview.blocking_warnings),
        "new_vocabulary": [],
        "formalization_warnings": preview.warnings,
        "formalization_blockers": preview.blocking_warnings,
        "normalization_steps": preview.normalization_steps,
        "parse_preview": preview.formal,
        "parse_formula_type": preview.formula_type,
        "requires_llm_fallback": preview.needs_llm_fallback,
    }
    if provenance is not None:
        info["premise_provenance"] = provenance
    return info, preview.normalized_text, clean


def _apply_llm_candidate(
    *,
    failure: dict[str, str],
    candidate: FormalizationItem,
    metadata: dict[str, dict[str, Any]],
) -> tuple[str | None, bool]:
    rewritten = re.sub(r"\s+", " ", candidate.controlled_english).strip()
    validation_kind = "question" if failure["kind"] == "question" else "premise"
    preview = preview_scientific_text(rewritten, kind=validation_kind, mode="controlled")
    if not preview.parse_ok or preview.blocking_warnings:
        metadata[failure["id"]].update({
            "formalization_source": "llm_fallback_failed",
            "formalization_confidence": candidate.confidence,
            "formalization_notes": "; ".join([candidate.notes, *preview.blocking_warnings]).strip("; "),
            "new_vocabulary": candidate.new_vocabulary,
            "formalization_blockers": preview.blocking_warnings,
            "parse_preview": preview.formal,
        })
        return None, False
    metadata[failure["id"]].update({
        "normalized_text": rewritten,
        "formalized_text": rewritten,
        "formalization_source": "llm_fallback",
        "formalization_confidence": candidate.confidence,
        "formalization_notes": candidate.notes,
        "new_vocabulary": candidate.new_vocabulary,
        "formalization_blockers": [],
        "requires_llm_fallback": False,
        "parse_preview": preview.formal,
        "parse_formula_type": preview.formula_type,
    })
    return rewritten, True


def _align_collections(
    *,
    premises: list[dict[str, Any]],
    reasoning_steps: list[dict[str, Any]] | None = None,
    query_text: str | None = None,
    query_id: str = "query_statement",
    query_kind: str = "query_statement",
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in premises:
        items.append({"id": str(row.get("id")), "kind": "premise", "text": str(row.get("text") or "")})
    for row in reasoning_steps or []:
        items.append({"id": str(row.get("id")), "kind": "premise", "text": str(row.get("text") or "")})
    if query_text is not None:
        items.append({"id": query_id, "kind": query_kind, "text": query_text})
    aligned = align_item_texts(items)
    by_id = {str(x.get("id")): x for x in aligned["items"]}
    for row in premises:
        node_id = str(row.get("id"))
        new_text = str((by_id.get(node_id) or {}).get("text") or row.get("text") or "")
        if new_text != str(row.get("text") or ""):
            old_text = str(row.get("text") or "")
            row["text"] = new_text
            info = metadata.setdefault(node_id, {})
            info["pre_alignment_text"] = old_text
            info["formalized_text"] = new_text
            parsed = parse_statement(new_text)
            info["parse_preview"] = parsed.formula.to_text() if parsed.formula is not None else None
            info["parse_formula_type"] = "rule" if getattr(parsed.formula, "antecedents", None) is not None else "atom"
            info["global_symbol_alignment"] = True
            info["formalization_source"] = "global_symbol_alignment"
            info["formalization_notes"] = (str(info.get("formalization_notes") or "") + "; predicate aliases aligned across the full context").strip("; ")
    for row in reasoning_steps or []:
        node_id = str(row.get("id"))
        new_text = str((by_id.get(node_id) or {}).get("text") or row.get("text") or "")
        if new_text != str(row.get("text") or ""):
            old_text = str(row.get("text") or "")
            row["text"] = new_text
            info = metadata.setdefault(node_id, {})
            info["pre_alignment_text"] = old_text
            info["formalized_text"] = new_text
            parsed = parse_statement(new_text)
            info["parse_preview"] = parsed.formula.to_text() if parsed.formula is not None else None
            info["parse_formula_type"] = "rule" if getattr(parsed.formula, "antecedents", None) is not None else "atom"
            info["global_symbol_alignment"] = True
            info["formalization_source"] = "global_symbol_alignment"
            info["formalization_notes"] = (str(info.get("formalization_notes") or "") + "; predicate aliases aligned across the full context").strip("; ")
    aligned_query = query_text
    if query_text is not None and query_id in by_id:
        aligned_query = str(by_id[query_id].get("text") or query_text)
        if aligned_query != query_text:
            info = metadata.setdefault(query_id, {})
            info["pre_alignment_text"] = query_text
            info["formalized_text"] = aligned_query
            parsed = parse_statement(aligned_query) if query_kind == "query_statement" else parse_question(aligned_query)
            info["parse_preview"] = parsed.formula.to_text() if parsed.formula is not None else None
            info["parse_formula_type"] = "rule" if getattr(parsed.formula, "antecedents", None) is not None else "atom"
            info["global_symbol_alignment"] = True
            info["formalization_source"] = "global_symbol_alignment"
            info["formalization_notes"] = (str(info.get("formalization_notes") or "") + "; query predicate aligned with the global symbol table").strip("; ")
    for decision in aligned.get("alignment_decisions") or []:
        info = metadata.setdefault(str(decision.get("item_id")), {})
        info.setdefault("symbol_alignment_decisions", []).append(decision)
    return {**aligned, "query_text": aligned_query}


def hybrid_formalize_case(
    case: dict[str, Any],
    *,
    use_llm_fallback: bool = True,
    model: str | None = None,
    reasoning_effort: str = "low",
    client: Any = None,
) -> dict[str, Any]:
    """Deterministic/scientific-normalizer first, semantic-anomaly aware.

    Items that merely *parse* but exhibit suspicious semantic structure are not
    silently accepted. They are routed to the controlled-English LLM fallback.
    """
    model = str(model or os.getenv("OPENAI_MODEL") or "gpt-5.6")
    mode = str(case.get("input_mode") or "controlled")
    transformed = {**case}
    transformed["premises"] = [dict(x) if isinstance(x, dict) else {"id": f"p{i}", "text": str(x)} for i, x in enumerate(case.get("premises") or [], 1)]
    output = dict(case.get("llm_output") or {})
    output["reasoning_steps"] = [dict(x) if isinstance(x, dict) else {"id": f"s{i}", "text": str(x)} for i, x in enumerate(output.get("reasoning_steps") or [], 1)]
    transformed["llm_output"] = output

    metadata: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    for item in transformed["premises"]:
        node_id, text = str(item.get("id")), str(item.get("text") or "")
        info, normalized, clean = _preview_metadata(node_id, text, kind="premise", mode=mode, provenance=item.get("premise_provenance", "context"))
        metadata[node_id] = info
        if clean:
            item["text"] = normalized
        else:
            failures.append({"id": node_id, "kind": "premise", "text": text})
    for item in transformed["llm_output"]["reasoning_steps"]:
        node_id, text = str(item.get("id")), str(item.get("text") or "")
        info, normalized, clean = _preview_metadata(node_id, text, kind="reasoning", mode=mode, provenance="prior_reasoning")
        metadata[node_id] = info
        if clean:
            item["text"] = normalized
        else:
            failures.append({"id": node_id, "kind": "reasoning", "text": text})
    question = str(transformed.get("question") or "")
    info, normalized_question, clean_question = _preview_metadata("question", question, kind="question", mode=mode)
    metadata["question"] = info
    if clean_question:
        transformed["question"] = normalized_question
    else:
        failures.append({"id": "question", "kind": "question", "text": question})

    vocabulary_items = [
        {"id": str(x.get("id")), "kind": "premise", "text": str(x.get("text") or "")}
        for x in transformed["premises"]
        if not (metadata.get(str(x.get("id"))) or {}).get("formalization_blockers")
    ] + [
        {"id": str(x.get("id")), "kind": "premise", "text": str(x.get("text") or "")}
        for x in transformed["llm_output"]["reasoning_steps"]
        if not (metadata.get(str(x.get("id"))) or {}).get("formalization_blockers")
    ]
    if clean_question:
        vocabulary_items.append({"id": "question", "kind": "question", "text": transformed["question"]})
    vocabulary = build_global_symbol_table(vocabulary_items)
    api_call = None
    if failures and use_llm_fallback:
        batch, api_call = _call_formalizer(failures, model=model, reasoning_effort=reasoning_effort, client=client, vocabulary=vocabulary)
        by_id = {x.id: x for x in batch.items}
        for failure in failures:
            candidate = by_id.get(failure["id"])
            if not candidate:
                continue
            rewritten, ok = _apply_llm_candidate(failure=failure, candidate=candidate, metadata=metadata)
            if not ok or rewritten is None:
                continue
            if failure["id"] == "question":
                transformed["question"] = rewritten
            else:
                collection = transformed["premises"] if failure["kind"] == "premise" else transformed["llm_output"]["reasoning_steps"]
                for item in collection:
                    if str(item.get("id")) == failure["id"]:
                        item["text"] = rewritten
                        break

    alignment = _align_collections(
        premises=transformed["premises"],
        reasoning_steps=transformed["llm_output"]["reasoning_steps"],
        query_text=transformed["question"],
        query_id="question",
        query_kind="question",
        metadata=metadata,
    )
    transformed["question"] = alignment["query_text"]
    unresolved = [node_id for node_id, info in metadata.items() if info["formalization_source"] in {"semantic_anomaly", "unresolved", "llm_fallback_failed"}]
    if alignment["diagnostics"].get("blocking_symbol_drift"):
        unresolved.append("global_symbol_table")
    unresolved = list(dict.fromkeys(unresolved))
    return {
        "case": transformed,
        "metadata": metadata,
        "summary": {
            "input_mode": mode,
            "item_count": len(metadata),
            "deterministic_count": sum(x["formalization_source"] == "deterministic_parser" for x in metadata.values()),
            "scientific_normalizer_count": sum(x["formalization_source"] == "scientific_normalizer" for x in metadata.values()),
            "llm_fallback_count": sum(x["formalization_source"] == "llm_fallback" for x in metadata.values()),
            "semantic_anomaly_count": sum(x["formalization_source"] == "semantic_anomaly" for x in metadata.values()),
            "unresolved_count": len(unresolved),
            "unresolved_ids": unresolved,
            "global_symbol_table": alignment["symbol_table"],
            "symbol_alignment_decisions": alignment["alignment_decisions"],
            "connectivity": alignment["diagnostics"],
        },
        "api_call": api_call,
    }


def formalize_proofwriter_record(
    record: dict[str, Any],
    *,
    use_llm_fallback: bool = True,
    model: str | None = None,
    reasoning_effort: str = "low",
    client: Any = None,
) -> dict[str, Any]:
    """Formalize context/query once before generation and verification.

    ProofWriter records keep their raw-logic cross-validation path. Individual
    Test Lab records use semantic-anomaly-aware scientific normalization.
    """
    from .proofwriter_logic import parse_raw_logic_program

    canonical = parse_raw_logic_program(record)
    if canonical is not None:
        transformed = deepcopy(record)
        transformed["context"] = [item["text"] for item in canonical.premises]
        transformed["question"] = canonical.query_statement
        counts: dict[str, int] = {}
        for info in canonical.metadata.values():
            source = str(info.get("formalization_source") or "unknown")
            counts[source] = counts.get(source, 0) + 1
        return {
            "record": deepcopy(record),
            "formalized_record": transformed,
            "premises": deepcopy(canonical.premises),
            "query_statement": canonical.query_statement,
            "query_formal": canonical.query.to_text(),
            "metadata": deepcopy(canonical.metadata),
            "summary": {
                "source": "proofwriter_context_plus_raw_logic_validation",
                "input_mode": "proofwriter_canonical",
                "source_counts": counts,
                "raw_natural_mismatch_count": sum(bool(x.get("raw_natural_mismatch")) for x in canonical.metadata.values()),
                "unresolved_ids": [],
            },
            "api_call": None,
        }

    from .proofwriter import extract_query_statement, split_context

    model = str(model or os.getenv("OPENAI_MODEL") or "gpt-5.6")
    mode = str(record.get("input_mode") or ("general_science" if record.get("source") == "individual_test_lab" else "controlled"))
    premises = split_context(record.get("context", record.get("premises")))
    query = extract_query_statement(record.get("question", record.get("query")))
    metadata: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    original_context_texts = [str(item["text"]) for item in premises]
    derived_type_rows = derive_label_type_premises([*original_context_texts, query]) if mode == "general_science" else []

    for item in premises:
        text = str(item["text"])
        info, normalized, clean = _preview_metadata(item["id"], text, kind="premise", mode=mode, provenance="context")
        metadata[item["id"]] = info
        if clean:
            item["text"] = normalized
        else:
            failures.append({"id": item["id"], "kind": "premise", "text": text})

    for derived in derived_type_rows:
        premise_id = f"p{len(premises) + 1}"
        statement = derived["statement"]
        preview = preview_scientific_text(statement, kind="premise", mode="controlled")
        premises.append({"id": premise_id, "text": statement, "premise_provenance": "lexical_entity_type"})
        metadata[premise_id] = {
            "original_text": statement,
            "normalized_text": statement,
            "formalized_text": statement,
            "formalization_source": "lexical_type_inference",
            "formalization_confidence": "high",
            "formalization_notes": f"Transparent type premise derived from labelled entity {derived['source_text']!r}.",
            "new_vocabulary": [derived["entity_type"]],
            "formalization_warnings": [],
            "formalization_blockers": [],
            "normalization_steps": [f"lexical_type:{derived['source_text']}->{statement}"],
            "parse_preview": preview.formal,
            "parse_formula_type": preview.formula_type,
            "requires_llm_fallback": False,
            "premise_provenance": "lexical_entity_type",
            "derived": True,
            "derived_from": derived["source_text"],
        }

    info, normalized_query, clean_query = _preview_metadata("query_statement", query, kind="query_statement", mode=mode)
    metadata["query_statement"] = info
    if clean_query:
        query = normalized_query
    else:
        failures.append({"id": "query_statement", "kind": "premise", "text": query})

    vocabulary_items = [
        {"id": str(x.get("id")), "kind": "premise", "text": str(x.get("text") or "")}
        for x in premises
        if not (metadata.get(str(x.get("id"))) or {}).get("formalization_blockers")
    ]
    if clean_query:
        vocabulary_items.append({"id": "query_statement", "kind": "query_statement", "text": query})
    vocabulary = build_global_symbol_table(vocabulary_items)
    api_call = None
    if failures and use_llm_fallback:
        batch, api_call = _call_formalizer(failures, model=model, reasoning_effort=reasoning_effort, client=client, vocabulary=vocabulary)
        by_id = {x.id: x for x in batch.items}
        for failure in failures:
            candidate = by_id.get(failure["id"])
            if not candidate:
                continue
            rewritten, ok = _apply_llm_candidate(failure=failure, candidate=candidate, metadata=metadata)
            if not ok or rewritten is None:
                continue
            if failure["id"] == "query_statement":
                query = rewritten
            else:
                for item in premises:
                    if item["id"] == failure["id"]:
                        item["text"] = rewritten
                        break

    alignment = _align_collections(
        premises=premises,
        query_text=query,
        query_id="query_statement",
        query_kind="query_statement",
        metadata=metadata,
    )
    query = str(alignment["query_text"] or query)
    transformed = dict(record)
    transformed["context"] = [item["text"] for item in premises]
    transformed["question"] = query
    unresolved = [x for x, info in metadata.items() if info["formalization_source"] in {"semantic_anomaly", "unresolved", "llm_fallback_failed"}]
    if alignment["diagnostics"].get("blocking_symbol_drift"):
        unresolved.append("global_symbol_table")
    unresolved = list(dict.fromkeys(unresolved))
    return {
        "record": deepcopy(record),
        "formalized_record": transformed,
        "premises": premises,
        "query_statement": query,
        "metadata": metadata,
        "summary": {
            "source": "semantic_anomaly_aware_hybrid_formalization",
            "input_mode": mode,
            "deterministic_count": sum(x["formalization_source"] == "deterministic_parser" for x in metadata.values()),
            "scientific_normalizer_count": sum(x["formalization_source"] == "scientific_normalizer" for x in metadata.values()),
            "llm_fallback_count": sum(x["formalization_source"] == "llm_fallback" for x in metadata.values()),
            "semantic_anomaly_count": sum(x["formalization_source"] == "semantic_anomaly" for x in metadata.values()),
            "derived_premise_count": sum(x["formalization_source"] == "lexical_type_inference" for x in metadata.values()),
            "global_symbol_count": len(alignment["symbol_table"].get("symbols") or []),
            "symbol_alignment_count": len(alignment["alignment_decisions"]),
            "global_symbol_table": alignment["symbol_table"],
            "symbol_alignment_decisions": alignment["alignment_decisions"],
            "connectivity": alignment["diagnostics"],
            "unresolved_ids": unresolved,
        },
        "api_call": api_call,
    }
