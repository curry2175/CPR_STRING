from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from .engine import ChainSupportEngine, KnowledgeItem
from .logic import Atom, Rule, formula_to_text
from .parser import parse_statement
from .profiler import build_reasoning_fingerprint
from .verifier import verify_case
from .proofwriter_logic import parse_raw_logic_program, normalize_reasoning_text


LABEL_ALIASES = {
    "a": "True",
    "a)": "True",
    "true": "True",
    "yes": "True",
    "entails": "True",
    "entailed": "True",
    "b": "False",
    "b)": "False",
    "false": "False",
    "no": "False",
    "contradicts": "False",
    "contradicted": "False",
    "c": "Unknown",
    "c)": "Unknown",
    "unknown": "Unknown",
    "both unknown": "Unknown",
    "undetermined": "Unknown",
}


def normalize_proofwriter_label(value: Any, field_name: str = "answer") -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value or "").strip().lower()
    text = re.sub(r"^[\s\"']+|[\s\"'.]+$", "", text)
    text = re.sub(r"^(?:answer\s*[:\-]\s*)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    if text in LABEL_ALIASES:
        return LABEL_ALIASES[text]
    match = re.match(r"^([abc])\)?(?:\s|$)", text, flags=re.I)
    if match:
        return {"a": "True", "b": "False", "c": "Unknown"}[match.group(1).lower()]
    raise ValueError(f"{field_name} must resolve to True, False, or Unknown; received: {value!r}")


def split_context(context: Any) -> list[dict[str, str]]:
    if isinstance(context, list):
        pieces = []
        for item in context:
            if isinstance(item, str):
                pieces.append(item.strip())
            elif isinstance(item, dict):
                pieces.append(str(item.get("text") or "").strip())
            else:
                raise ValueError(f"Unsupported context item: {type(item).__name__}")
    else:
        text = str(context or "").strip()
        if not text:
            raise ValueError("ProofWriter context is empty")
        pieces = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return [{"id": f"p{index}", "text": piece} for index, piece in enumerate(pieces, 1)]


def extract_query_statement(question: Any) -> str:
    text = str(question or "").strip()
    if not text:
        raise ValueError("ProofWriter question is empty")
    # Common ProofWriter wrapper: ... true, false, or unknown? Charlie is not red.
    match = re.search(r"(?:true\s*,\s*false\s*,\s*or\s*unknown|true\s+or\s+false)\s*\?\s*(.+)$", text, flags=re.I | re.S)
    if match:
        candidate = match.group(1).strip()
    elif "?" in text:
        candidate = text.rsplit("?", 1)[-1].strip()
    else:
        candidate = text
    candidate = candidate.strip().strip('"\'')
    candidate = re.sub(r"^(?:statement|claim)\s*:\s*", "", candidate, flags=re.I)
    if not candidate:
        raise ValueError("Could not extract the queried statement")
    if candidate[-1] not in ".?!":
        candidate += "."
    return candidate


def atom_to_question(atom: Atom) -> str:
    # Keep canonical underscore entities in the verifier-facing question so
    # compound names such as bald_eagle remain a single argument. The
    # Universal Graph viewer restores the original natural-language query.
    if len(atom.args) == 1:
        entity = atom.args[0]
        prop = atom.predicate
        return f"Is {entity} {'not ' if atom.negated else ''}{prop}?"
    if len(atom.args) == 2:
        subject = atom.args[0]
        obj = atom.args[1]
        return f"Does {subject} {'not ' if atom.negated else ''}{atom.predicate} {obj}?"
    raise ValueError("Only unary and binary atomic ProofWriter queries are supported")


def _clean_reasoning_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*(?:step\s*)?\d+\s*[.)\-:]\s*", "", line, flags=re.I)
    line = re.sub(r"^\s*[-*•]\s*", "", line)
    line = re.sub(r"^(?:reasoning|analysis|proof)\s*:\s*", "", line, flags=re.I)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def split_proofwriter_response(text: Any, explicit_answer: Any = None) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw and explicit_answer is None:
        raise ValueError("AI response is empty")
    answer: str | None = None
    answer_span: tuple[int, int] | None = None
    patterns = [
        r"(?i)(?:final\s+answer|answer|final)\s*[:\-]\s*([abc]\)?|true|false|unknown)\b[^\n]*$",
        r"(?im)^\s*(?:final\s+answer|answer|final)\s*[:\-]\s*([abc]\)?|true|false|unknown)\b.*$",
        r"(?im)^\s*([abc]\)?|true|false|unknown)\s*[.!]?\s*$",
        r"(?i)\b(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+(?:answer|statement)\s+is\s+)?(true|false|unknown)\b[.!]?\s*$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, raw):
            if answer_span is None or match.start() > answer_span[0]:
                answer = normalize_proofwriter_label(match.group(1), "predicted answer")
                answer_span = (match.start(), match.end())
    warnings: list[str] = []
    if explicit_answer is not None and str(explicit_answer).strip():
        explicit = normalize_proofwriter_label(explicit_answer, "predicted answer")
        if answer and answer != explicit:
            warnings.append(f"Response label {answer} differed from explicit label {explicit}; explicit label was used.")
        answer = explicit
    if answer is None:
        raise ValueError("Could not extract True, False, or Unknown from the AI response")

    reasoning_text = raw
    if answer_span:
        reasoning_text = (raw[: answer_span[0]] + "\n" + raw[answer_span[1] :]).strip()
    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    if len(lines) <= 1:
        lines = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", reasoning_text) if part.strip()]
    steps: list[dict[str, Any]] = []
    ignored: list[str] = []
    seen: set[str] = set()
    for line in lines:
        cleaned = _clean_reasoning_line(line)
        if not cleaned:
            continue
        # Keep common "X, so Y" responses atomic when both halves are parseable.
        candidates = [cleaned]
        if re.search(r",\s*(?:so|therefore|thus|hence)\s+", cleaned, flags=re.I):
            candidates = [part.strip() for part in re.split(r",\s*(?:so|therefore|thus|hence)\s+", cleaned, flags=re.I) if part.strip()]
        for candidate in candidates:
            if re.search(r"(?i)\b(?:final\s+answer|the\s+answer\s+is)\b", candidate):
                ignored.append(candidate)
                continue
            candidate = re.sub(r"^(?:therefore|thus|hence|consequently|so)\s*,?\s*", "", candidate, flags=re.I)
            candidate = candidate.strip()
            if candidate and candidate[-1] not in ".?!":
                candidate += "."
            key = candidate.lower()
            if not candidate or key in seen:
                continue
            seen.add(key)
            steps.append({"id": f"s{len(steps) + 1}", "text": candidate})
    return {
        "reasoning_steps": steps,
        "answer": answer,
        "extraction": {
            "strategy": "proofwriter_true_false_unknown",
            "step_count": len(steps),
            "ignored_segments": ignored,
            "warnings": warnings,
        },
    }


def _context_classification(premises: list[dict[str, str]], query: Atom) -> dict[str, Any]:
    items: list[KnowledgeItem] = []
    parse_errors: list[dict[str, str]] = []
    node_kinds: dict[str, str] = {}
    node_orders: dict[str, int] = {}
    for index, premise in enumerate(premises):
        parsed = parse_statement(premise["text"])
        if parsed.formula is None:
            parse_errors.append({"id": premise["id"], "text": premise["text"], "error": str(parsed.error)})
            continue
        items.append(KnowledgeItem(premise["id"], parsed.formula))
        node_kinds[premise["id"]] = "premise"
        node_orders[premise["id"]] = index
    if parse_errors:
        raise ValueError("ProofWriter context contains untranslatable statements: " + "; ".join(f"{x['id']}: {x['error']}" for x in parse_errors))
    path_engine = ChainSupportEngine(node_kinds, node_orders)
    positive_paths = path_engine.support_paths(items, query, max_paths=30)
    negative_paths = path_engine.support_paths(items, query.complement(), max_paths=30)
    if positive_paths and negative_paths:
        label = "Both/Inconsistent"
        dependencies = sorted(set(positive_paths[0]) | set(negative_paths[0]))
    elif positive_paths:
        label = "True"
        dependencies = min(positive_paths, key=lambda path: (len(path), path))
    elif negative_paths:
        label = "False"
        dependencies = min(negative_paths, key=lambda path: (len(path), path))
    else:
        label = "Unknown"
        dependencies = []
    return {
        "label": label,
        "query_entailed": bool(positive_paths),
        "opposite_entailed": bool(negative_paths),
        "selected_dependencies": list(dependencies),
        "query_proof_paths": positive_paths,
        "opposite_proof_paths": negative_paths,
        "engine": "finite_horn_open_world",
    }


def _map_binary_label(label: str) -> str:
    if label == "True":
        return "Yes"
    if label == "False":
        return "No"
    return "Yes"  # placeholder used only to verify reasoning nodes before Unknown final rewrite


def _prefer_compact_inferred_paths(graph: dict[str, Any]) -> None:
    """Use the shortest prior support path for ProofWriter graph display.

    The generic chain engine intentionally prefers using many prior CoT nodes.
    In cyclic rule sets this can select a valid but unnecessarily expanded path.
    ProofWriter profiling instead prefers the most compact acyclic candidate,
    then uses more prior reasoning nodes only as a tie-breaker.
    """
    nodes = graph.get("nodes") or []
    lookup = {str(node.get("id")): node for node in nodes}
    for node in nodes:
        if node.get("kind") not in {"reasoning", "answer"}:
            continue
        if node.get("chain_dependency_source") == "declared":
            continue
        paths = [list(map(str, path)) for path in node.get("candidate_reasoning_paths") or []]
        if not paths:
            continue
        def score(path: list[str]) -> tuple[int, int, tuple[str, ...]]:
            reasoning_count = sum(lookup.get(item, {}).get("kind") == "reasoning" for item in path)
            return (len(path), -reasoning_count, tuple(path))
        selected = min(paths, key=score)
        if node.get("proof_status") == "contradiction":
            node["reasoning_conflict_dependencies"] = selected
            node["reasoning_dependencies"] = []
        else:
            node["reasoning_dependencies"] = selected
            node["reasoning_conflict_dependencies"] = []
        node["dependencies"] = sorted(set(selected + list(node.get("semantic_normalizations") or []) + list(node.get("semantic_proof_dependencies") or [])))
        node["compact_inferred_path"] = selected
        node["chain_detail"] = (str(node.get("chain_detail") or "") + "; compact ProofWriter path selected").strip("; ")

    graph["edges"] = [
        edge for edge in graph.get("edges", [])
        if not (edge.get("target") in lookup and edge.get("relation") in {"reasoning_dependency", "reasoning_conflict"})
    ]
    for node in nodes:
        relation = "reasoning_conflict" if node.get("proof_status") == "contradiction" else "reasoning_dependency"
        deps = node.get("reasoning_conflict_dependencies") if relation == "reasoning_conflict" else node.get("reasoning_dependencies")
        for source in deps or []:
            if source in lookup and source != node.get("id"):
                graph["edges"].append({"source": source, "target": node.get("id"), "relation": relation})

    # Refresh chain reach metrics used by the UI and profiler.
    adjacency: dict[str, set[str]] = {}
    for edge in graph.get("edges", []):
        if edge.get("relation") not in {"source_match", "reasoning_dependency", "reasoning_conflict", "semantic_bridge", "semantic_normalization", "error_propagation"}:
            continue
        adjacency.setdefault(str(edge.get("source")), set()).add(str(edge.get("target")))
    for node in nodes:
        node_id = str(node.get("id"))
        seen: set[str] = set()
        stack = list(adjacency.get(node_id, set()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency.get(current, set()))
        node["chain_direct_dependents"] = len(adjacency.get(node_id, set()))
        node["chain_descendant_count"] = len(seen)
        node["chain_reaches_final"] = node_id == "final" or "final" in seen


def _rewrite_unknown_final(graph: dict[str, Any], query: Atom, context: dict[str, Any]) -> None:
    final = next((node for node in graph.get("nodes", []) if node.get("id") == "final"), None)
    if not final:
        return
    correct = context.get("label") == "Unknown"
    dependencies = list(context.get("selected_dependencies") or [])
    final.update({
        "text": "Answer: Unknown",
        "status": "valid" if correct else "contradiction",
        "proof_status": "valid" if correct else "contradiction",
        "chain_status": "valid" if correct else "contradiction",
        "formal": f"unknown({formula_to_text(query)})",
        "raw_formal": f"unknown({formula_to_text(query)})",
        "proof_dependencies": dependencies,
        "proof_dependencies_raw": dependencies,
        "reasoning_dependencies": [],
        "reasoning_conflict_dependencies": [],
        "dependencies": dependencies,
        "blocking_parent_nodes": [],
        "upstream_error_nodes": [] if correct else ["final"],
        "engine_detail": (
            "Neither the query nor its explicit opposite is derivable under open-world semantics"
            if correct else f"Unknown is contradicted because the context classifies the query as {context.get('label')}"
        ),
        "chain_detail": "Three-way classification final; no positive claim dependency is required",
        "chain_dependency_source": "three_way_classification",
        "candidate_reasoning_paths": [],
        "minimal_proof_paths": [],
        "minimal_proof_count": 0,
        "dependency_confidence": "classification_exact",
        "local_support_status": "three_way_unknown",
    })
    graph["edges"] = [edge for edge in graph.get("edges", []) if edge.get("target") != "final"]
    relation = "proof_conflict" if not correct else "classification_unknown"
    for source in dependencies:
        graph["edges"].append({"source": source, "target": "final", "relation": relation})
    summary = graph.setdefault("summary", {})
    summary["final_status"] = final["proof_status"]
    summary["final_proof_status"] = final["proof_status"]
    summary["final_chain_status"] = final["chain_status"]


def analyze_proofwriter(payload: dict[str, Any]) -> dict[str, Any]:
    record = payload.get("record", payload.get("case", payload))
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    canonical_program = parse_raw_logic_program(record)
    if canonical_program is not None:
        premises = canonical_program.premises
        query_statement = canonical_program.query_statement
        query = canonical_program.query
    else:
        premises = split_context(record.get("context", record.get("premises")))
        query_statement = extract_query_statement(record.get("question", record.get("query")))
        parsed_query = parse_statement(query_statement)
        if parsed_query.formula is None or not isinstance(parsed_query.formula, Atom):
            raise ValueError(f"ProofWriter query must be an atomic statement: {parsed_query.error}")
        query = parsed_query.formula
    question = atom_to_question(query)
    gold = normalize_proofwriter_label(record.get("answer", record.get("label", record.get("gold_answer"))), "gold answer")
    structured = payload.get("structured_llm_output")
    if structured is not None:
        if not isinstance(structured, dict):
            raise ValueError("structured_llm_output must be a JSON object")
        predicted = normalize_proofwriter_label(
            structured.get("answer", structured.get("final_answer")),
            "predicted answer",
        )
        raw_steps = structured.get("reasoning_steps") or []
        if not isinstance(raw_steps, list):
            raise ValueError("structured_llm_output.reasoning_steps must be a list")
        reasoning_steps = []
        for index, item in enumerate(raw_steps, 1):
            if not isinstance(item, dict):
                raise ValueError(f"structured reasoning step {index} must be an object")
            original_step_text = str(item.get("text") or "").strip()
            step = {
                "id": str(item.get("id") or f"s{index}"),
                "text": normalize_reasoning_text(original_step_text, canonical_program),
                "original_text": original_step_text,
            }
            if isinstance(item.get("depends_on"), list):
                step["depends_on"] = [str(x) for x in item.get("depends_on") or []]
            for metadata_key in (
                "model_declared_id",
                "model_declared_dependencies",
                "invalid_model_dependencies",
                "generation_source",
            ):
                if metadata_key in item:
                    step[metadata_key] = deepcopy(item[metadata_key])
            reasoning_steps.append(step)
        extracted = {
            "reasoning_steps": reasoning_steps,
            "answer": predicted,
            "answer_depends_on": [str(x) for x in structured.get("answer_depends_on") or []],
            "extraction": {
                "strategy": "openai_structured_output",
                "step_count": len(reasoning_steps),
                "ignored_segments": [],
                "warnings": [],
            },
        }
    else:
        response = payload.get("llm_response", record.get("llm_response", record.get("model_output", record.get("response", ""))))
        explicit = payload.get("predicted_answer", record.get("predicted_answer"))
        extracted = split_proofwriter_response(response, explicit_answer=explicit)
        predicted = extracted["answer"]
    context = _context_classification(premises, query)

    core_case = {
        "id": str(record.get("id") or "proofwriter_case"),
        "premises": premises,
        "question": question,
        "llm_output": {
            "reasoning_steps": extracted["reasoning_steps"],
            "answer": _map_binary_label(predicted),
            "answer_depends_on": list(extracted.get("answer_depends_on") or []),
        },
        "gold_answer": _map_binary_label(gold if gold != "Unknown" else predicted),
    }
    graph = verify_case(
        core_case,
        prefer_z3=bool(payload.get("prefer_z3", True)),
        compute_counterfactuals=bool(payload.get("compute_counterfactuals", False)),
    )
    if predicted == "Unknown":
        _rewrite_unknown_final(graph, query, context)
    _prefer_compact_inferred_paths(graph)
    graph["schema_version"] = "0.19.0"
    graph["predicted_answer"] = predicted
    graph["gold_answer"] = gold
    graph["answer_correct"] = predicted == gold
    graph["proofwriter"] = {
        "query_statement": query_statement,
        "query_formal": formula_to_text(query),
        "three_way_predicted_label": predicted,
        "three_way_gold_label": gold,
        "context_derived_label": context["label"],
        "dataset_label_matches_context": gold == context["label"],
        "prediction_matches_context": predicted == context["label"],
        "open_world_policy": "True if query is derivable; False if its explicit opposite is derivable; Unknown if neither is derivable.",
    }
    fingerprint = build_reasoning_fingerprint(
        graph,
        context_proof_dependencies=context.get("selected_dependencies") or [],
    )
    graph["reasoning_fingerprint"] = fingerprint

    return {
        "schema_version": "0.19.0",
        "source_format": "proofwriter",
        "record_id": str(record.get("id") or "proofwriter_case"),
        "adapter": {
            "premise_count": len(premises),
            "query_statement": query_statement,
            "generated_yes_no_question": question,
            "reasoning_step_count": len(extracted["reasoning_steps"]),
            "extraction": extracted["extraction"],
            "adapted_case": deepcopy(core_case),
        },
        "classification": {
            **context,
            "gold_label": gold,
            "predicted_label": predicted,
            "answer_correct": predicted == gold,
            "dataset_label_matches_context": gold == context["label"],
            "prediction_matches_context": predicted == context["label"],
        },
        "verified_graph": graph,
        "reasoning_fingerprint": fingerprint,
    }
