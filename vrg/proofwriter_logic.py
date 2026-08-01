from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .logic import Atom, Rule, Formula, formula_to_text
from .parser import parse_statement


class RawLogicError(ValueError):
    pass


@dataclass
class CanonicalProofWriterProgram:
    premises: list[dict[str, Any]]
    query: Atom
    query_statement: str
    metadata: dict[str, dict[str, Any]]
    predicate_map: dict[str, str]
    entity_map: dict[str, str]
    raw_program: str


def _camel_to_snake(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value.strip())
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_").lower()


def canonical_entity(value: str) -> str:
    value = value.strip()
    if value.startswith("$"):
        return "?" + re.sub(r"[^A-Za-z0-9_]", "", value[1:]).lower()
    return _camel_to_snake(value)


def canonical_predicate_name(name: str) -> str:
    value = _camel_to_snake(name)
    irregular = {
        "chases": "chase", "eats": "eat", "sees": "see", "needs": "need",
        "likes": "like", "visits": "visit", "has": "have", "does": "do",
        "flies": "fly", "tries": "try", "carries": "carry",
    }
    if value in irregular:
        return irregular[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    # Most ProofWriter relation names are third-person singular. Preserve nouns/adjectives.
    if value.endswith("ches") or value.endswith("shes") or value.endswith("xes") or value.endswith("zes") or value.endswith("oes") or value.endswith("sses"):
        return value[:-2]
    if value.endswith("s") and not value.endswith(("ss", "ous")) and len(value) > 3:
        return value[:-1]
    return value


def _split_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return [x for x in args if x]


def _parse_declared_predicate(logic: str, description: str) -> tuple[str, str]:
    match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*\((.*)\)$", logic.strip())
    if not match:
        raise RawLogicError(f"Invalid predicate declaration: {logic}")
    raw_name = match.group(1)
    args = _split_args(match.group(2))
    canonical = canonical_predicate_name(raw_name)
    desc = description.strip().rstrip("?").lower()
    if len(args) == 3:
        m = re.match(r"^does\s+\w+\s+(?:not\s+)?([a-z][a-z0-9_-]*)\s+\w+$", desc)
        if m:
            canonical = canonical_predicate_name(m.group(1))
    elif len(args) == 2:
        m = re.match(r"^is\s+\w+\s+(?:not\s+)?(.+)$", desc)
        if m:
            prop = re.sub(r"[^a-z0-9_-]+", "_", m.group(1)).strip("_")
            if prop:
                canonical = prop
    return raw_name, canonical


def _parse_atom(text: str, predicate_map: dict[str, str]) -> Atom:
    raw = text.strip()
    explicit_bang = raw.startswith("!")
    if explicit_bang:
        raw = raw[1:].strip()
    match = re.match(r"^([A-Za-z][A-Za-z0-9_]*)\s*\((.*)\)$", raw)
    if not match:
        raise RawLogicError(f"Invalid raw atom: {text}")
    raw_pred = match.group(1)
    args = _split_args(match.group(2))
    if not args or args[-1].lower() not in {"true", "false"}:
        raise RawLogicError(f"Raw atom lacks final boolean polarity: {text}")
    negated = (args[-1].lower() == "false") ^ explicit_bang
    terms = tuple(canonical_entity(x) for x in args[:-1])
    predicate = predicate_map.get(raw_pred, canonical_predicate_name(raw_pred))
    return Atom(predicate, terms, negated)


def _parse_formula(text: str, predicate_map: dict[str, str]) -> Formula:
    if ">>>" not in text:
        return _parse_atom(text, predicate_map)
    left, right = [x.strip() for x in text.split(">>>", 1)]
    antecedents = tuple(_parse_atom(x.strip(), predicate_map) for x in left.split("&&") if x.strip())
    if not antecedents:
        raise RawLogicError(f"Rule has no antecedent: {text}")
    return Rule(antecedents, _parse_atom(right, predicate_map))


def _entity_display(term: str) -> str:
    if term.startswith("?"):
        return "something"
    return term


def atom_to_controlled(atom: Atom) -> str:
    if len(atom.args) == 1:
        subject = _entity_display(atom.args[0])
        return f"{subject} is {'not ' if atom.negated else ''}{atom.predicate}."
    if len(atom.args) == 2:
        subject = _entity_display(atom.args[0])
        obj = _entity_display(atom.args[1])
        return f"{subject} {'does not ' if atom.negated else ''}{atom.predicate if atom.negated else atom.predicate + 's'} {obj}."
    raise RawLogicError(f"Unsupported arity in {formula_to_text(atom)}")


def formula_to_controlled(formula: Formula) -> str:
    if isinstance(formula, Atom):
        return atom_to_controlled(formula)
    antecedents = " and ".join(atom_to_controlled(x).rstrip(".") for x in formula.antecedents)
    consequent = atom_to_controlled(formula.consequent).rstrip(".")
    return f"If {antecedents} then {consequent}."


def _human_entity_map(record: dict[str, Any], constants: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical in constants:
        mapping[canonical.replace("_", " ")] = canonical
        mapping[canonical] = canonical
    # Raw CamelCase constants map naturally to their spaced form after canonicalization.
    for text in list(record.get("context") or []) + [str(record.get("question") or "")]:
        lower = str(text).lower()
        for canonical in constants:
            human = canonical.replace("_", " ")
            if human in lower:
                mapping[human] = canonical
    return mapping


def _normalize_with_entities(text: str, entity_map: dict[str, str]) -> str:
    normalized = str(text or "")
    for human, canonical in sorted(entity_map.items(), key=lambda x: len(x[0]), reverse=True):
        if " " not in human:
            continue
        normalized = re.sub(rf"\b{re.escape(human)}\b", canonical, normalized, flags=re.I)
    return normalized


def _alpha_normalize_formula(formula: Formula) -> Formula:
    mapping: dict[str, str] = {}
    def atom_norm(atom: Atom) -> Atom:
        args = []
        for arg in atom.args:
            if arg.startswith("?"):
                mapping.setdefault(arg, f"?v{len(mapping) + 1}")
                args.append(mapping[arg])
            else:
                args.append(arg)
        return Atom(atom.predicate, tuple(args), atom.negated)
    if isinstance(formula, Atom):
        return atom_norm(formula)
    return Rule(tuple(atom_norm(x) for x in formula.antecedents), atom_norm(formula.consequent))


def _formula_equal(left: Formula, right: Formula) -> bool:
    return _alpha_normalize_formula(left) == _alpha_normalize_formula(right)


def parse_raw_logic_program(record: dict[str, Any]) -> CanonicalProofWriterProgram | None:
    programs = record.get("raw_logic_programs")
    if not isinstance(programs, list) or not programs or not str(programs[0]).strip():
        return None
    raw_program = str(programs[0])
    sections: dict[str, list[tuple[str, str]]] = {"Predicates": [], "Facts": [], "Rules": [], "Query": []}
    section: str | None = None
    for raw_line in raw_program.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.rstrip(":") in sections and line.endswith(":"):
            section = line.rstrip(":")
            continue
        if section and ":::" in line:
            logic, natural = [x.strip() for x in line.split(":::", 1)]
            sections[section].append((logic, natural))

    predicate_map: dict[str, str] = {}
    for logic, natural in sections["Predicates"]:
        raw_name, canonical = _parse_declared_predicate(logic, natural)
        predicate_map[raw_name] = canonical

    raw_rows: list[tuple[Formula, str, str]] = []
    constants: set[str] = set()
    for section_name in ("Facts", "Rules"):
        for logic, natural in sections[section_name]:
            formula = _parse_formula(logic, predicate_map)
            raw_rows.append((formula, natural, logic))
            atoms = [formula] if isinstance(formula, Atom) else [*formula.antecedents, formula.consequent]
            for atom in atoms:
                constants.update(x for x in atom.args if not x.startswith("?"))
    if len(sections["Query"]) != 1:
        raise RawLogicError(f"Expected exactly one Query entry, found {len(sections['Query'])}")
    query_logic, query_natural = sections["Query"][0]
    raw_query = _parse_atom(query_logic, predicate_map)
    constants.update(x for x in raw_query.args if not x.startswith("?"))
    entity_map = _human_entity_map(record, constants)

    def norm_natural(value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().rstrip(".?!").lower())

    by_natural: dict[str, tuple[Formula, str, str]] = {}
    for row in raw_rows:
        by_natural.setdefault(norm_natural(row[1]), row)

    context = record.get("context") or []
    if isinstance(context, str):
        context = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", context) if x.strip()]
    if not context:
        context = [row[1] for row in raw_rows]

    premises: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for index, original in enumerate(context, 1):
        pid = f"p{index}"
        raw_match = by_natural.get(norm_natural(original))
        normalized_natural = _normalize_with_entities(str(original), entity_map)
        parsed = parse_statement(normalized_natural)
        natural_formula = parsed.formula
        mismatch = False
        if natural_formula is not None:
            formula = natural_formula
            if raw_match is not None:
                mismatch = not _formula_equal(natural_formula, raw_match[0])
                source = "proofwriter_raw_logic_verified" if not mismatch else "context_over_raw_mismatch"
                notes = raw_match[2] if not mismatch else f"Raw/natural mismatch; raw={formula_to_text(raw_match[0])}"
            else:
                source = "context_parser_raw_missing"
                notes = "Natural-language premise is absent from raw_logic_programs"
        elif raw_match is not None:
            formula = raw_match[0]
            source = "proofwriter_raw_logic_fallback"
            notes = f"Natural parse failed: {parsed.error}; raw={raw_match[2]}"
        else:
            raise RawLogicError(f"Could not formalize context premise {pid}: {original} ({parsed.error})")
        canonical_text = formula_to_controlled(formula)
        premises.append({
            "id": pid,
            "text": canonical_text,
            "canonical_formula": formula_to_text(formula),
            "raw_logic": raw_match[2] if raw_match else None,
            "premise_provenance": source,
        })
        metadata[pid] = {
            "original_text": str(original),
            "formalized_text": canonical_text,
            "formalization_source": source,
            "formalization_confidence": "exact" if source == "proofwriter_raw_logic_verified" else "high",
            "formalization_notes": notes,
            "new_vocabulary": [],
            "premise_provenance": source,
            "canonical_formula": formula_to_text(formula),
            "raw_natural_mismatch": mismatch,
        }

    original_query = str(record.get("question") or query_natural)
    normalized_query = _normalize_with_entities(original_query, entity_map)
    parsed_query = parse_statement(normalized_query)
    if isinstance(parsed_query.formula, Atom):
        query = parsed_query.formula
        query_mismatch = query != raw_query
        query_source = "proofwriter_raw_logic_verified" if not query_mismatch else "context_over_raw_mismatch"
        query_notes = query_logic if not query_mismatch else f"Raw/natural mismatch; raw={formula_to_text(raw_query)}"
    else:
        query = raw_query
        query_mismatch = False
        query_source = "proofwriter_raw_logic_fallback"
        query_notes = f"Natural parse failed: {parsed_query.error}; raw={query_logic}"
    query_statement = atom_to_controlled(query)
    metadata["query_statement"] = {
        "original_text": original_query,
        "formalized_text": query_statement,
        "formalization_source": query_source,
        "formalization_confidence": "exact" if query_source == "proofwriter_raw_logic_verified" else "high",
        "formalization_notes": query_notes,
        "new_vocabulary": [],
        "canonical_formula": formula_to_text(query),
        "raw_natural_mismatch": query_mismatch,
    }
    metadata["question"] = dict(metadata["query_statement"])
    return CanonicalProofWriterProgram(
        premises=premises,
        query=query,
        query_statement=query_statement,
        metadata=metadata,
        predicate_map=predicate_map,
        entity_map=entity_map,
        raw_program=raw_program,
    )

def normalize_reasoning_text(text: str, program: CanonicalProofWriterProgram | None) -> str:
    if program is None:
        return text
    original = str(text or "").strip()
    normalized = original
    # Exact source sentence -> exact canonical formula text.
    source_lookup = {
        str(info.get("original_text") or "").strip().rstrip(".?!").lower(): str(info.get("formalized_text") or "")
        for key, info in program.metadata.items() if key.startswith("p")
    }
    key = original.rstrip(".?!").lower()
    if key in source_lookup:
        return source_lookup[key]
    q_key = str(program.metadata["query_statement"].get("original_text") or "").strip().rstrip(".?!").lower()
    if key == q_key:
        return program.query_statement
    # Protect multiword entities before the generic controlled-English parser.
    for human, canonical in sorted(program.entity_map.items(), key=lambda x: len(x[0]), reverse=True):
        if " " not in human:
            continue
        normalized = re.sub(rf"\b{re.escape(human)}\b", canonical, normalized, flags=re.I)
    return normalized
