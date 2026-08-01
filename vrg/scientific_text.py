from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Literal

from .logic import Atom, Formula, Rule, formula_to_text
from .parser import parse_question, parse_statement


InputKind = Literal["premise", "reasoning", "question", "query_statement"]

# Common labelled entities in articles and scientific prose. The list is
# intentionally conservative: only label-like two-token names are collapsed.
_LABELLED_ENTITY = re.compile(
    r"\b(Treatment|Drug|Intervention|Therapy|Regimen|Group|Arm|Cohort|Study|Model|Agent|Compound|Protocol|Strategy)\s+([A-Z0-9][A-Za-z0-9-]*)\b"
)
_UNDERSCORED_LABELLED_ENTITY = re.compile(
    r"\b(Treatment|Drug|Intervention|Therapy|Regimen|Group|Arm|Cohort|Study|Model|Agent|Compound|Protocol|Strategy)_([A-Z0-9][A-Za-z0-9-]*)\b",
    flags=re.I,
)

# Verbs commonly used in compact scientific causal/mechanistic rules.
_SCIENTIFIC_VERBS = (
    "reduce|reduces|reduced|decrease|decreases|decreased|lower|lowers|lowered|"
    "increase|increases|increased|raise|raises|raised|improve|improves|improved|"
    "worsen|worsens|worsened|slow|slows|slowed|accelerate|accelerates|accelerated|"
    "prevent|prevents|prevented|inhibit|inhibits|inhibited|promote|promotes|promoted|"
    "cause|causes|caused|predict|predicts|predicted|indicate|indicates|indicated|"
    "mediate|mediates|mediated|affect|affects|affected|visit|visits|like|likes|"
    "chase|chases|see|sees|need|needs|use|uses|eat|eats"
)
_RELATIVE_RULE = re.compile(
    rf"^(?:all|any)\s+(?P<class>[a-z][a-z0-9_ -]*?)\s+that\s+"
    rf"(?P<v1>{_SCIENTIFIC_VERBS})\s+(?P<obj1>.+?)\s+"
    rf"(?P<v2>{_SCIENTIFIC_VERBS})\s+(?P<obj2>.+)$",
    flags=re.I,
)

_HEAD_NOUNS = (
    "studies|patients|treatments|drugs|interventions|therapies|regimens|groups|arms|"
    "cohorts|models|agents|compounds|protocols|strategies|trials|cells|animals"
)
_ADJECTIVAL_HEAD_UNIVERSAL = re.compile(
    rf"^(?:all|any|every)\s+(?P<descriptor>[a-z][a-z0-9_ -]*?)\s+(?P<head>{_HEAD_NOUNS})\s+(?:is|are)\s+(?P<consequent>.+)$",
    flags=re.I,
)


def rewrite_adjectival_head_universal(text: str) -> tuple[str, list[str]]:
    """Rewrite ``All observational studies are limited`` without symbol drift.

    Modifier and head noun become separate unary predicates so later sentences
    reuse ``observational`` rather than inventing ``observational_study``.
    """
    body, terminal = _clean_terminal(text)
    match = _ADJECTIVAL_HEAD_UNIVERSAL.match(body)
    if not match:
        return text, []
    descriptor = match.group("descriptor").strip()
    head = _singularize(match.group("head"))
    consequent = match.group("consequent").strip()
    rewritten = (
        f"If something is a {head} and it is {descriptor}, "
        f"then it is {consequent}."
    )
    return rewritten, ["modifier_head_rule:decomposed_into_type_and_property"]


_MODAL_OR_EPISTEMIC = re.compile(
    r"\b(may|might|could|possibly|probably|suggests?|appears?|seems?|"
    r"is associated with|are associated with|correlates? with|supports?|proves?)\b",
    flags=re.I,
)


@dataclass
class ScientificPreview:
    kind: str
    original_text: str
    normalized_text: str
    parse_ok: bool
    formal: str | None
    formula_type: str | None
    warnings: list[str]
    blocking_warnings: list[str]
    needs_llm_fallback: bool
    normalization_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_terminal(text: str) -> tuple[str, str]:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    terminal = "." if value.endswith(".") else ("?" if value.endswith("?") else "")
    return value.rstrip(".?!").strip(), terminal


def _singularize(noun_phrase: str) -> str:
    value = noun_phrase.strip().lower().replace("-", "_").replace(" ", "_")
    irregular = {"therapies": "therapy", "studies": "study", "people": "person"}
    if value in irregular:
        return irregular[value]
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("ses") and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def protect_labelled_entities(text: str) -> tuple[str, list[str]]:
    steps: list[str] = []

    def repl(match: re.Match[str]) -> str:
        original = match.group(0)
        replacement = f"{match.group(1)}_{match.group(2)}"
        if original != replacement:
            steps.append(f"labelled_entity:{original}->{replacement}")
        return replacement

    return _LABELLED_ENTITY.sub(repl, text), steps


def rewrite_relative_scientific_rule(text: str) -> tuple[str, list[str]]:
    body, terminal = _clean_terminal(text)
    match = _RELATIVE_RULE.match(body)
    if not match:
        return text, []
    klass = _singularize(match.group("class"))
    rewritten = (
        f"If something is a {klass} and it {match.group('v1')} {match.group('obj1')}, "
        f"then it {match.group('v2')} {match.group('obj2')}."
    )
    return rewritten, ["relative_clause_rule:rewritten_to_explicit_if_then"]


def normalize_scientific_text(text: str, *, kind: InputKind = "premise") -> tuple[str, list[str]]:
    """Conservatively normalize common article/discussion sentence shapes.

    The function never adds a new scientific claim. It only makes entity
    boundaries and an already-explicit universal relative rule parseable.
    """
    value = re.sub(r"\s+", " ", str(text or "").strip())
    steps: list[str] = []
    value, entity_steps = protect_labelled_entities(value)
    steps.extend(entity_steps)
    if kind != "question":
        value, rule_steps = rewrite_relative_scientific_rule(value)
        steps.extend(rule_steps)
        value, head_steps = rewrite_adjectival_head_universal(value)
        steps.extend(head_steps)
    return value, steps


def _formula_shape(formula: Formula | None) -> tuple[str | None, str | None]:
    if formula is None:
        return None, None
    return formula_to_text(formula), "rule" if isinstance(formula, Rule) else "atom"


def _semantic_warnings(original: str, normalized: str, formula: Formula | None, kind: InputKind) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    blocking: list[str] = []
    lower = original.lower()

    if re.match(r"^\s*(all|any|every)\b", original, flags=re.I) and not isinstance(formula, Rule):
        blocking.append("universal_sentence_was_not_parsed_as_a_rule")
    if " that " in lower and not isinstance(formula, Rule):
        blocking.append("relative_clause_was_not_parsed_as_a_rule")
    if _LABELLED_ENTITY.search(original) and normalized == original:
        blocking.append("multiword_labelled_entity_not_preserved")
    if isinstance(formula, Atom):
        if formula.predicate in {"a", "an", "the", "all", "any", "every"}:
            blocking.append(f"suspicious_predicate:{formula.predicate}")
        if any(arg in {"all", "any", "every"} for arg in formula.args):
            blocking.append("quantifier_was_parsed_as_an_entity")
        if any("_that_" in arg for arg in formula.args):
            blocking.append("relative_clause_was_absorbed_into_an_entity")
        if " not " in f" {lower} " and not formula.negated:
            blocking.append("negation_may_have_been_lost")
    if _MODAL_OR_EPISTEMIC.search(original):
        # These expressions require typed relations (supports/association/modal
        # strength) rather than being silently flattened into ordinary facts.
        warnings.append("epistemic_or_modal_language_requires_careful_formalization")
        if kind in {"premise", "reasoning", "query_statement"}:
            blocking.append("epistemic_or_modal_relation_requires_llm_fallback")
    # A deterministic parse of a labelled multiword subject should retain the
    # joined entity in at least one argument.
    labelled = _LABELLED_ENTITY.search(original)
    if labelled and isinstance(formula, Atom):
        expected = f"{labelled.group(1)}_{labelled.group(2)}".lower().replace("-", "_")
        if expected not in formula.args:
            blocking.append("labelled_entity_boundary_changed_during_parse")
    return list(dict.fromkeys(warnings)), list(dict.fromkeys(blocking))


def preview_scientific_text(text: str, *, kind: InputKind = "premise", mode: str = "general_science") -> ScientificPreview:
    original = str(text or "").strip()
    normalized, steps = normalize_scientific_text(original, kind=kind) if mode == "general_science" else (original, [])
    parsed = parse_question(normalized) if kind == "question" else parse_statement(normalized)
    formal, formula_type = _formula_shape(parsed.formula)
    warnings, blocking = _semantic_warnings(original, normalized, parsed.formula, kind)
    if parsed.formula is None:
        blocking.insert(0, f"parse_error:{parsed.error}")
    blocking = list(dict.fromkeys(blocking))
    return ScientificPreview(
        kind=kind,
        original_text=original,
        normalized_text=normalized,
        parse_ok=parsed.formula is not None and not blocking,
        formal=formal,
        formula_type=formula_type,
        warnings=warnings,
        blocking_warnings=blocking,
        needs_llm_fallback=bool(blocking),
        normalization_steps=steps,
    )



def derive_label_type_premises(texts: list[str]) -> list[dict[str, str]]:
    """Derive transparent lexical type facts from labels such as Treatment A.

    This is not external world knowledge: the type word is part of the entity
    label itself. The derived fact is surfaced in the preview and provenance.
    """
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for text in texts:
        value = str(text or "")
        matches = [*_LABELLED_ENTITY.finditer(value), *_UNDERSCORED_LABELLED_ENTITY.finditer(value)]
        for match in matches:
            entity = f"{match.group(1)}_{match.group(2)}"
            entity_type = _singularize(match.group(1))
            key = (entity.lower(), entity_type)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "entity": entity,
                "entity_type": entity_type,
                "statement": f"{entity} is a {entity_type}.",
                "source_text": match.group(0),
                "provenance": "lexical_entity_type",
            })
    return rows

def preview_record_items(context: list[str], question: str, *, mode: str = "general_science") -> dict[str, Any]:
    from .symbol_alignment import align_item_texts
    items: list[dict[str, Any]] = []
    for index, text in enumerate(context, 1):
        row = preview_scientific_text(text, kind="premise", mode=mode).to_dict()
        row["id"] = f"p{index}"
        row["derived"] = False
        items.append(row)
    derived = derive_label_type_premises([*context, question]) if mode == "general_science" else []
    for offset, info in enumerate(derived, len(items) + 1):
        row = preview_scientific_text(info["statement"], kind="premise", mode="controlled").to_dict()
        row.update({
            "id": f"p{offset}",
            "derived": True,
            "derivation_type": "lexical_entity_type",
            "derived_from": info["source_text"],
        })
        items.append(row)
    q = preview_scientific_text(question, kind="query_statement", mode=mode).to_dict()
    q["id"] = "question"
    q["derived"] = False
    items.append(q)
    blockers = [x["id"] for x in items if x["blocking_warnings"]]
    align_input = [
        {"id": x["id"], "kind": "query_statement" if x["id"] == "question" else "premise", "text": x["normalized_text"]}
        for x in items if not x["blocking_warnings"]
    ]
    alignment = align_item_texts(align_input)
    aligned_by_id = {x["id"]: x for x in alignment["items"]}
    for row in items:
        aligned = aligned_by_id.get(row["id"])
        if aligned and aligned.get("text") != row["normalized_text"]:
            row["globally_aligned_text"] = aligned["text"]
            row["global_alignment_applied"] = True
        else:
            row["globally_aligned_text"] = row["normalized_text"]
            row["global_alignment_applied"] = False
    drift_blocked = bool(alignment["diagnostics"].get("blocking_symbol_drift"))
    return {
        "mode": mode,
        "items": items,
        "global_symbol_table": alignment["symbol_table"],
        "alignment_decisions": alignment["alignment_decisions"],
        "connectivity": alignment["diagnostics"],
        "summary": {
            "item_count": len(items),
            "clean_count": sum(not x["blocking_warnings"] for x in items),
            "normalized_count": sum(x["original_text"] != x["normalized_text"] for x in items),
            "globally_aligned_count": sum(bool(x.get("global_alignment_applied")) for x in items),
            "derived_premise_count": len(derived),
            "fallback_needed_count": len(blockers),
            "fallback_needed_ids": blockers,
            "symbol_drift_blocked": drift_blocked,
            "safe_for_deterministic_verification": not blockers and not drift_blocked,
            "query_predicate_connected": alignment["diagnostics"].get("query_predicate_connected"),
            "new_api_calls": 0,
        },
    }

