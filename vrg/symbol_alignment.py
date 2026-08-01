from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .logic import Atom, Formula, Rule
from .parser import parse_question, parse_statement


KNOWN_HEAD_TYPES = {
    "study", "patient", "treatment", "drug", "intervention", "therapy",
    "regimen", "group", "arm", "cohort", "model", "agent", "compound",
    "protocol", "strategy", "person", "animal", "cell", "trial",
}


@dataclass
class SymbolOccurrence:
    item_id: str
    role: str
    predicate: str
    arity: int
    negated: bool


@dataclass
class AlignmentDecision:
    item_id: str
    original_predicate: str
    canonical_predicate: str
    added_type_predicate: str | None
    reason: str


def _atoms(formula: Formula) -> list[Atom]:
    if isinstance(formula, Atom):
        return [formula]
    return [*formula.antecedents, formula.consequent]


def _predicate_base_and_type(predicate: str) -> tuple[str, str] | None:
    for head in sorted(KNOWN_HEAD_TYPES, key=len, reverse=True):
        suffix = f"_{head}"
        if predicate.endswith(suffix) and len(predicate) > len(suffix):
            return predicate[: -len(suffix)], head
    return None


def _collect_occurrences(items: Iterable[dict[str, Any]]) -> tuple[list[SymbolOccurrence], dict[str, Formula]]:
    rows: list[SymbolOccurrence] = []
    formulas: dict[str, Formula] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or "")
        kind = str(item.get("kind") or "premise")
        parsed = parse_question(text) if kind == "question" else parse_statement(text)
        if parsed.formula is None:
            continue
        formulas[item_id] = parsed.formula
        if isinstance(parsed.formula, Atom):
            rows.append(SymbolOccurrence(item_id, "query" if kind in {"question", "query_statement"} else "fact", parsed.formula.predicate, len(parsed.formula.args), parsed.formula.negated))
        else:
            for atom in parsed.formula.antecedents:
                rows.append(SymbolOccurrence(item_id, "antecedent", atom.predicate, len(atom.args), atom.negated))
            atom = parsed.formula.consequent
            rows.append(SymbolOccurrence(item_id, "consequent", atom.predicate, len(atom.args), atom.negated))
    return rows, formulas


def build_global_symbol_table(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    occurrences, formulas = _collect_occurrences(items)
    by_symbol: dict[tuple[str, int], list[SymbolOccurrence]] = defaultdict(list)
    for row in occurrences:
        by_symbol[(row.predicate, row.arity)].append(row)
    symbols = []
    for (predicate, arity), rows in sorted(by_symbol.items()):
        symbols.append({
            "predicate": predicate,
            "arity": arity,
            "roles": sorted({x.role for x in rows}),
            "item_ids": sorted({x.item_id for x in rows}),
            "negative_occurrences": sum(x.negated for x in rows),
            "compound_head": (_predicate_base_and_type(predicate) or (None, None))[1],
        })
    return {
        "symbols": symbols,
        "predicate_names": sorted({x.predicate for x in occurrences}),
        "formula_count": len(formulas),
    }


def _align_atom(atom: Atom, *, known_predicates: set[str], role: str, item_id: str) -> tuple[list[Atom], list[AlignmentDecision]]:
    split = _predicate_base_and_type(atom.predicate)
    if split is None or len(atom.args) != 1:
        return [atom], []
    base, head = split
    # A compound modifier+head predicate is decomposed only when there is
    # evidence that the modifier is used elsewhere, or when it occurs in a
    # rule antecedent where head-noun decomposition is semantically explicit.
    if base not in known_predicates and role != "antecedent":
        return [atom], []
    canonical = Atom(base, atom.args, atom.negated)
    decision = AlignmentDecision(
        item_id=item_id,
        original_predicate=atom.predicate,
        canonical_predicate=base,
        added_type_predicate=head if role == "antecedent" else None,
        reason="modifier_head_decomposition",
    )
    if role == "antecedent":
        type_atom = Atom(head, atom.args, False)
        return [type_atom, canonical], [decision]
    return [canonical], [decision]


def align_formula(formula: Formula, *, known_predicates: set[str], item_id: str) -> tuple[Formula, list[AlignmentDecision]]:
    if isinstance(formula, Atom):
        atoms, decisions = _align_atom(formula, known_predicates=known_predicates, role="fact", item_id=item_id)
        # A standalone fact cannot represent a conjunction. Retain the semantic
        # modifier and rely on transparent lexical type premises for the head.
        return atoms[-1], decisions
    antecedents: list[Atom] = []
    decisions: list[AlignmentDecision] = []
    for atom in formula.antecedents:
        aligned, rows = _align_atom(atom, known_predicates=known_predicates, role="antecedent", item_id=item_id)
        antecedents.extend(aligned)
        decisions.extend(rows)
    consequent_rows, rows = _align_atom(formula.consequent, known_predicates=known_predicates, role="consequent", item_id=item_id)
    decisions.extend(rows)
    consequent = consequent_rows[-1]
    # Stable de-duplication preserves semantic order.
    seen: set[tuple[str, tuple[str, ...], bool]] = set()
    unique: list[Atom] = []
    for atom in antecedents:
        key = (atom.predicate, atom.args, atom.negated)
        if key not in seen:
            seen.add(key)
            unique.append(atom)
    return Rule(tuple(unique), consequent), decisions


def _human_predicate(predicate: str) -> str:
    return predicate.replace("_", " ")


def _verb_form(predicate: str) -> str:
    if predicate.endswith("y") and len(predicate) > 2 and predicate[-2] not in "aeiou":
        return predicate[:-1] + "ies"
    if predicate.endswith(("s", "x", "z", "ch", "sh")):
        return predicate + "es"
    return predicate + "s"


def _term(term: str, *, first_variable: bool = False) -> str:
    if term.startswith("?"):
        return "something" if first_variable else "it"
    return term


def atom_to_controlled_english(atom: Atom, *, first_variable: bool = False) -> str:
    subject = _term(atom.args[0], first_variable=first_variable)
    predicate = _human_predicate(atom.predicate)
    if len(atom.args) == 1:
        return f"{subject} is {'not ' if atom.negated else ''}{predicate}"
    obj = _term(atom.args[1], first_variable=False)
    if atom.negated:
        return f"{subject} does not {predicate} {obj}"
    verb = _verb_form(predicate) if subject not in {"something", "it"} else predicate
    return f"{subject} {verb} {obj}"


def formula_to_controlled_english(formula: Formula) -> str:
    if isinstance(formula, Atom):
        return atom_to_controlled_english(formula, first_variable=True) + "."
    parts = [atom_to_controlled_english(atom, first_variable=(index == 0)) for index, atom in enumerate(formula.antecedents)]
    consequent = atom_to_controlled_english(formula.consequent, first_variable=False)
    return f"If {' and '.join(parts)}, then {consequent}."


def align_item_texts(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Align near-duplicate predicates across an entire context/query batch.

    The input texts must already be parseable controlled English. The function
    returns rewritten texts plus an auditable global symbol table and alignment
    decisions. No external knowledge is added.
    """
    cloned = deepcopy(items)
    occurrences, formulas = _collect_occurrences(cloned)
    known_predicates = {x.predicate for x in occurrences}
    decisions: list[AlignmentDecision] = []
    for item in cloned:
        item_id = str(item.get("id") or "")
        formula = formulas.get(item_id)
        if formula is None:
            continue
        aligned, rows = align_formula(formula, known_predicates=known_predicates, item_id=item_id)
        if rows:
            item["text"] = formula_to_controlled_english(aligned)
            item["aligned_formula"] = aligned.to_text()
            decisions.extend(rows)
    table = build_global_symbol_table(cloned)
    diagnostics = connectivity_diagnostics(cloned)
    return {
        "items": cloned,
        "symbol_table": table,
        "alignment_decisions": [asdict(x) for x in decisions],
        "diagnostics": diagnostics,
    }


def connectivity_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    occurrences, formulas = _collect_occurrences(items)
    fact_signatures: set[tuple[str, int]] = set()
    consequent_signatures: set[tuple[str, int]] = set()
    antecedent_signatures: set[tuple[str, int]] = set()
    rules: list[Rule] = []
    query_signature: tuple[str, int] | None = None
    for item in items:
        item_id = str(item.get("id") or "")
        formula = formulas.get(item_id)
        if formula is None:
            continue
        kind = str(item.get("kind") or "premise")
        if kind in {"question", "query_statement"} and isinstance(formula, Atom):
            query_signature = formula.signature()
        elif isinstance(formula, Atom):
            fact_signatures.add(formula.signature())
        else:
            rules.append(formula)
            consequent_signatures.add(formula.consequent.signature())
            antecedent_signatures.update(x.signature() for x in formula.antecedents)
    producible = set(fact_signatures)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if all(atom.signature() in producible for atom in rule.antecedents):
                sig = rule.consequent.signature()
                if sig not in producible:
                    producible.add(sig)
                    changed = True
    orphans = sorted(antecedent_signatures - fact_signatures - consequent_signatures)
    aliasable_orphans = []
    predicate_names = {x.predicate for x in occurrences}
    for predicate, arity in orphans:
        split = _predicate_base_and_type(predicate)
        if split and split[0] in predicate_names:
            aliasable_orphans.append({"predicate": predicate, "arity": arity, "candidate_base": split[0], "head_type": split[1]})
    query_connected = query_signature in producible if query_signature is not None else None
    return {
        "fact_signatures": sorted(f"{p}/{a}" for p, a in fact_signatures),
        "rule_consequent_signatures": sorted(f"{p}/{a}" for p, a in consequent_signatures),
        "orphan_antecedents": [{"predicate": p, "arity": a} for p, a in orphans],
        "aliasable_orphan_antecedents": aliasable_orphans,
        "query_signature": f"{query_signature[0]}/{query_signature[1]}" if query_signature else None,
        "query_predicate_connected": query_connected,
        "connectivity_interpretation": (
            "connected" if query_connected else
            "disconnected_but_may_correctly_imply_unknown" if query_signature is not None else
            "query_unavailable"
        ),
        "blocking_symbol_drift": bool(aliasable_orphans),
    }
