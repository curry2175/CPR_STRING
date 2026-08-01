from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .logic import Atom, Formula, Rule


ALLOWED_RELATION_TYPES = {"same_as", "implies", "related_to"}


@dataclass(frozen=True)
class SemanticRelation:
    id: str
    relation_type: str
    arity: int
    approved: bool
    source: str | None = None
    target: str | None = None
    predicates: tuple[str, ...] = ()
    canonical: str | None = None
    description: str = ""

    @property
    def proof_usable(self) -> bool:
        return self.approved and self.relation_type in {"same_as", "implies"}

    def display_text(self) -> str:
        if self.relation_type == "same_as":
            members = " ≡ ".join(self.predicates)
            return f"{members}  → canonical: {self.canonical}"
        arrow = "→" if self.relation_type == "implies" else "~"
        return f"{self.source} {arrow} {self.target}  (arity {self.arity})"

    def bridge_rule(self) -> Rule | None:
        if self.relation_type != "implies" or not self.approved:
            return None
        assert self.source and self.target
        variables = tuple(f"?x{i + 1}" for i in range(self.arity))
        return Rule((Atom(self.source, variables),), Atom(self.target, variables))


def _predicate_name(value: Any, field: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not text or not text.replace("_", "").isalnum():
        raise ValueError(f"Invalid semantic predicate in {field}: {value!r}")
    return text


def parse_semantic_relations(raw: Any) -> list[SemanticRelation]:
    if raw in (None, ""):
        return []
    if not isinstance(raw, list):
        raise ValueError("semantic_relations must be a list")

    relations: list[SemanticRelation] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"semantic_relations[{index}] must be an object")
        relation_id = str(item.get("id") or f"m{index}")
        if relation_id in seen_ids:
            raise ValueError(f"Duplicate semantic relation id: {relation_id}")
        seen_ids.add(relation_id)

        relation_type = str(item.get("type") or "").strip().lower()
        if relation_type not in ALLOWED_RELATION_TYPES:
            raise ValueError(
                f"Semantic relation {relation_id} type must be one of "
                f"{sorted(ALLOWED_RELATION_TYPES)}"
            )
        arity = int(item.get("arity", 1))
        if arity < 1 or arity > 4:
            raise ValueError(f"Semantic relation {relation_id} arity must be 1-4")
        approved = bool(item.get("approved", relation_type != "related_to"))
        description = str(item.get("description") or "")

        if relation_type == "same_as":
            predicates_raw = item.get("predicates")
            if not isinstance(predicates_raw, list) or len(predicates_raw) < 2:
                raise ValueError(f"same_as relation {relation_id} needs at least two predicates")
            predicates = tuple(dict.fromkeys(_predicate_name(value, "predicates") for value in predicates_raw))
            canonical = _predicate_name(item.get("canonical") or predicates[0], "canonical")
            if canonical not in predicates:
                raise ValueError(f"same_as relation {relation_id} canonical must be in predicates")
            relations.append(
                SemanticRelation(
                    id=relation_id,
                    relation_type=relation_type,
                    arity=arity,
                    approved=approved,
                    predicates=predicates,
                    canonical=canonical,
                    description=description,
                )
            )
        else:
            source = _predicate_name(item.get("source"), "source")
            target = _predicate_name(item.get("target"), "target")
            if source == target:
                raise ValueError(f"Semantic relation {relation_id} source and target must differ")
            relations.append(
                SemanticRelation(
                    id=relation_id,
                    relation_type=relation_type,
                    arity=arity,
                    approved=approved,
                    source=source,
                    target=target,
                    description=description,
                )
            )
    return relations


class SemanticLayer:
    """Safe predicate-semantic preprocessing for MVP v005.

    - approved ``same_as`` relations canonicalize predicates before SMT/Horn use;
    - approved ``implies`` relations become explicit Horn/Z3 bridge rules;
    - ``related_to`` relations are advisory only and never enter the proof theory.
    """

    def __init__(self, relations: Iterable[SemanticRelation], *, disabled_ids: set[str] | None = None) -> None:
        self.relations = list(relations)
        self.disabled_ids = disabled_ids or set()
        self.alias_map: dict[tuple[str, int], tuple[str, str]] = {}
        for relation in self.relations:
            if relation.id in self.disabled_ids or not relation.proof_usable or relation.relation_type != "same_as":
                continue
            assert relation.canonical
            for predicate in relation.predicates:
                self.alias_map[(predicate, relation.arity)] = (relation.canonical, relation.id)

    def normalize_atom(self, atom: Atom) -> tuple[Atom, list[str]]:
        key = (atom.predicate, len(atom.args))
        replacement = self.alias_map.get(key)
        if replacement is None:
            return atom, []
        canonical, relation_id = replacement
        if canonical == atom.predicate:
            return atom, []
        return Atom(canonical, atom.args, atom.negated), [relation_id]

    def normalize_formula(self, formula: Formula) -> tuple[Formula, list[str]]:
        if isinstance(formula, Atom):
            return self.normalize_atom(formula)
        relation_ids: set[str] = set()
        antecedents: list[Atom] = []
        for atom in formula.antecedents:
            normalized, used = self.normalize_atom(atom)
            antecedents.append(normalized)
            relation_ids.update(used)
        consequent, used = self.normalize_atom(formula.consequent)
        relation_ids.update(used)
        return Rule(tuple(antecedents), consequent), sorted(relation_ids)

    def bridge_rules(self) -> list[tuple[SemanticRelation, Rule]]:
        result: list[tuple[SemanticRelation, Rule]] = []
        for relation in self.relations:
            if relation.id in self.disabled_ids:
                continue
            bridge = relation.bridge_rule()
            if bridge is not None:
                normalized, _ = self.normalize_formula(bridge)
                assert isinstance(normalized, Rule)
                result.append((relation, normalized))
        return result

    def relation(self, relation_id: str) -> SemanticRelation | None:
        return next((relation for relation in self.relations if relation.id == relation_id), None)

    def related_hints(
        self,
        target: Formula,
        prior_predicates: dict[str, set[tuple[str, int]]],
    ) -> list[dict[str, Any]]:
        if not isinstance(target, Atom):
            return []
        target_key = (target.predicate, len(target.args))
        hints: list[dict[str, Any]] = []
        for relation in self.relations:
            if relation.id in self.disabled_ids or relation.relation_type != "related_to":
                continue
            assert relation.source and relation.target
            left = (relation.source, relation.arity)
            right = (relation.target, relation.arity)
            if target_key == left:
                other = right
            elif target_key == right:
                other = left
            else:
                continue
            from_nodes = sorted(node_id for node_id, predicates in prior_predicates.items() if other in predicates)
            if not from_nodes:
                continue
            hints.append(
                {
                    "relation_id": relation.id,
                    "type": relation.relation_type,
                    "target_predicate": target.predicate,
                    "related_predicate": other[0],
                    "from_nodes": from_nodes,
                    "proof_usable": False,
                    "message": (
                        f"{other[0]} and {target.predicate} are marked related, "
                        "but related_to is advisory and cannot prove the target."
                    ),
                }
            )
        return hints
