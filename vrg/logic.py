from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


VARIABLE_PREFIX = "?"


def is_variable(term: str) -> bool:
    return term.startswith(VARIABLE_PREFIX)


@dataclass(frozen=True)
class Atom:
    predicate: str
    args: tuple[str, ...]
    negated: bool = False

    def complement(self) -> "Atom":
        return Atom(self.predicate, self.args, not self.negated)

    def ground(self, substitution: Mapping[str, str]) -> "Atom":
        return Atom(
            self.predicate,
            tuple(substitution.get(arg, arg) for arg in self.args),
            self.negated,
        )

    @property
    def variables(self) -> set[str]:
        return {arg for arg in self.args if is_variable(arg)}

    def signature(self) -> tuple[str, int]:
        return self.predicate, len(self.args)

    def to_text(self) -> str:
        prefix = "not " if self.negated else ""
        return f"{prefix}{self.predicate}({', '.join(self.args)})"


@dataclass(frozen=True)
class Rule:
    antecedents: tuple[Atom, ...]
    consequent: Atom

    @property
    def variables(self) -> set[str]:
        result: set[str] = set()
        for atom in self.antecedents:
            result.update(atom.variables)
        result.update(self.consequent.variables)
        return result

    def to_text(self) -> str:
        left = " and ".join(atom.to_text() for atom in self.antecedents)
        return f"({left}) -> {self.consequent.to_text()}"


Formula = Atom | Rule


def formula_to_text(formula: Formula) -> str:
    return formula.to_text()


def constants_in_formula(formula: Formula) -> set[str]:
    atoms: Iterable[Atom]
    if isinstance(formula, Atom):
        atoms = [formula]
    else:
        atoms = [*formula.antecedents, formula.consequent]
    return {
        arg
        for atom in atoms
        for arg in atom.args
        if not is_variable(arg)
    }


def predicates_in_formula(formula: Formula) -> set[tuple[str, int]]:
    atoms: Iterable[Atom]
    if isinstance(formula, Atom):
        atoms = [formula]
    else:
        atoms = [*formula.antecedents, formula.consequent]
    return {atom.signature() for atom in atoms}
