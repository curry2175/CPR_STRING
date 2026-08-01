from __future__ import annotations

import re
from dataclasses import dataclass

from .logic import Atom, Formula, Rule


class ParseError(ValueError):
    pass


@dataclass
class ParseResult:
    formula: Formula | None
    error: str | None = None


VARIABLE_WORDS = {
    "something",
    "someone",
    "somebody",
    "anything",
    "anyone",
    "person",
    "thing",
}
PRONOUN_WORDS = {"it", "they", "them", "he", "she"}


def _clean(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(?:step\s*)?\d+\s*[.)\-:]\s*", "", text, flags=re.I)
    text = re.sub(r"^(therefore|thus|hence|so|consequently)\s*,?\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip().rstrip(".?!").strip()


def _entity(text: str, variables: dict[str, str], last_variable: str | None) -> tuple[str, str | None]:
    value = text.strip().lower()
    value = re.sub(r"^(the|a|an)\s+", "", value)
    if value in PRONOUN_WORDS and last_variable:
        return last_variable, last_variable
    if value in VARIABLE_WORDS:
        if value not in variables:
            variables[value] = f"?x{len(variables) + 1}"
        return variables[value], variables[value]
    value = re.sub(r"[^a-z0-9_\- ]", "", value)
    value = value.replace("-", "_").replace(" ", "_")
    if not value:
        raise ParseError("Empty entity")
    return value, last_variable


def _lemma(verb: str) -> str:
    verb = verb.lower().strip()
    irregular = {
        "has": "have",
        "does": "do",
        "goes": "go",
        "flies": "fly",
        "tries": "try",
        "carries": "carry",
        "is": "be",
        "are": "be",
    }
    if verb in irregular:
        return irregular[verb]
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb in {"chases": "chase", "sees": "see", "uses": "use", "loses": "lose", "needs": "need", "likes": "like", "eats": "eat", "visits": "visit"}:
        return {"chases": "chase", "sees": "see", "uses": "use", "loses": "lose", "needs": "need", "likes": "like", "eats": "eat", "visits": "visit"}[verb]
    if verb.endswith(("ches", "shes", "xes", "zes", "oes", "sses")) and len(verb) > 4:
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss") and len(verb) > 3:
        return verb[:-1]
    return verb


def _property(text: str) -> str:
    value = text.strip().lower()
    value = re.sub(r"^(a|an|the)\s+", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    if not value:
        raise ParseError("Empty property")
    return value


def _parse_clause(
    text: str,
    variables: dict[str, str] | None = None,
    last_variable: str | None = None,
) -> tuple[Atom, str | None]:
    variables = variables if variables is not None else {}
    clause = _clean(text).lower()

    # Copular negative: Bob is not quiet / Bob is not a doctor
    match = re.match(r"^(.+?)\s+(?:is|are)\s+not\s+(.+)$", clause)
    if match:
        subject, last_variable = _entity(match.group(1), variables, last_variable)
        return Atom(_property(match.group(2)), (subject,), True), last_variable

    # Copular positive: Bob is quiet / the mouse is big
    match = re.match(r"^(.+?)\s+(?:is|are)\s+(.+)$", clause)
    if match:
        subject, last_variable = _entity(match.group(1), variables, last_variable)
        prop = match.group(2)
        # Reject obviously modal/vague language in the deterministic parser.
        if re.search(r"\b(maybe|possibly|probably|seems|appears|might|may|aura|vibe)\b", prop):
            raise ParseError("Modal or vague expression is outside the controlled logic subset")
        return Atom(_property(prop), (subject,), False), last_variable

    # Relational negative: Bob does not visit Alice
    match = re.match(r"^((?:(?:the|a|an)\s+)?[a-z0-9_-]+)\s+(?:does|do)\s+not\s+([a-z][a-z0-9_-]*)\s+(.+)$", clause)
    if match:
        subject, last_variable = _entity(match.group(1), variables, last_variable)
        obj, last_variable = _entity(match.group(3), variables, last_variable)
        return Atom(_lemma(match.group(2)), (subject, obj), True), last_variable

    # Relational positive: The mouse eats the tiger
    match = re.match(r"^((?:(?:the|a|an)\s+)?[a-z0-9_-]+)\s+([a-z][a-z0-9_-]*)\s+(.+)$", clause)
    if match:
        subject_text, verb, object_text = match.groups()
        if verb in {"and", "or", "than"}:
            raise ParseError("Unsupported relational clause")
        subject, last_variable = _entity(subject_text, variables, last_variable)
        obj, last_variable = _entity(object_text, variables, last_variable)
        return Atom(_lemma(verb), (subject, obj), False), last_variable

    raise ParseError(f"Unsupported controlled-English clause: {text}")


def _descriptor_atom(descriptor: str, variable: str) -> Atom:
    value = descriptor.strip()
    negated = False
    if re.match(r"^not\s+", value, flags=re.I):
        negated = True
        value = re.sub(r"^not\s+", "", value, flags=re.I)
    return Atom(_property(value), (variable,), negated)


def _split_descriptor_list(text: str) -> list[str]:
    """Split ProofWriter-style unary adjective conjunctions."""
    return [
        piece.strip()
        for piece in re.split(r"\s*,\s*|\s+and\s+", text, flags=re.I)
        if piece.strip()
    ]


def _expand_antecedent_shorthand(text: str) -> list[str]:
    """Expand ``someone is quiet and red`` into complete clauses."""
    raw_parts = [piece.strip() for piece in re.split(r"\s+and\s+", text, flags=re.I) if piece.strip()]
    if len(raw_parts) <= 1:
        return raw_parts
    first = raw_parts[0]
    match = re.match(r"^(.+?)\s+(is|are)\s+(.+)$", first, flags=re.I)
    if not match:
        return raw_parts
    subject, copula = match.group(1).strip(), match.group(2).strip()
    expanded = [first]
    for part in raw_parts[1:]:
        complete_relation = re.match(r"^((?:the|a|an)\s+)?[a-z0-9_-]+\s+(?:does\s+not\s+|do\s+not\s+)?[a-z][a-z0-9_-]*\s+.+$", part, flags=re.I)
        if re.search(r"\b(is|are|does|do)\b", part, flags=re.I) or complete_relation:
            expanded.append(part)
        else:
            expanded.append(f"{subject} {copula} {part}")
    return expanded


def _split_antecedents(text: str) -> list[str]:
    if re.search(r"\bor\b", text, flags=re.I):
        raise ParseError("OR rules are not supported in the controlled logic subset")
    return _expand_antecedent_shorthand(text)


def parse_statement(text: str) -> ParseResult:
    cleaned = _clean(text)
    if not cleaned:
        return ParseResult(None, "Empty statement")
    if re.search(r"[.!?]\s+\S|;", cleaned):
        return ParseResult(None, "Compound reasoning step contains multiple sentence-level claims; split it into atomic steps")

    try:
        # Explicit conditional.
        conditional = re.match(r"^if\s+(.+?)\s*,?\s+then\s+(.+)$", cleaned, flags=re.I)
        if conditional:
            variables: dict[str, str] = {}
            last_variable: str | None = None
            antecedents: list[Atom] = []
            for part in _split_antecedents(conditional.group(1)):
                atom, last_variable = _parse_clause(part, variables, last_variable)
                antecedents.append(atom)
            consequent, _ = _parse_clause(conditional.group(2), variables, last_variable)
            return ParseResult(Rule(tuple(antecedents), consequent))

        # ProofWriter universals, including conjunctive descriptors.
        universal = re.match(r"^all\s+(.+?)\s+(?:people|things|persons|objects)\s+are\s+(.+)$", cleaned, flags=re.I)
        if universal:
            variable = "?x1"
            antecedents = tuple(_descriptor_atom(item, variable) for item in _split_descriptor_list(universal.group(1)))
            return ParseResult(Rule(antecedents, _descriptor_atom(universal.group(2), variable)))

        universal = re.match(r"^(.+?)\s+(people|things|persons|objects)\s+are\s+(.+)$", cleaned, flags=re.I)
        if universal:
            variable = "?x1"
            antecedents = tuple(_descriptor_atom(item, variable) for item in _split_descriptor_list(universal.group(1)))
            return ParseResult(Rule(antecedents, _descriptor_atom(universal.group(3), variable)))


        atom, _ = _parse_clause(cleaned)
        return ParseResult(atom)
    except ParseError as exc:
        return ParseResult(None, str(exc))


def parse_question(text: str) -> ParseResult:
    cleaned = _clean(text)
    try:
        # Is Bob kind? / Is the mouse not big?
        match = re.match(r"^(?:is|are)\s+(.+?)\s+not\s+(.+)$", cleaned, flags=re.I)
        if match:
            variables: dict[str, str] = {}
            subject, _ = _entity(match.group(1), variables, None)
            return ParseResult(Atom(_property(match.group(2)), (subject,), True))

        match = re.match(r"^(?:is|are)\s+(.+?)\s+(.+)$", cleaned, flags=re.I)
        if match:
            variables = {}
            subject, _ = _entity(match.group(1), variables, None)
            return ParseResult(Atom(_property(match.group(2)), (subject,), False))

        # Does the mouse not visit the rabbit?
        match = re.match(r"^(?:does|do)\s+((?:(?:the|a|an)\s+)?[a-z0-9_-]+)\s+not\s+([a-z][a-z0-9_-]*)\s+(.+)$", cleaned, flags=re.I)
        if match:
            variables = {}
            subject, _ = _entity(match.group(1), variables, None)
            obj, _ = _entity(match.group(3), variables, None)
            return ParseResult(Atom(_lemma(match.group(2)), (subject, obj), True))

        # Does the mouse visit the rabbit?
        match = re.match(r"^(?:does|do)\s+((?:(?:the|a|an)\s+)?[a-z0-9_-]+)\s+([a-z][a-z0-9_-]*)\s+(.+)$", cleaned, flags=re.I)
        if match:
            variables = {}
            subject, _ = _entity(match.group(1), variables, None)
            obj, _ = _entity(match.group(3), variables, None)
            return ParseResult(Atom(_lemma(match.group(2)), (subject, obj), False))
    except ParseError as exc:
        return ParseResult(None, str(exc))
    return ParseResult(None, f"Unsupported Yes/No question: {text}")
