from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable

from .logic import Atom, Formula, Rule, constants_in_formula, is_variable

try:
    import z3  # type: ignore
except ImportError:  # pragma: no cover - fallback is tested in this environment
    z3 = None


@dataclass
class KnowledgeItem:
    node_id: str
    formula: Formula


@dataclass
class CheckResult:
    status: str
    dependencies: list[str]
    engine: str
    detail: str
    consistency_result: str = "not_run"
    entailment_result: str = "not_run"
    target_smtlib: str | None = None
    consistency_query_smtlib: str | None = None
    entailment_query_smtlib: str | None = None
    raw_dependencies: list[str] = field(default_factory=list)
    minimized_dependencies: list[str] = field(default_factory=list)
    core_minimized: bool = False
    solver_instances: int = 0
    solver_checks: int = 0
    knowledge_assertions: int = 0
    core_minimization_checks: int = 0


@dataclass
class SupportResult:
    found: bool
    dependencies: list[str]
    detail: str


class ChainSupportEngine:
    """Finite Horn path finder used for transparent derivation tracing.

    The engine has two related modes:
    1. ``support`` prefers prior LLM reasoning nodes, approximating the path the
       model actually wrote.
    2. ``support_paths`` enumerates a small number of distinct Horn support sets,
       allowing the UI to show concrete alternative derivations.

    It deliberately does not use logical explosion: parseable invalid steps can
    remain in the shadow chain so downstream contamination can be displayed.
    """

    def __init__(self, node_kinds: dict[str, str], node_orders: dict[str, int]) -> None:
        self.node_kinds = node_kinds
        self.node_orders = node_orders

    def _score(self, dependencies: set[str]) -> tuple[int, int, int]:
        reasoning_count = sum(self.node_kinds.get(node_id) == "reasoning" for node_id in dependencies)
        newest_reasoning = max(
            (self.node_orders.get(node_id, -1) for node_id in dependencies if self.node_kinds.get(node_id) == "reasoning"),
            default=-1,
        )
        # Prefer more CoT nodes, then fewer total nodes, then the latest CoT node.
        return reasoning_count, -len(dependencies), newest_reasoning

    def _path_score(self, dependencies: frozenset[str]) -> tuple[int, int, int]:
        # For an alternatives list, prefer compact proofs; use later CoT nodes as
        # a stable tie-breaker so the display remains deterministic.
        reasoning_count = sum(self.node_kinds.get(node_id) == "reasoning" for node_id in dependencies)
        newest = max((self.node_orders.get(node_id, -1) for node_id in dependencies), default=-1)
        return -len(dependencies), reasoning_count, newest

    def _better(self, candidate: set[str], current: set[str] | None) -> bool:
        return current is None or self._score(candidate) > self._score(current)

    def _closure(self, items: list[KnowledgeItem]) -> dict[Atom, set[str]]:
        proof: dict[Atom, set[str]] = {}
        rules: list[tuple[str, Rule]] = []
        constants: set[str] = set()

        for item in items:
            constants.update(constants_in_formula(item.formula))
            if isinstance(item.formula, Atom):
                candidate = {item.node_id}
                if self._better(candidate, proof.get(item.formula)):
                    proof[item.formula] = candidate
            else:
                rules.append((item.node_id, item.formula))

        if not constants:
            constants.add("default_entity")

        changed = True
        while changed:
            changed = False
            for rule_id, rule in rules:
                variables = sorted(rule.variables)
                assignments = [dict(zip(variables, values)) for values in product(sorted(constants), repeat=len(variables))]
                if not assignments:
                    assignments = [{}]
                for substitution in assignments:
                    antecedents = [atom.ground(substitution) for atom in rule.antecedents]
                    if not all(atom in proof for atom in antecedents):
                        continue
                    conclusion = rule.consequent.ground(substitution)
                    dependencies = {rule_id}
                    for antecedent in antecedents:
                        dependencies.update(proof[antecedent])
                    if self._better(dependencies, proof.get(conclusion)):
                        proof[conclusion] = dependencies
                        changed = True
        return proof

    def _all_closure(self, items: list[KnowledgeItem], *, cap_per_atom: int = 12) -> dict[Atom, list[frozenset[str]]]:
        paths: dict[Atom, list[frozenset[str]]] = {}
        rules: list[tuple[str, Rule]] = []
        constants: set[str] = set()

        def add_path(atom: Atom, candidate: frozenset[str]) -> bool:
            current = paths.setdefault(atom, [])
            if candidate in current:
                return False
            current.append(candidate)
            current.sort(key=self._path_score, reverse=True)
            if len(current) > cap_per_atom:
                del current[cap_per_atom:]
            return candidate in current

        for item in items:
            constants.update(constants_in_formula(item.formula))
            if isinstance(item.formula, Atom):
                add_path(item.formula, frozenset({item.node_id}))
            else:
                rules.append((item.node_id, item.formula))

        if not constants:
            constants.add("default_entity")

        # The cap keeps cyclic or redundant rule sets bounded for the MVP.
        changed = True
        rounds = 0
        while changed and rounds < 100:
            rounds += 1
            changed = False
            for rule_id, rule in rules:
                variables = sorted(rule.variables)
                assignments = [dict(zip(variables, values)) for values in product(sorted(constants), repeat=len(variables))]
                if not assignments:
                    assignments = [{}]
                for substitution in assignments:
                    antecedents = [atom.ground(substitution) for atom in rule.antecedents]
                    if not all(paths.get(atom) for atom in antecedents):
                        continue
                    antecedent_path_lists = [paths[atom] for atom in antecedents]
                    for selected_paths in product(*antecedent_path_lists):
                        combined: set[str] = {rule_id}
                        for selected in selected_paths:
                            combined.update(selected)
                        conclusion = rule.consequent.ground(substitution)
                        if add_path(conclusion, frozenset(combined)):
                            changed = True
        return paths

    def support(self, items: list[KnowledgeItem], target: Formula) -> SupportResult:
        if isinstance(target, Rule):
            candidates = [{item.node_id} for item in items if item.formula == target]
            if not candidates:
                return SupportResult(False, [], "No equivalent prior rule in chain view")
            best = max(candidates, key=self._score)
            return SupportResult(True, sorted(best), "Equivalent prior rule supports this step")

        proof = self._closure(items)
        dependencies = proof.get(target)
        if dependencies is None:
            return SupportResult(False, [], "No Horn support path found in prior premises/CoT steps")
        return SupportResult(True, sorted(dependencies), "Preferred CoT-aware Horn support path found")

    def support_paths(self, items: list[KnowledgeItem], target: Formula, *, max_paths: int = 3) -> list[list[str]]:
        """Return up to ``max_paths`` distinct finite Horn support sets."""
        if isinstance(target, Rule):
            candidates = [frozenset({item.node_id}) for item in items if item.formula == target]
        else:
            candidates = self._all_closure(items).get(target, [])
        unique = sorted(set(candidates), key=self._path_score, reverse=True)
        return [sorted(path) for path in unique[:max_paths]]


class HornEngine:
    """Deterministic fallback for finite Horn-rule examples."""

    name = "horn-fallback"

    @staticmethod
    def _pseudo_smtlib(formula: Formula) -> str:
        if isinstance(formula, Atom):
            expression = f"({formula.predicate} {' '.join(formula.args)})"
            return f"(not {expression})" if formula.negated else expression
        variables = " ".join(f"({name[1:]} Entity)" for name in sorted(formula.variables))
        left_parts = [HornEngine._pseudo_smtlib(atom) for atom in formula.antecedents]
        left = left_parts[0] if len(left_parts) == 1 else f"(and {' '.join(left_parts)})"
        body = f"(=> {left} {HornEngine._pseudo_smtlib(formula.consequent)})"
        return f"(forall ({variables}) {body})" if variables else body

    def formula_smtlib(self, formula: Formula) -> str:
        return self._pseudo_smtlib(formula)

    def _query_text(self, knowledge: list[KnowledgeItem], target: Formula) -> str:
        lines = ["; Horn fallback preview — this query was not executed by Z3", "(declare-sort Entity 0)"]
        for item in knowledge:
            lines.append(f"; {item.node_id}")
            lines.append(f"(assert {self._pseudo_smtlib(item.formula)})")
        lines.append("; target assumption")
        lines.append(f"(assert {self._pseudo_smtlib(target)})")
        lines.append("(check-sat)")
        return "\n".join(lines)

    def _closure(self, items: list[KnowledgeItem]) -> tuple[set[Atom], dict[Atom, set[str]]]:
        facts: set[Atom] = set()
        proof: dict[Atom, set[str]] = {}
        rules: list[tuple[str, Rule]] = []
        constants: set[str] = set()

        for item in items:
            constants.update(constants_in_formula(item.formula))
            if isinstance(item.formula, Atom):
                facts.add(item.formula)
                proof.setdefault(item.formula, {item.node_id})
            else:
                rules.append((item.node_id, item.formula))

        if not constants:
            constants.add("default_entity")

        changed = True
        while changed:
            changed = False
            for rule_id, rule in rules:
                variables = sorted(rule.variables)
                assignments = [dict(zip(variables, values)) for values in product(sorted(constants), repeat=len(variables))]
                if not assignments:
                    assignments = [{}]
                for substitution in assignments:
                    grounded_antecedents = [atom.ground(substitution) for atom in rule.antecedents]
                    if all(atom in facts for atom in grounded_antecedents):
                        conclusion = rule.consequent.ground(substitution)
                        dependencies = {rule_id}
                        for antecedent in grounded_antecedents:
                            dependencies.update(proof.get(antecedent, set()))
                        if conclusion not in facts:
                            facts.add(conclusion)
                            proof[conclusion] = dependencies
                            changed = True
                        else:
                            existing = proof.get(conclusion, set())
                            if not existing or len(dependencies) < len(existing):
                                proof[conclusion] = dependencies
                                changed = True
        return facts, proof

    def check(self, knowledge: list[KnowledgeItem], target: Formula) -> CheckResult:
        target_smtlib = self.formula_smtlib(target)
        positive_query = self._query_text(knowledge, target)
        negative_target = target.complement() if isinstance(target, Atom) else target
        negative_query = self._query_text(knowledge, negative_target)

        if isinstance(target, Rule):
            exact = [item.node_id for item in knowledge if item.formula == target]
            if exact:
                return CheckResult(
                    "valid", exact, self.name, "Equivalent rule already exists in knowledge",
                    consistency_result="sat", entailment_result="unsat",
                    target_smtlib=target_smtlib,
                    consistency_query_smtlib=positive_query,
                    entailment_query_smtlib=negative_query,
                    raw_dependencies=exact,
                    minimized_dependencies=exact,
                )
            return CheckResult(
                "ungrounded", [], self.name, "Rule is neither established nor contradicted in Horn fallback",
                consistency_result="sat", entailment_result="sat",
                target_smtlib=target_smtlib,
                consistency_query_smtlib=positive_query,
                entailment_query_smtlib=negative_query,
            )

        facts, proof = self._closure(knowledge)
        if target.complement() in facts:
            deps = sorted(proof.get(target.complement(), set()))
            return CheckResult(
                "contradiction", deps, self.name, "The complement of the target is derivable",
                consistency_result="unsat", entailment_result="not_run",
                target_smtlib=target_smtlib,
                consistency_query_smtlib=positive_query,
                entailment_query_smtlib=negative_query,
                raw_dependencies=deps,
                minimized_dependencies=deps,
            )
        if target in facts:
            deps = sorted(proof.get(target, set()))
            return CheckResult(
                "valid", deps, self.name, "The target is derivable by finite Horn forward chaining",
                consistency_result="sat", entailment_result="unsat",
                target_smtlib=target_smtlib,
                consistency_query_smtlib=positive_query,
                entailment_query_smtlib=negative_query,
                raw_dependencies=deps,
                minimized_dependencies=deps,
            )
        return CheckResult(
            "ungrounded", [], self.name, "Neither the target nor its complement is derivable",
            consistency_result="sat", entailment_result="sat",
            target_smtlib=target_smtlib,
            consistency_query_smtlib=positive_query,
            entailment_query_smtlib=negative_query,
        )


class Z3Engine:
    """Persistent per-case Z3 session.

    Knowledge is asserted once as assumption-guarded implications. Each claim is
    checked with ``check(active_knowledge_labels + target_label)``. This keeps a
    single tracked solver and a single clean SMT-LIB display solver alive for the
    whole case, while still yielding auditable unsat cores.
    """

    name = "z3"

    def __init__(self) -> None:
        if z3 is None:
            raise RuntimeError("z3-solver is not installed")
        self.entity_sort = z3.DeclareSort("Entity")
        self.constants: dict[str, object] = {}
        self.predicates: dict[tuple[str, int], object] = {}
        self.tracked_solver = z3.Solver()
        self.clean_solver = z3.Solver()
        self.active_labels: list[object] = []
        self.label_to_node: dict[str, str] = {}
        self.node_to_label: dict[str, object] = {}
        self._target_counter = 0
        self._reported_solver_instances = False
        self._pending_knowledge_assertions = 0

    def _constant(self, name: str):
        if name not in self.constants:
            self.constants[name] = z3.Const(name, self.entity_sort)
        return self.constants[name]

    def _predicate(self, name: str, arity: int):
        key = (name, arity)
        if key not in self.predicates:
            self.predicates[key] = z3.Function(name, *([self.entity_sort] * arity), z3.BoolSort())
        return self.predicates[key]

    def _atom_expr(self, atom: Atom, variables: dict[str, object] | None = None):
        variables = variables or {}
        fn = self._predicate(atom.predicate, len(atom.args))
        args = [variables[arg] if is_variable(arg) else self._constant(arg) for arg in atom.args]
        expr = fn(*args)
        return z3.Not(expr) if atom.negated else expr

    def _formula_expr(self, formula: Formula):
        if isinstance(formula, Atom):
            return self._atom_expr(formula)
        variables = {name: z3.Const(name[1:], self.entity_sort) for name in sorted(formula.variables)}
        antecedents = [self._atom_expr(atom, variables) for atom in formula.antecedents]
        left = z3.And(*antecedents) if len(antecedents) > 1 else antecedents[0]
        body = z3.Implies(left, self._atom_expr(formula.consequent, variables))
        return z3.ForAll(list(variables.values()), body) if variables else body

    def formula_smtlib(self, formula: Formula) -> str:
        return self._formula_expr(formula).sexpr()

    @staticmethod
    def _safe_label(node_id: str, index: int) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in node_id)
        return f"k__{index}__{cleaned}"

    def add_knowledge(self, item: KnowledgeItem) -> None:
        """Add one trusted item once to both persistent solvers."""
        if item.node_id in self.node_to_label:
            return
        expr = self._formula_expr(item.formula)
        label_name = self._safe_label(item.node_id, len(self.active_labels))
        label = z3.Bool(label_name)
        self.tracked_solver.add(z3.Implies(label, expr))
        self.clean_solver.add(expr)
        self.active_labels.append(label)
        self.label_to_node[label_name] = item.node_id
        self.node_to_label[item.node_id] = label
        # One assertion in each solver.
        self._pending_knowledge_assertions += 2

    def _next_target_label(self, suffix: str):
        self._target_counter += 1
        return z3.Bool(f"target__{self._target_counter}__{suffix}")

    def _query_smt2(self, target_expr) -> str:
        self.clean_solver.push()
        self.clean_solver.add(target_expr)
        query = self.clean_solver.to_smt2()
        self.clean_solver.pop()
        return query

    def _dependency_ids_from_core(self, core) -> list[str]:
        names = {str(label) for label in core}
        return sorted(self.label_to_node[name] for name in names if name in self.label_to_node)

    def _minimize_assumption_core(self, dependency_ids: list[str], target_label) -> tuple[list[str], int]:
        """Deletion-minimize a core using the same persistent tracked solver."""
        core = [node_id for node_id in dependency_ids if node_id in self.node_to_label]
        checks = 0
        index = 0
        while index < len(core):
            candidate = core[:index] + core[index + 1 :]
            assumptions = [self.node_to_label[node_id] for node_id in candidate] + [target_label]
            checks += 1
            if self.tracked_solver.check(*assumptions) == z3.unsat:
                core = candidate
            else:
                index += 1
        return sorted(core), checks

    def _assumption_check(self, target_expr, suffix: str) -> tuple[str, list[str], list[str], str, bool, int, int]:
        target_label = self._next_target_label(suffix)
        # Target guards are temporary. Keeping only trusted knowledge in the
        # persistent base makes repeated full/incremental runs more stable and
        # prevents the solver from accumulating inactive target clauses.
        self.tracked_solver.push()
        self.tracked_solver.add(z3.Implies(target_label, target_expr))
        result = self.tracked_solver.check(*(self.active_labels + [target_label]))
        solver_checks = 1
        if result == z3.unsat:
            raw = self._dependency_ids_from_core(self.tracked_solver.unsat_core())
            minimized, extra = self._minimize_assumption_core(raw, target_label)
            solver_checks += extra
            status = "unsat"
            changed = minimized != raw
        elif result == z3.sat:
            raw, minimized, extra, status, changed = [], [], 0, "sat", False
        else:
            raw, minimized, extra, status, changed = [], [], 0, "unknown", False
        self.tracked_solver.pop()
        return status, raw, minimized, self._query_smt2(target_expr), changed, extra, solver_checks

    def check(self, knowledge: list[KnowledgeItem], target: Formula) -> CheckResult:
        # Keep compatibility with callers that pass the current knowledge list.
        # Missing items are asserted once; already-active items are ignored.
        for item in knowledge:
            self.add_knowledge(item)

        target_expr = self._formula_expr(target)
        target_smtlib = target_expr.sexpr()
        solver_instances = 0 if self._reported_solver_instances else 2
        self._reported_solver_instances = True
        knowledge_assertions = self._pending_knowledge_assertions
        self._pending_knowledge_assertions = 0
        solver_checks = 0
        minimization_checks = 0

        consistency, conflict_raw, conflict_min, positive_query, conflict_changed, extra, checks = self._assumption_check(
            target_expr, "positive"
        )
        solver_checks += checks
        minimization_checks += extra
        if consistency == "unsat":
            return CheckResult(
                "contradiction", conflict_min, self.name, "K ∧ F is UNSAT",
                consistency_result=consistency, entailment_result="not_run",
                target_smtlib=target_smtlib,
                consistency_query_smtlib=positive_query,
                entailment_query_smtlib=None,
                raw_dependencies=conflict_raw,
                minimized_dependencies=conflict_min,
                core_minimized=conflict_changed,
                solver_instances=solver_instances, solver_checks=solver_checks,
                knowledge_assertions=knowledge_assertions, core_minimization_checks=minimization_checks,
            )

        entailment, support_raw, support_min, negative_query, support_changed, extra, checks = self._assumption_check(
            z3.Not(target_expr), "negative"
        )
        solver_checks += checks
        minimization_checks += extra
        if entailment == "unsat":
            return CheckResult(
                "valid", support_min, self.name, "K ∧ ¬F is UNSAT",
                consistency_result=consistency, entailment_result=entailment,
                target_smtlib=target_smtlib,
                consistency_query_smtlib=positive_query,
                entailment_query_smtlib=negative_query,
                raw_dependencies=support_raw,
                minimized_dependencies=support_min,
                core_minimized=support_changed,
                solver_instances=solver_instances, solver_checks=solver_checks,
                knowledge_assertions=knowledge_assertions, core_minimization_checks=minimization_checks,
            )

        detail = "Z3 returned unknown for at least one check" if (consistency == "unknown" or entailment == "unknown") else "Both K ∧ F and K ∧ ¬F are SAT"
        return CheckResult(
            "ungrounded", [], self.name, detail,
            consistency_result=consistency, entailment_result=entailment,
            target_smtlib=target_smtlib,
            consistency_query_smtlib=positive_query,
            entailment_query_smtlib=negative_query,
            solver_instances=solver_instances, solver_checks=solver_checks,
            knowledge_assertions=knowledge_assertions, core_minimization_checks=minimization_checks,
        )


class LogicEngine:
    def __init__(self, prefer_z3: bool = True) -> None:
        if prefer_z3 and z3 is not None:
            self.backend = Z3Engine()
        else:
            self.backend = HornEngine()

    @property
    def name(self) -> str:
        return self.backend.name

    def add_knowledge(self, item: KnowledgeItem) -> None:
        add = getattr(self.backend, "add_knowledge", None)
        if add is not None:
            add(item)

    def check(self, knowledge: list[KnowledgeItem], target: Formula) -> CheckResult:
        return self.backend.check(knowledge, target)

    def formula_smtlib(self, formula: Formula) -> str:
        return self.backend.formula_smtlib(formula)
