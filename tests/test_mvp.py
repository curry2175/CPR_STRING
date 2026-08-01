import json
import pytest
from pathlib import Path

from vrg.parser import parse_question, parse_statement
from vrg.verifier import verify_case, verify_case_incremental, plan_incremental_revalidation
from vrg.batch import BatchOptions, evaluate_cases, parse_jsonl
from vrg.benchmark import BenchmarkOptions, evaluate_incremental_scenarios, load_builtin_scenarios
from vrg.preflight import preflight_case
from vrg.mutation import mutation_test_case
from vrg.audit import build_audit_package
from vrg.ingest import split_llm_response, build_case_from_raw
from vrg.dataset import DatasetOptions, parse_dataset_jsonl, adapt_records, evaluate_dataset_records
from vrg.audit_verify import verify_audit_package_bytes
from vrg.atomize import atomize_case, split_atomic_candidates
from vrg.gold_eval import GoldEvalOptions, parse_gold_jsonl, evaluate_gold_records, build_review_queue
from vrg.proofwriter import analyze_proofwriter, normalize_proofwriter_label, extract_query_statement

ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def node_map(result):
    return {node["id"]: node for node in result["nodes"]}


def test_parser_basic():
    assert parse_statement("Bob is quiet.").formula is not None
    assert parse_statement("Quiet people are kind.").formula is not None
    assert parse_statement("If something is big, then it visits the rabbit.").formula is not None
    assert parse_question("Does the mouse visit the rabbit?").formula is not None


def test_valid_sample():
    result = verify_case(load("sample_valid.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s4"]["proof_status"] == "valid"
    assert nodes["s4"]["chain_status"] == "valid"
    assert nodes["final"]["chain_status"] == "valid"
    assert result["answer_correct"] is True
    assert result["summary"]["valid_answer_but_invalid_reasoning"] is False


def test_isolated_ungrounded_is_detected_without_blocking_final():
    result = verify_case(load("sample_ungrounded.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s2"]["proof_status"] == "ungrounded"
    assert nodes["s2"]["chain_status"] == "ungrounded"
    assert nodes["final"]["proof_status"] == "valid"
    assert nodes["final"]["chain_status"] == "valid"
    assert result["summary"]["valid_answer_but_invalid_reasoning"] is True


def test_isolated_contradiction_is_detected_without_blocking_final():
    result = verify_case(load("sample_contradiction.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s2"]["proof_status"] == "contradiction"
    assert nodes["s2"]["reasoning_conflict_dependencies"] == ["p2", "s1"]
    assert nodes["final"]["chain_status"] == "valid"


def test_rejects_maybe():
    case = load("sample_valid.json")
    case["gold_answer"] = "Maybe"
    try:
        verify_case(case, prefer_z3=False)
    except ValueError as exc:
        assert "strictly Yes or No" in str(exc)
    else:
        raise AssertionError("Maybe must be rejected")


def test_v004_separates_chain_from_global_proof():
    result = verify_case(load("sample_valid.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s1"]["source_matches"] == ["p1"]
    assert nodes["s2"]["reasoning_dependencies"] == ["p2", "s1"]
    assert nodes["s2"]["proof_dependencies"] == ["p1", "p2"]
    assert nodes["s1"]["chain_reaches_final"] is True
    assert nodes["s1"]["logical_final_changes_if_removed"] is False
    assert nodes["s1"]["alternative_proof_exists"] is True
    assert result["schema_version"] == "0.15.0"


def test_ungrounded_upstream_blocks_llm_chain_but_not_answer_proof():
    result = verify_case(load("sample_alternative_path_ungrounded.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s1"]["proof_status"] == "ungrounded"
    assert nodes["s2"]["proof_status"] == "valid"
    assert nodes["s2"]["chain_status"] == "blocked_by_upstream_error"
    assert nodes["s2"]["upstream_error_nodes"] == ["s1"]
    assert nodes["final"]["proof_status"] == "valid"
    assert nodes["final"]["chain_status"] == "blocked_by_upstream_error"
    assert result["summary"]["valid_answer_but_invalid_reasoning"] is True


def test_contradictory_upstream_blocks_llm_chain_but_alternative_proof_survives():
    result = verify_case(load("sample_alternative_path_contradiction.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s1"]["proof_status"] == "contradiction"
    assert nodes["s2"]["proof_status"] == "valid"
    assert nodes["s2"]["chain_status"] == "blocked_by_upstream_error"
    assert nodes["final"]["proof_status"] == "valid"
    assert nodes["final"]["chain_status"] == "blocked_by_upstream_error"
    assert result["summary"]["root_error_nodes"] == ["s1"]


def test_v004_exposes_transparent_solver_fields():
    result = verify_case(load("sample_valid.json"), prefer_z3=False)
    nodes = node_map(result)
    s2 = nodes["s2"]
    assert s2["smtlib_formula"] == "(visit tiger mouse)"
    assert s2["consistency_check_result"] == "sat"
    assert s2["entailment_check_result"] == "unsat"
    assert "(check-sat)" in s2["consistency_query_smtlib"]
    assert s2["proof_dependencies_raw"] == ["p1", "p2"]


def test_v004_lists_a_concrete_alternative_path():
    result = verify_case(load("sample_valid.json"), prefer_z3=False)
    nodes = node_map(result)
    assert ["p2", "s1"] in nodes["s2"]["alternative_proof_paths"]


def test_v004_separates_strict_break_from_repairability():
    result = verify_case(load("sample_valid.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["s1"]["strict_chain_breaks_if_removed"] is True
    assert nodes["s1"]["strict_chain_affected_nodes"] == 4
    assert nodes["s1"]["chain_repairable"] is True
    assert nodes["s1"]["logical_answer_preserved_if_removed"] is True
    assert nodes["p1"]["chain_repairable"] is False
    assert nodes["p1"]["logical_answer_preserved_if_removed"] is False


def test_v005_same_as_canonicalizes_before_verification():
    result = verify_case(load("sample_semantic_same_as.json"), prefer_z3=False)
    nodes = node_map(result)
    assert result["schema_version"] == "0.15.0"
    assert nodes["p1"]["raw_formal"] == "consume(mouse, apple)"
    assert nodes["p1"]["formal"] == "eat(mouse, apple)"
    assert nodes["p1"]["semantic_normalizations"] == ["m1"]
    assert nodes["s1"]["proof_status"] == "valid"
    assert nodes["final"]["proof_status"] == "valid"
    assert nodes["m1"]["semantic_proof_usable"] is True


def test_v005_implies_becomes_explicit_bridge_rule():
    result = verify_case(load("sample_semantic_implies.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["m1"]["semantic_relation_type"] == "implies"
    assert nodes["m1"]["semantic_proof_usable"] is True
    assert nodes["s1"]["proof_status"] == "valid"
    assert "m1" in nodes["s1"]["proof_dependencies"]
    assert nodes["final"]["proof_status"] == "valid"


def test_v005_related_to_is_advisory_not_proof():
    result = verify_case(load("sample_semantic_related.json"), prefer_z3=False)
    nodes = node_map(result)
    assert nodes["m1"]["semantic_proof_usable"] is False
    assert nodes["s1"]["proof_status"] == "contradiction"
    assert nodes["s1"]["semantic_hints"]
    assert nodes["s1"]["semantic_hints"][0]["relation_id"] == "m1"
    assert "m1" not in nodes["s1"]["proof_dependencies"]
    assert nodes["final"]["proof_status"] == "contradiction"


def test_v005_disabling_same_as_removes_semantic_entailment():
    case = load("sample_semantic_same_as.json")
    baseline = verify_case(case, prefer_z3=False)
    nodes = node_map(baseline)
    assert nodes["m1"]["logical_final_changes_if_removed"] is True
    assert nodes["m1"]["logical_answer_preserved_if_removed"] is False


def test_v006_parse_jsonl_and_batch_summary():
    jsonl = (ROOT / "data" / "batch_sample.jsonl").read_text(encoding="utf-8")
    cases = parse_jsonl(jsonl)
    assert len(cases) == 9
    result = evaluate_cases(cases, BatchOptions(prefer_z3=False, compute_counterfactuals=False))
    summary = result["summary"]
    assert result["schema_version"] == "0.15.0"
    assert summary["total_cases"] == 9
    assert summary["completed_cases"] == 9
    assert summary["failed_cases"] == 0
    assert summary["answer_correct_count"] == 8
    assert summary["answer_accuracy_percent"] == 88.89
    assert summary["engine_distribution"] == {"horn-fallback": 9}


def test_v006_batch_keeps_reasoning_and_answer_metrics_separate():
    cases = [
        load("sample_alternative_path_ungrounded.json"),
        load("sample_semantic_related.json"),
    ]
    result = evaluate_cases(cases, BatchOptions(prefer_z3=False, compute_counterfactuals=False))
    rows = {row["case_id"]: row for row in result["cases"]}
    assert rows["alternative_path_ungrounded"]["answer_correct"] is True
    assert rows["alternative_path_ungrounded"]["valid_answer_but_invalid_reasoning"] is True
    assert rows["semantic_related_only"]["answer_correct"] is False
    assert rows["semantic_related_only"]["final_proof_status"] == "contradiction"


def test_v006_bad_case_does_not_stop_batch():
    good = load("sample_valid.json")
    bad = load("sample_valid.json")
    bad["id"] = "bad_maybe"
    bad["gold_answer"] = "Maybe"
    result = evaluate_cases([good, bad], BatchOptions(prefer_z3=False))
    assert result["summary"]["completed_cases"] == 1
    assert result["summary"]["failed_cases"] == 1
    assert result["errors"][0]["case_id"] == "bad_maybe"


def test_v007_plan_reuses_unchanged_reasoning_prefix():
    previous = load("sample_long_chain_incremental.json")
    updated = json.loads(json.dumps(previous))
    updated["llm_output"]["reasoning_steps"][7]["text"] = "Bob is not confident."
    plan = plan_incremental_revalidation(previous, updated, verify_case(previous, prefer_z3=False, compute_counterfactuals=False))
    assert plan["mode"] == "suffix_incremental"
    assert plan["reuse_reasoning_ids"] == ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    assert plan["changed_node_ids"] == ["s8"]


def test_v007_incremental_result_matches_full_and_marks_reused_nodes():
    previous = load("sample_long_chain_incremental.json")
    previous_result = verify_case(previous, prefer_z3=False, compute_counterfactuals=False)
    updated = json.loads(json.dumps(previous))
    updated["llm_output"]["reasoning_steps"][7]["text"] = "Bob is not confident."
    result = verify_case_incremental(
        previous, updated, previous_result, prefer_z3=False, validate_against_full=True
    )
    nodes = node_map(result)
    assert result["incremental"]["mode"] == "suffix_incremental"
    assert result["incremental"]["reused_reasoning_count"] == 7
    assert result["incremental"]["revalidated_reasoning_count"] == 4
    assert result["incremental"]["parity_validation"]["matches_full_verification"] is True
    assert nodes["s7"]["verification_origin"] == "reused_unaffected"
    assert nodes["s8"]["verification_origin"] == "revalidated"
    assert nodes["s8"]["proof_status"] == "contradiction"
    assert nodes["final"]["chain_status"] == "valid"


def test_v007_predicted_answer_edit_rechecks_final_only():
    previous = load("sample_valid.json")
    previous_result = verify_case(previous, prefer_z3=False, compute_counterfactuals=False)
    updated = json.loads(json.dumps(previous))
    updated["llm_output"]["answer"] = "No"
    result = verify_case_incremental(
        previous, updated, previous_result, prefer_z3=False, validate_against_full=True
    )
    assert result["incremental"]["mode"] == "final_only"
    assert result["incremental"]["reused_reasoning_count"] == 4
    assert result["incremental"]["revalidated_reasoning_count"] == 0
    assert result["summary"]["final_proof_status"] == "contradiction"


def test_v007_premise_edit_uses_full_fallback():
    previous = load("sample_valid.json")
    previous_result = verify_case(previous, prefer_z3=False, compute_counterfactuals=False)
    updated = json.loads(json.dumps(previous))
    updated["premises"][0]["text"] = "The mouse does not eat the tiger."
    result = verify_case_incremental(
        previous, updated, previous_result, prefer_z3=False, validate_against_full=True
    )
    assert result["incremental"]["mode"] == "full_fallback"
    assert result["incremental"]["reused_reasoning_count"] == 0



def test_v008_graph_local_revalidates_only_affected_branch_and_reuses_final():
    previous = load("sample_branched_graph_local.json")
    previous_result = verify_case(previous, prefer_z3=False, compute_counterfactuals=False)
    updated = json.loads(json.dumps(previous))
    updated["llm_output"]["reasoning_steps"][1]["text"] = "Bob is not focused."
    result = verify_case_incremental(
        previous, updated, previous_result, prefer_z3=False, validate_against_full=True
    )
    inc = result["incremental"]
    nodes = node_map(result)
    assert inc["mode"] == "graph_local_incremental"
    assert inc["revalidated_reasoning_node_ids"] == ["s2", "s3", "s4"]
    assert inc["reused_reasoning_count"] == 5
    assert inc["final_reused"] is True
    assert inc["scope_reduction_percent"] == 57.14
    assert inc["parity_validation"]["matches_full_verification"] is True
    assert nodes["s5"]["verification_origin"] == "reused_unaffected"
    assert nodes["final"]["verification_origin"] == "reused_unaffected"


def test_v008_edit_diff_reports_status_and_root_error_change():
    previous = load("sample_branched_graph_local.json")
    previous_result = verify_case(previous, prefer_z3=False, compute_counterfactuals=False)
    updated = json.loads(json.dumps(previous))
    updated["llm_output"]["reasoning_steps"][1]["text"] = "Bob is not focused."
    result = verify_case_incremental(
        previous, updated, previous_result, prefer_z3=False, validate_against_full=True
    )
    diff = result["edit_diff"]
    assert diff["text_changes"][0]["node_id"] == "s2"
    assert any(change["node_id"] == "s2" for change in diff["status_changes"])
    assert diff["new_root_error_nodes"] == ["s2"]
    assert diff["final_changed"] is False


def test_v008_builtin_incremental_benchmark_has_full_parity():
    scenarios = load_builtin_scenarios(ROOT / "data")
    result = evaluate_incremental_scenarios(
        scenarios, BenchmarkOptions(prefer_z3=False, repetitions=2)
    )
    assert result["schema_version"] == "0.15.0"
    assert result["summary"]["scenario_count"] == 4
    assert result["summary"]["parity_pass_count"] == 4
    rows = {row["scenario_id"]: row for row in result["scenarios"]}
    assert rows["branched_unrelated_edit"]["mode"] == "graph_local_incremental"
    assert rows["branched_unrelated_edit"]["final_reused"] is True
    assert rows["final_only_edit"]["revalidated_reasoning_count"] == 0


def test_v009_preflight_reports_parser_coverage_and_semantics():
    result = preflight_case(load("sample_semantic_same_as.json"))
    assert result["schema_version"] == "0.15.0"
    assert result["ready_for_verification"] is True
    assert result["summary"]["parser_coverage_percent"] == 100.0
    premise = next(row for row in result["statements"] if row["id"] == "p1")
    assert premise["raw_formal"] == "consume(mouse, apple)"
    assert premise["normalized_formal"] == "eat(mouse, apple)"
    assert premise["semantic_normalizations"] == ["m1"]


def test_v009_preflight_catches_bad_question_and_answer():
    case = load("sample_valid.json")
    case["question"] = "Could the mouse perhaps visit the rabbit?"
    case["llm_output"]["answer"] = "Maybe"
    result = preflight_case(case)
    assert result["ready_for_verification"] is False
    assert result["summary"]["question_parseable"] is False
    assert any("strictly Yes or No" in error for error in result["errors"])


def test_v009_mutation_test_detects_flip_and_novel_predicate():
    result = mutation_test_case(load("sample_valid.json"), prefer_z3=False, max_nodes=2)
    assert result["schema_version"] == "0.15.0"
    assert result["summary"]["eligible_reasoning_nodes"] == 2
    assert result["summary"]["mutation_count"] == 10
    assert result["summary"]["mutation_type_summary"]["wrong_declared_parent"]["detection_percent"] == 100.0
    assert result["summary"]["overall_detection_percent"] == 100.0
    assert all(row["parity_match"] is True for row in result["mutations"])


def test_v009_audit_package_contains_replay_material(tmp_path):
    case = load("sample_valid.json")
    result = verify_case(case, prefer_z3=False, compute_counterfactuals=False)
    path = build_audit_package(case, result, tmp_path)
    assert path.exists()
    import zipfile
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert "input_case.json" in names
        assert "verified_graph.json" in names
        assert "report.html" in names
        assert "nodes.csv" in names
        assert "edges.csv" in names
        assert "smt2/index.json" in names
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == "0.15.0"
        assert manifest["case_id"] == "proofwriter_mouse_valid"


def test_v009_persistent_z3_session_uses_two_solver_instances_when_available():
    try:
        import z3  # noqa: F401
    except ImportError:
        return
    previous = load("sample_long_chain_incremental.json")
    previous_result = verify_case(previous, prefer_z3=True, compute_counterfactuals=False)
    updated = json.loads(json.dumps(previous))
    updated["llm_output"]["reasoning_steps"][7]["text"] = "Bob is not confident."
    result = verify_case_incremental(previous, updated, previous_result, prefer_z3=True, validate_against_full=False)
    stats = result["incremental"]["solver_stats"]
    assert stats["backend_solver_instances"] == 2
    assert "Persistent per-case Z3 session" in stats["strategy"]



def test_v010_raw_llm_ingest_extracts_steps_answer_and_preflight():
    payload = load("raw_ingest_sample.json")
    result = build_case_from_raw(payload)
    assert result["schema_version"] == "0.15.0"
    assert result["case"]["llm_output"]["answer"] == "Yes"
    assert len(result["case"]["llm_output"]["reasoning_steps"]) == 3
    assert result["preflight"]["ready_for_verification"] is True
    assert result["preflight"]["summary"]["parser_coverage_percent"] == 100.0


def test_v010_response_splitter_rejects_missing_yes_no():
    try:
        split_llm_response("1. Bob is quiet.\n2. Bob is kind.")
    except ValueError as exc:
        assert "strict Yes/No" in str(exc)
    else:
        raise AssertionError("Missing final Yes/No must be rejected")


def test_v010_dataset_adapter_handles_mixed_formats_and_keeps_failures():
    records = parse_dataset_jsonl((ROOT / "data" / "dataset_sample.jsonl").read_text(encoding="utf-8"))
    result = adapt_records(records)
    assert result["schema_version"] == "0.15.0"
    assert result["summary"]["total_records"] == 5
    assert result["summary"]["adapted_records"] == 4
    assert result["summary"]["failed_records"] == 1
    assert result["summary"]["ready_for_verification_count"] == 4
    assert result["errors"][0]["case_id"] == "unsupported_unknown_label"


def test_v010_dataset_evaluation_produces_research_metrics():
    records = parse_dataset_jsonl((ROOT / "data" / "dataset_sample.jsonl").read_text(encoding="utf-8"))
    result = evaluate_dataset_records(records, DatasetOptions(prefer_z3=False))
    summary = result["research_summary"]
    assert summary["input_records"] == 5
    assert summary["adapted_records"] == 4
    assert summary["verified_cases"] == 4
    assert summary["answer_accuracy_percent"] == 100.0
    assert len(result["rows"]) == 4


def test_v010_audit_verifier_checks_hashes_and_metadata(tmp_path):
    case = load("sample_multiple_proofs.json")
    result = verify_case(case, prefer_z3=False, compute_counterfactuals=False)
    package = build_audit_package(case, result, tmp_path)
    checked = verify_audit_package_bytes(package.read_bytes(), rerun_smt=False)
    assert checked["schema_version"] == "0.15.0"
    assert checked["summary"]["integrity_pass"] is True
    assert checked["summary"]["hash_match_count"] == checked["summary"]["hash_check_count"]
    assert checked["summary"]["metadata_match_count"] == checked["summary"]["metadata_check_count"]


def test_v010_audit_verifier_detects_tampering(tmp_path):
    import io
    import zipfile
    case = load("sample_valid.json")
    result = verify_case(case, prefer_z3=False, compute_counterfactuals=False)
    package = build_audit_package(case, result, tmp_path)
    source = zipfile.ZipFile(package)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            payload = source.read(name)
            if name == "nodes.csv":
                payload += b"tampered"
            target.writestr(name, payload)
    checked = verify_audit_package_bytes(buffer.getvalue(), rerun_smt=False)
    assert checked["summary"]["integrity_pass"] is False
    assert any(row["path"] == "nodes.csv" and row["match"] is False for row in checked["hash_checks"])


def test_v010_declared_dependencies_override_inferred_chain_path():
    case = load("sample_multiple_proofs.json")
    case["llm_output"]["reasoning_steps"][1]["depends_on"] = ["p3", "p4"]
    case["llm_output"]["answer_depends_on"] = ["s2"]
    result = verify_case(case, prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(result)
    assert nodes["s2"]["chain_dependency_source"] == "declared"
    assert nodes["s2"]["declared_reasoning_dependencies"] == ["p3", "p4"]
    assert nodes["s2"]["reasoning_dependencies"] == ["p3", "p4"]
    assert nodes["s2"]["inferred_reasoning_dependencies"] == ["p2", "s1"]
    assert nodes["s2"]["chain_status"] == "valid"


def test_v010_preflight_rejects_future_declared_dependency():
    case = load("sample_valid.json")
    case["llm_output"]["reasoning_steps"][0]["depends_on"] = ["s4"]
    result = preflight_case(case)
    assert result["ready_for_verification"] is False
    assert any("unknown or future" in error for error in result["errors"])


def test_v010_mutation_locks_original_chain_and_blocks_final():
    result = mutation_test_case(load("sample_long_chain_incremental.json"), prefer_z3=False, max_nodes=1)
    assert result["summary"]["strict_authored_path_mode"] is True
    assert result["summary"]["final_chain_blocked_or_invalid_count"] == result["summary"]["mutation_count"]
    assert all(row["final_chain_status"] == "blocked_by_upstream_error" for row in result["mutations"])


def test_v012_declared_parent_must_locally_support_claim():
    result = verify_case(load("sample_declared_insufficient.json"), prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(result)
    assert nodes["s2"]["proof_status"] == "valid"
    assert nodes["s2"]["chain_status"] == "insufficient_declared_support"
    assert nodes["s2"]["declared_dependency_sufficient"] is False
    assert nodes["final"]["chain_status"] == "blocked_by_upstream_error"
    assert result["summary"]["valid_answer_but_invalid_reasoning"] is True


def test_v012_inferred_dependency_ambiguity_and_minimal_proofs():
    result = verify_case(load("sample_dependency_ambiguity.json"), prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(result)
    assert nodes["s1"]["dependency_confidence"] == "inferred_ambiguous"
    assert nodes["s1"]["dependency_candidate_count"] >= 2
    assert nodes["s1"]["minimal_proof_count"] == 2
    assert result["summary"]["ambiguous_dependency_step_count"] == 1
    assert result["summary"]["final_minimal_proof_count"] >= 2


def test_v012_reasoning_quality_profile_marks_irrelevant_restatement():
    result = verify_case(load("sample_reasoning_quality.json"), prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(result)
    assert nodes["s2"]["reasoning_role"] == "premise_restatement"
    assert nodes["s2"]["final_proof_necessity"] == "not_in_minimal_final_proof"
    assert result["reasoning_quality_profile"]["premise_restatement_percent"] > 0
    assert result["summary"]["reasoning_integrity_score"] > 0


def test_v012_compound_step_is_rejected_and_flagged():
    case = load("sample_compound_reasoning.json")
    preflight = preflight_case(case)
    assert preflight["summary"]["compound_reasoning_step_count"] == 1
    result = verify_case(case, prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(result)
    assert nodes["s1"]["proof_status"] == "untranslatable"
    assert nodes["s1"]["atomicity_status"] == "compound"


def test_v015_atomizer_splits_safe_compound_step_and_preserves_origin():
    case = load("sample_compound_reasoning.json")
    result = atomize_case(case, apply=True)
    assert result["schema_version"] == "0.15.0"
    assert result["summary"]["compound_steps_split"] == 1
    steps = result["atomized_case"]["llm_output"]["reasoning_steps"]
    assert [row["id"] for row in steps] == ["s1a", "s1b"]
    assert all(row["origin_step_id"] == "s1" for row in steps)
    verified = verify_case(result["atomized_case"], prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(verified)
    assert nodes["s1a"]["proof_status"] == "valid"
    assert nodes["s1b"]["proof_status"] == "valid"


def test_v015_atomizer_does_not_split_atomic_step():
    case = load("sample_valid.json")
    result = atomize_case(case, apply=True)
    assert result["summary"]["compound_steps_split"] == 0
    assert result["summary"]["original_step_count"] == result["summary"]["atomized_step_count"]


def test_v015_gold_benchmark_metrics_and_parent_paths():
    records = parse_gold_jsonl((ROOT / "data" / "gold_reasoning_benchmark.jsonl").read_text(encoding="utf-8"))
    result = evaluate_gold_records(records, GoldEvalOptions(prefer_z3=False))
    assert result["schema_version"] == "0.15.0"
    assert result["summary"]["evaluated_cases"] == 8
    assert result["summary"]["proof_status"]["accuracy_percent"] == 100.0
    assert result["summary"]["chain_status"]["accuracy_percent"] == 100.0
    assert result["summary"]["parent_path_exact_accuracy_percent"] == 100.0
    assert result["summary"]["root_error_localization"]["f1"] == 100.0


def test_v015_gold_evaluator_detects_wrong_annotation():
    records = parse_gold_jsonl((ROOT / "data" / "gold_reasoning_benchmark.jsonl").read_text(encoding="utf-8"))
    records[0]["gold"]["nodes"]["s1"]["proof_status"] = "ungrounded"
    result = evaluate_gold_records(records, GoldEvalOptions(prefer_z3=False))
    assert result["summary"]["proof_status"]["accuracy_percent"] < 100.0


def test_v015_review_queue_surfaces_ambiguity_and_path_errors():
    records = parse_gold_jsonl((ROOT / "data" / "gold_reasoning_benchmark.jsonl").read_text(encoding="utf-8"))
    result = evaluate_gold_records(records, GoldEvalOptions(prefer_z3=False))
    reasons = " ".join(row["review_reasons"] for row in result["review_queue"])
    assert "inferred_ambiguous" in reasons
    assert "insufficient_declared_support" in reasons
    assert "ungrounded" in reasons



def test_v015_proofwriter_conjunctive_rule_grammar():
    rule = parse_statement("All blue, green people are red.").formula
    assert rule is not None
    assert len(rule.antecedents) == 2
    assert {atom.predicate for atom in rule.antecedents} == {"blue", "green"}
    conditional = parse_statement("If someone is quiet and red then they are blue.").formula
    assert conditional is not None
    assert {atom.predicate for atom in conditional.antecedents} == {"quiet", "red"}


def test_v015_proofwriter_label_and_wrapped_query():
    assert normalize_proofwriter_label("B") == "False"
    assert normalize_proofwriter_label("C) Unknown") == "Unknown"
    query = extract_query_statement(
        "Based on the above information, is the following statement true, false, or unknown? Charlie is not red."
    )
    assert query == "Charlie is not red."


def test_v015_real_proofwriter_record_false_and_profile():
    payload = load("proofwriter_real_sample.json")
    result = analyze_proofwriter({**payload, "prefer_z3": False})
    classification = result["classification"]
    assert classification["label"] == "False"
    assert classification["gold_label"] == "False"
    assert classification["predicted_label"] == "False"
    assert classification["answer_correct"] is True
    assert classification["selected_dependencies"] == ["p1", "p12", "p15", "p16"]
    graph = result["verified_graph"]
    nodes = node_map(graph)
    assert nodes["s1"]["reasoning_dependencies"] == ["p1"]
    assert nodes["s2"]["reasoning_dependencies"] == ["p15", "s1"]
    assert nodes["s3"]["reasoning_dependencies"] == ["p12", "s2"]
    assert nodes["s4"]["reasoning_dependencies"] == ["p16", "s1", "s3"]
    assert nodes["final"]["reasoning_dependencies"] == ["s4"]
    fingerprint = result["reasoning_fingerprint"]
    assert fingerprint["grounding"]["logical_premise_utilization_percent"] == 23.53
    assert fingerprint["grounding"]["distractor_premise_count"] == 13
    assert fingerprint["error_profile"]["root_error_count"] == 0


def test_v015_proofwriter_unknown_open_world():
    payload = {
        "record": {
            "id": "unknown_case",
            "context": "Bob is quiet. Quiet people are kind.",
            "question": "Is the following statement true, false, or unknown? Bob is blue.",
            "answer": "C",
        },
        "llm_response": "Bob is quiet. Final answer: Unknown",
        "prefer_z3": False,
    }
    result = analyze_proofwriter(payload)
    assert result["classification"]["label"] == "Unknown"
    assert result["classification"]["answer_correct"] is True
    final = node_map(result["verified_graph"])["final"]
    assert final["text"] == "Answer: Unknown"
    assert final["proof_status"] == "valid"

from types import SimpleNamespace

from vrg.openai_runner import (
    GeneratedProofWriterOutput,
    GeneratedReasoningStep,
    build_proofwriter_prompt,
    generate_and_analyze_proofwriter,
    normalize_generated_output,
)


def test_v015_openai_prompt_does_not_leak_gold_answer():
    payload = load("proofwriter_real_sample.json")
    record = payload["record"]
    prompt = build_proofwriter_prompt(record)
    assert prompt["gold_answer_was_sent"] is False
    assert "p1: Charlie is green." in prompt["user"]
    assert "Charlie is not red." in prompt["user"]
    assert "B) False" not in prompt["user"]
    assert "\"answer\": \"B\"" not in prompt["user"]


def test_v015_normalizes_model_ids_and_preserves_valid_dependencies():
    generated = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(id="step-a", text="Charlie is green", depends_on=["p1"]),
            GeneratedReasoningStep(id="step-b", text="Charlie is quiet", depends_on=["p15", "step-a", "future"]),
        ],
        final_answer="False",
        answer_depends_on=["step-b"],
    )
    normalized = normalize_generated_output(generated, {"p1", "p15"})
    steps = normalized["llm_output"]["reasoning_steps"]
    assert steps[0]["id"] == "s1"
    assert steps[1]["id"] == "s2"
    assert steps[1]["depends_on"] == ["p15", "s1"]
    assert steps[1]["invalid_model_dependencies"] == ["future"]
    assert normalized["llm_output"]["answer_depends_on"] == ["s2"]


class _FakeResponses:
    def parse(self, **kwargs):
        parsed = GeneratedProofWriterOutput(
            reasoning_steps=[
                GeneratedReasoningStep(id="s1", text="Charlie is green.", depends_on=["p1"]),
                GeneratedReasoningStep(id="s2", text="Charlie is quiet.", depends_on=["p15", "s1"]),
                GeneratedReasoningStep(id="s3", text="Charlie is blue.", depends_on=["p12", "s2"]),
                GeneratedReasoningStep(id="s4", text="Charlie is red.", depends_on=["p16", "s1", "s3"]),
            ],
            final_answer="False",
            answer_depends_on=["s4"],
        )
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 200, "output_tokens": 80, "total_tokens": 280})
        return SimpleNamespace(
            id="resp_test",
            model=kwargs["model"],
            status="completed",
            output=[message],
            usage=usage,
        )


class _FakeOpenAIClient:
    def __init__(self):
        self.responses = _FakeResponses()


def test_v015_openai_generation_flows_directly_into_vrg():
    payload = load("proofwriter_real_sample.json")
    result = generate_and_analyze_proofwriter(
        {
            "record": payload["record"],
            "model": "gpt-5.6",
            "reasoning_effort": "low",
            "repetitions": 1,
            "max_output_tokens": 4000,
            "prefer_z3": False,
        },
        client=_FakeOpenAIClient(),
    )
    assert result["schema_version"] == "0.15.0"
    assert result["summary"]["gold_answer_sent_to_model"] is False
    assert result["summary"]["gold_correct_count"] == 1
    assert result["summary"]["context_match_count"] == 1
    run = result["runs"][0]
    assert run["generation"]["response_id"] == "resp_test"
    assert run["generation"]["usage"]["total_tokens"] == 280
    analysis = run["analysis"]
    assert analysis["classification"]["predicted_label"] == "False"
    assert analysis["classification"]["answer_correct"] is True
    nodes = node_map(analysis["verified_graph"])
    assert nodes["s2"]["chain_dependency_source"] == "declared"
    assert nodes["s2"]["declared_dependency_sufficient"] is True
    assert nodes["final"]["chain_status"] == "valid"

from vrg.hybrid_runner import run_hybrid_proofwriter
from vrg.universal_graph import build_universal_graph
from vrg.openai_runner import GeneratedProofWriterOutput, GeneratedReasoningStep


class _HybridSequenceResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.outputs.pop(0)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140})
        return SimpleNamespace(id=f"resp_{len(self.calls)}", model=kwargs["model"], status="completed", output=[message], usage=usage)


class _HybridSequenceClient:
    def __init__(self, outputs):
        self.responses = _HybridSequenceResponses(outputs)


def test_v016_hybrid_preserves_universal_graph_layers_on_valid_case():
    payload = load("proofwriter_real_sample.json")
    good = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(id="s1", text="Charlie is quiet.", depends_on=["p1", "p15"]),
            GeneratedReasoningStep(id="s2", text="Charlie is blue.", depends_on=["s1", "p12"]),
            GeneratedReasoningStep(id="s3", text="Charlie is red.", depends_on=["s2", "p1", "p16"]),
        ],
        final_answer="False",
        answer_depends_on=["s3"],
    )
    result = run_hybrid_proofwriter({
        "record": payload["record"], "model": "gpt-5.6", "prefer_z3": False,
        "use_llm_formalizer": False, "use_premise_grounder": False, "max_repair_iterations": 1,
    }, client=_HybridSequenceClient([good]))
    assert result["schema_version"] == "0.23.0"
    assert result["summary"]["initial_pass"] is True
    graph = result["final_universal_graph"]
    assert graph["universal_viewer"]["legend_always_visible"] is True
    assert "contradiction" in graph["universal_viewer"]["status_counts"]
    relations = {x["relation"] for x in graph["edges"]}
    assert "authored_dependency" in relations
    assert "proof_support" in relations


def test_v016_graph_guided_repair_rechecks_new_output():
    payload = load("proofwriter_real_sample.json")
    bad = GeneratedProofWriterOutput(
        reasoning_steps=[GeneratedReasoningStep(id="s1", text="Charlie is not red.", depends_on=["p1"])],
        final_answer="True", answer_depends_on=["s1"],
    )
    good = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(id="s1", text="Charlie is quiet.", depends_on=["p1", "p15"]),
            GeneratedReasoningStep(id="s2", text="Charlie is blue.", depends_on=["s1", "p12"]),
            GeneratedReasoningStep(id="s3", text="Charlie is red.", depends_on=["s2", "p1", "p16"]),
        ],
        final_answer="False", answer_depends_on=["s3"],
    )
    client = _HybridSequenceClient([bad, good])
    result = run_hybrid_proofwriter({
        "record": payload["record"], "model": "gpt-5.6", "prefer_z3": False,
        "use_llm_formalizer": False, "use_premise_grounder": False,
        "max_repair_iterations": 1, "repair_mode": "blind",
    }, client=client)
    assert result["summary"]["initial_pass"] is False
    assert result["summary"]["final_pass"] is True
    assert result["summary"]["repair_count"] == 1
    assert result["summary"]["initial_answer"] == "True"
    assert result["summary"]["final_answer"] == "False"
    assert result["attempts"][1]["graph_diff_from_previous"]["changed_nodes"]
    repair_prompt = client.responses.calls[1]["input"][1]["content"]
    assert "gold" not in repair_prompt.lower() or "gold answer is not provided" in client.responses.calls[1]["input"][0]["content"].lower()

from vrg.hybrid_formalizer import FormalizationBatch, FormalizationItem, hybrid_formalize_case


class _FormalizerResponses:
    def parse(self, **kwargs):
        parsed = FormalizationBatch(items=[
            FormalizationItem(id="s1", kind="reasoning", controlled_english="Bob is kind.", confidence="medium", notes="Removed vague marker without adding a new claim")
        ])
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 60, "output_tokens": 20, "total_tokens": 80})
        return SimpleNamespace(id="formalize_1", model=kwargs["model"], output=[message], usage=usage)


class _FormalizerClient:
    def __init__(self):
        self.responses = _FormalizerResponses()


def test_v016_deterministic_first_llm_fallback_preserves_original_text():
    case = {
        "id": "fallback_case",
        "premises": [{"id": "p1", "text": "Bob is quiet."}],
        "question": "Is Bob kind?",
        "llm_output": {"reasoning_steps": [{"id": "s1", "text": "Bob is maybe kind."}], "answer": "Yes"},
        "gold_answer": "Yes",
    }
    result = hybrid_formalize_case(case, use_llm_fallback=True, model="gpt-5.6", client=_FormalizerClient())
    assert result["summary"]["llm_fallback_count"] == 1
    assert result["case"]["llm_output"]["reasoning_steps"][0]["text"] == "Bob is kind."
    assert result["metadata"]["s1"]["original_text"] == "Bob is maybe kind."
    assert result["metadata"]["s1"]["formalization_source"] == "llm_fallback"

from vrg.hybrid_runner import run_hybrid_batch


def test_v016_hybrid_batch_runs_proofwriter_jsonl():
    payload = load("proofwriter_real_sample.json")
    good = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(id="s1", text="Charlie is quiet.", depends_on=["p1", "p15"]),
            GeneratedReasoningStep(id="s2", text="Charlie is blue.", depends_on=["s1", "p12"]),
            GeneratedReasoningStep(id="s3", text="Charlie is red.", depends_on=["s2", "p1", "p16"]),
        ],
        final_answer="False", answer_depends_on=["s3"],
    )
    result = run_hybrid_batch({
        "jsonl": __import__('json').dumps(payload["record"]),
        "model": "gpt-5.6", "prefer_z3": False,
        "use_llm_formalizer": False, "use_premise_grounder": False,
        "max_repair_iterations": 0,
    }, client=_HybridSequenceClient([good]))
    assert result["summary"]["input_records"] == 1
    assert result["summary"]["completed_records"] == 1
    assert result["summary"]["final_pass_count"] == 1

from vrg.operational_batch import OperationalBatchManager, TERMINAL_STATUSES, parse_records_input


def test_v017_dataset_input_accepts_json_array_and_jsonl():
    rows = [{"id": "a"}, {"id": "b"}]
    assert len(parse_records_input(json.dumps(rows))) == 2
    assert len(parse_records_input("\n".join(json.dumps(row) for row in rows))) == 2
    assert len(parse_records_input({"records": rows})) == 2


def test_v017_pilot_then_full_resumes_checkpoint(monkeypatch, tmp_path):
    def fake_run(payload):
        record = payload["record"]
        idx = int(record["id"].split("_")[-1])
        repaired = idx % 3 == 0
        return {
            "response_id": f"resp_{idx}",
            "summary": {
                "final_pass": True,
                "initial_pass": not repaired,
                "initial_answer": "True",
                "final_answer": "True",
                "context_label": "True",
                "gold_label": "True",
                "final_answer_correct": True,
                "repair_count": 1 if repaired else 0,
                "attempt_count": 2 if repaired else 1,
                "total_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
            "attempts": [],
            "final_universal_graph": {},
        }

    monkeypatch.setattr("vrg.operational_batch.run_hybrid_proofwriter", fake_run)
    manager = OperationalBatchManager(tmp_path)
    records = [{"id": f"record_{i}", "context": "Bob is quiet.", "question": "Bob is quiet.", "answer": "A"} for i in range(1, 16)]
    first = manager.start({"records": records, "mode": "pilot", "pilot_count": 10, "max_workers": 2, "max_retries": 0})
    run_id = first["run_id"]
    import time as _time
    while manager.status(run_id)["status"] not in TERMINAL_STATUSES:
        _time.sleep(0.01)
    pilot = manager.status(run_id)
    assert pilot["summary"]["processed_records"] == 10
    manager.start({"run_id": run_id, "mode": "full", "max_workers": 2, "max_retries": 0})
    while manager.status(run_id)["status"] not in TERMINAL_STATUSES:
        _time.sleep(0.01)
    full = manager.status(run_id)
    assert full["summary"]["processed_records"] == 15
    assert full["summary"]["total_tokens"] == 225
    run_dir = tmp_path / "hybrid_runs" / run_id
    assert len(list((run_dir / "cases").glob("*.json"))) == 15
    assert (run_dir / "summary.csv").exists()
    assert len(full["phase_history"]) == 2


def test_v017_token_limit_stops_at_checkpoint(monkeypatch, tmp_path):
    def fake_run(payload):
        return {
            "response_id": payload["record"]["id"],
            "summary": {
                "final_pass": True, "initial_pass": True, "initial_answer": "True", "final_answer": "True",
                "context_label": "True", "gold_label": "True", "final_answer_correct": True,
                "repair_count": 0, "attempt_count": 1,
                "total_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
            "attempts": [], "final_universal_graph": {},
        }
    monkeypatch.setattr("vrg.operational_batch.run_hybrid_proofwriter", fake_run)
    manager = OperationalBatchManager(tmp_path)
    records = [{"id": f"r{i}"} for i in range(8)]
    started = manager.start({"records": records, "mode": "full", "max_workers": 1, "max_retries": 0, "max_total_tokens": 30})
    import time as _time
    while manager.status(started["run_id"])["status"] not in TERMINAL_STATUSES:
        _time.sleep(0.01)
    status = manager.status(started["run_id"])
    assert status["status"] == "stopped"
    assert status["summary"]["processed_records"] == 2
    assert status["stop_reason"] == "max_total_tokens"

from urllib.parse import parse_qs, urlparse
from vrg.dataset_download import download_proofwriter_600, proofwriter_download_status


def _proofwriter_record(i: int) -> dict:
    return {
        "id": f"ProofWriter_Test_{i}",
        "context": "Bob is quiet.",
        "question": "Based on the above information, is the following statement true, false, or unknown? Bob is quiet.",
        "options": ["A) True", "B) False", "C) Unknown"],
        "answer": "A",
        "raw_logic_programs": [],
    }


def test_v018_auto_download_paginates_600_rows(tmp_path):
    records = [_proofwriter_record(i) for i in range(600)]
    calls = []

    def fake_fetch(url: str):
        calls.append(url)
        query = parse_qs(urlparse(url).query)
        offset = int(query["offset"][0])
        length = int(query["length"][0])
        rows = [{"row_idx": i, "row": records[i], "truncated_cells": []} for i in range(offset, min(600, offset + length))]
        return {"rows": rows, "num_rows_total": 600}

    result = download_proofwriter_600(tmp_path, fetch_json=fake_fetch)
    assert result.row_count == 600
    assert result.cached is False
    assert result.dataset_path.exists()
    assert len(result.dataset_path.read_text(encoding="utf-8").splitlines()) == 600
    assert len(calls) == 6
    assert proofwriter_download_status(tmp_path)["available"] is True


def test_v018_auto_download_reuses_valid_cache(tmp_path):
    target = tmp_path / "renma_ProofWriter_validation_600.jsonl"
    target.write_text(json.dumps(_proofwriter_record(1)) + "\n", encoding="utf-8")

    def should_not_call(url: str):
        raise AssertionError(url)

    result = download_proofwriter_600(tmp_path, fetch_json=should_not_call)
    assert result.cached is True
    assert result.row_count == 1


def test_v018_operational_run_records_dataset_source(monkeypatch, tmp_path):
    def fake_run(payload):
        return {
            "response_id": "resp",
            "summary": {
                "final_pass": True, "initial_pass": True, "initial_answer": "True", "final_answer": "True",
                "context_label": "True", "gold_label": "True", "final_answer_correct": True,
                "repair_count": 0, "attempt_count": 1,
                "total_usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            },
            "attempts": [], "final_universal_graph": {},
        }

    monkeypatch.setattr("vrg.operational_batch.run_hybrid_proofwriter", fake_run)
    manager = OperationalBatchManager(tmp_path)
    source = {"dataset_id": "renma/ProofWriter", "split": "validation", "row_count": 600}
    started = manager.start({"records": [_proofwriter_record(1)], "mode": "pilot", "dataset_source": source})
    import time as _time
    while manager.status(started["run_id"])["status"] not in TERMINAL_STATUSES:
        _time.sleep(0.01)
    status = manager.status(started["run_id"])
    assert status["dataset_source"] == source
    settings = json.loads((tmp_path / "hybrid_runs" / started["run_id"] / "settings.json").read_text())
    assert settings["dataset_source"] == source


from vrg.proofwriter_logic import parse_raw_logic_program
from vrg.reverify_run import reverify_case


def _v019_relation_record():
    return {
        "id": "ProofWriter_RelNeg-Test_Q1",
        "context": [
            "The bald eagle chases the cat.",
            "If something chases the cat then it sees the dog.",
        ],
        "question": "The bald eagle does not see the dog.",
        "answer": "B",
        "raw_logic_programs": ["""Predicates:
Chases($x, $y, bool) ::: Does x chase y?
Sees($x, $y, bool) ::: Does x see y?

Facts:
Chases(BaldEagle, Cat, True) ::: The bald eagle chases the cat.

Rules:
Chases($x, Cat, True) >>> Sees($x, Dog, True) ::: If something chases the cat then it sees the dog.

Query:
Sees(BaldEagle, Dog, False) ::: The bald eagle does not see the dog."""],
    }


def test_v019_raw_logic_canonicalization_preserves_compound_entities_and_relations():
    record = _v019_relation_record()
    program = parse_raw_logic_program(record)
    assert program is not None
    assert program.query.to_text() == "not see(bald_eagle, dog)"
    assert program.premises[0]["canonical_formula"] == "chase(bald_eagle, cat)"
    context = __import__('vrg.proofwriter', fromlist=['_context_classification'])._context_classification(program.premises, program.query)
    assert context["label"] == "False"
    assert all(info["formalization_source"] == "proofwriter_raw_logic_verified" for key, info in program.metadata.items() if key != "question")


def test_v019_natural_context_wins_when_raw_gloss_conflicts():
    record = _v019_relation_record()
    record["context"][1] = "If something chases the cat then the bald eagle sees the dog."
    # Keep the raw consequent incorrectly tied to x; the model-visible natural context must win.
    program = parse_raw_logic_program(record)
    assert program is not None
    assert program.premises[1]["canonical_formula"].endswith("-> see(bald_eagle, dog)")
    assert program.metadata["p2"]["formalization_source"] in {"context_over_raw_mismatch", "context_parser_raw_missing"}
    context = __import__('vrg.proofwriter', fromlist=['_context_classification'])._context_classification(program.premises, program.query)
    assert context["label"] == "False"


def test_v019_offline_reverification_ignores_legacy_repair_when_initial_now_passes():
    record = _v019_relation_record()
    initial = {
        "reasoning_steps": [
            {"id": "s1", "text": "The bald eagle sees the dog.", "depends_on": ["p1", "p2"]}
        ],
        "answer": "False",
        "answer_depends_on": ["s1"],
    }
    bad_repair = {"reasoning_steps": [], "answer": "Unknown", "answer_depends_on": []}
    legacy = {
        "schema_version": "0.18.0",
        "record_id": record["id"],
        "settings": {"model": "gpt-5.6", "reasoning_effort": "low"},
        "attempts": [
            {"llm_output": initial, "passed": False, "analysis": {"record_formalization": {"record": record}, "verified_graph": {"summary": {}}}},
            {"llm_output": bad_repair, "passed": True, "analysis": {"record_formalization": {"record": record}, "verified_graph": {"summary": {}}}},
        ],
        "summary": {"initial_pass": False, "final_pass": True, "final_answer": "Unknown", "total_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
    }
    corrected = reverify_case(legacy, record, prefer_z3=False)
    assert corrected["summary"]["initial_pass"] is True
    assert corrected["summary"]["selected_attempt_index"] == 0
    assert corrected["summary"]["final_answer"] == "False"
    assert corrected["summary"]["new_api_calls"] == 0

from vrg.experiment import run_fault_injection_experiment, run_natural_repair_experiment, FAULT_TYPES


def test_v020_fault_injection_runs_offline_and_writes_paper_outputs(tmp_path):
    source = ROOT / "outputs" / "hybrid_runs" / "proofwriter_600_v019_reverified"
    if not source.exists():
        pytest.skip("large retained ProofWriter run fixture is not bundled in the slim distribution")
    result = run_fault_injection_experiment(
        source,
        tmp_path,
        sample_count=1,
        seed=2026,
        fault_types=["polarity_flip", "wrong_parent", "answer_flip"],
        difficulties=["local"],
        prefer_z3=False,
    )
    summary = result["summary"]
    assert summary["new_api_calls"] == 0
    assert summary["sampled_clean_cases"] == 1
    assert summary["mutation_count"] >= 2
    out = Path(result["output_dir"])
    assert (out / "fault_mutations.csv").exists()
    assert (out / "paper_table.csv").exists()
    assert (out / "mutation_details.jsonl").exists()


def test_v020_natural_no_repair_condition_requires_no_api(tmp_path):
    source = ROOT / "outputs" / "hybrid_runs" / "proofwriter_600_v019_reverified"
    if not source.exists():
        pytest.skip("large retained ProofWriter run fixture is not bundled in the slim distribution")
    result = run_natural_repair_experiment(
        source,
        tmp_path,
        modes=["no_repair"],
        max_cases=1,
        prefer_z3=False,
    )
    summary = result["summary"]
    assert summary["natural_failure_cases"] == 1
    assert summary["total_api_calls"] == 0
    assert summary["gold_answer_was_sent"] is False
    assert summary["conditions"][0]["condition"] == "no_repair"
    assert (Path(result["output_dir"]) / "paper_table.csv").exists()


def test_v020_fault_catalog_contains_requested_structural_and_semantic_faults():
    assert set(FAULT_TYPES) == {
        "polarity_flip", "entity_swap", "argument_swap", "predicate_swap",
        "parent_deletion", "wrong_parent", "step_deletion", "answer_flip",
    }


from vrg.test_lab import build_custom_record, run_custom_test
import csv
from app import app


def test_v021_strict_authored_chain_rejects_empty_reasoning_parents():
    case = {
        "id": "strict_empty_parent",
        "premises": [{"id": "p1", "text": "Bob is kind."}],
        "question": "Is Bob kind?",
        "llm_output": {
            "reasoning_steps": [{"id": "s1", "text": "Bob is kind.", "depends_on": []}],
            "answer": "Yes",
            "answer_depends_on": ["s1"],
        },
        "gold_answer": "Yes",
        "verification_policy": {
            "require_declared_reasoning_dependencies": True,
            "require_declared_answer_dependencies": True,
        },
    }
    result = verify_case(case, prefer_z3=False, compute_counterfactuals=False)
    nodes = node_map(result)
    assert nodes["s1"]["proof_status"] == "valid"
    assert nodes["s1"]["chain_status"] == "insufficient_declared_support"
    assert "s1" in result["summary"]["root_error_nodes"]


def test_v021_custom_input_builder_needs_no_raw_json():
    record, gold_provided = build_custom_record({
        "context": "Bob is quiet.\nAll quiet people are kind.",
        "question": "Bob is kind.",
        "gold_answer": "",
    })
    assert record["context"] == ["Bob is quiet.", "All quiet people are kind."]
    assert record["answer"] == "Unknown"
    assert gold_provided is False


def test_v021_individual_test_lab_runs_and_saves_graph(tmp_path):
    good = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(id="s1", text="Bob is kind.", depends_on=["p1", "p2"]),
        ],
        final_answer="True",
        answer_depends_on=["s1"],
    )
    result = run_custom_test({
        "context": "Bob is quiet.\nAll quiet people are kind.",
        "question": "Bob is kind.",
        "gold_answer": "True",
        "model": "gpt-5.6",
        "use_llm_formalizer": False,
        "use_premise_grounder": False,
        "max_repair_iterations": 0,
        "prefer_z3": False,
    }, output_root=tmp_path, client=_HybridSequenceClient([good]))
    assert result["summary"]["final_answer"] == "True"
    assert result["summary"]["final_pass"] is True
    assert result["final_universal_graph"]["nodes"]
    run_dir = tmp_path / result["test_lab"]["run_id"]
    assert (run_dir / "result.json").exists()


def test_v021_fault_experiment_uses_corrected_metrics_and_no_duplicate_signatures(tmp_path):
    source = ROOT / "outputs" / "hybrid_runs" / "proofwriter_600_v019_reverified"
    if not source.exists():
        pytest.skip("large retained ProofWriter run fixture is not bundled in the slim distribution")
    result = run_fault_injection_experiment(
        source, tmp_path, sample_count=2, seed=2026,
        fault_types=["parent_deletion", "predicate_swap", "step_deletion", "answer_flip"],
        difficulties=["local", "upstream"], prefer_z3=False,
    )
    summary = result["summary"]
    assert summary["strict_empty_parent_policy"] is True
    assert summary["mutation_validity_gate"] is True
    assert summary["duplicate_mutations_removed"] is True
    assert "affected_node_f1" in summary
    assert summary["structural_schema_rejection_rate"] == 100.0
    rows = list(csv.DictReader((Path(result["output_dir"]) / "fault_mutations.csv").open(encoding="utf-8-sig")))
    signatures = {
        (r["record_id"], r["fault_type"], r.get("mutated_node_id"), r.get("mutated_text"), r.get("mutated_depends_on"), r.get("deleted_node_id"), r.get("mutated_answer"))
        for r in rows
    }
    assert len(signatures) == len(rows)


def test_v021_app_exposes_test_lab_and_case_browser_routes():
    paths = {route.path for route in app.routes}
    assert "/test-lab" in paths
    assert "/case-browser" in paths
    assert "/api/test-lab/run" in paths
    assert "/api/case-browser/{run_id}/case/{case_index}" in paths

from vrg.scientific_text import preview_record_items, preview_scientific_text
from vrg.hybrid_formalizer import formalize_proofwriter_record
from vrg.test_lab import run_custom_test


def test_v023_scientific_preview_normalizes_labelled_entity_and_relative_rule():
    preview = preview_record_items(
        [
            "Treatment A reduces inflammation.",
            "All treatments that reduce inflammation reduce fibrosis progression.",
        ],
        "Treatment A reduces fibrosis progression.",
        mode="general_science",
    )
    assert preview["summary"]["safe_for_deterministic_verification"] is True
    assert preview["summary"]["derived_premise_count"] == 1
    rows = {row["id"]: row for row in preview["items"]}
    assert rows["p1"]["normalized_text"] == "Treatment_A reduces inflammation."
    assert rows["p1"]["formal"] == "reduce(treatment_a, inflammation)"
    assert rows["p2"]["formula_type"] == "rule"
    assert "treatment(?x1)" in rows["p2"]["formal"]
    assert rows["p3"]["derived"] is True
    assert rows["p3"]["formal"] == "treatment(treatment_a)"
    assert rows["question"]["formal"] == "reduce(treatment_a, fibrosis_progression)"


def test_v023_modal_scientific_language_is_not_silently_accepted():
    preview = preview_scientific_text(
        "Coffee consumption may reduce cardiovascular mortality.",
        kind="premise",
        mode="general_science",
    )
    assert preview.needs_llm_fallback is True
    assert "epistemic_or_modal_relation_requires_llm_fallback" in preview.blocking_warnings


def test_v023_scientific_test_lab_example_passes_without_parser_induced_repair(tmp_path):
    good = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(
                id="s1",
                text="Treatment_A reduces fibrosis_progression.",
                depends_on=["p1", "p2", "p3"],
            )
        ],
        final_answer="True",
        answer_depends_on=["s1"],
    )
    client = _HybridSequenceClient([good])
    result = run_custom_test(
        {
            "context": (
                "Treatment A reduces inflammation.\n"
                "All treatments that reduce inflammation reduce fibrosis progression."
            ),
            "question": "Treatment A reduces fibrosis progression.",
            "gold_answer": "True",
            "input_mode": "general_science",
            "model": "gpt-5.6",
            "use_llm_formalizer": True,
            "use_premise_grounder": False,
            "max_repair_iterations": 1,
            "prefer_z3": False,
        },
        output_root=tmp_path,
        client=client,
    )
    assert result["schema_version"] == "0.23.0"
    assert result["summary"]["context_label"] == "True"
    assert result["summary"]["initial_pass"] is True
    assert result["summary"]["final_pass"] is True
    assert result["summary"]["repair_count"] == 0
    assert result["summary"]["api_call_count"] == 1
    assert result["summary"]["formalization_fallback_used"] is False
    assert result["settings"]["generation_used_formalized_context"] is True
    generation_prompt = client.responses.calls[0]["input"][1]["content"]
    assert "p3: Treatment_A is a treatment." in generation_prompt
    assert "Treatment_A reduces fibrosis progression." in generation_prompt

from vrg.symbol_alignment import align_item_texts


def test_v023_modifier_head_universals_share_global_vocabulary():
    preview = preview_record_items(
        [
            "Study_S is observational.",
            "All observational studies are causality_limited.",
            "All causality_limited studies are not causation_establishing.",
        ],
        "Study_S is causation_establishing.",
        mode="general_science",
    )
    rows = {row["id"]: row for row in preview["items"]}
    assert preview["summary"]["safe_for_deterministic_verification"] is True
    assert preview["summary"]["fallback_needed_count"] == 0
    assert preview["summary"]["derived_premise_count"] == 1
    assert preview["summary"]["query_predicate_connected"] is True
    assert rows["p2"]["normalized_text"] == "If something is a study and it is observational, then it is causality_limited."
    assert rows["p3"]["normalized_text"] == "If something is a study and it is causality_limited, then it is not causation_establishing."
    assert rows["p4"]["formal"] == "study(study_s)"
    assert preview["connectivity"]["orphan_antecedents"] == []


def test_v023_global_alignment_repairs_formalizer_predicate_drift():
    aligned = align_item_texts([
        {"id": "p1", "kind": "premise", "text": "Study_S is observational."},
        {"id": "p2", "kind": "premise", "text": "If something is an observational_study, then it is causality_limited."},
        {"id": "p3", "kind": "premise", "text": "If something is a causality_limited_study, then it is not causation_establishing."},
        {"id": "p4", "kind": "premise", "text": "Study_S is a study."},
        {"id": "query_statement", "kind": "query_statement", "text": "Study_S is causation_establishing."},
    ])
    by_id = {x["id"]: x["text"] for x in aligned["items"]}
    assert "observational_study" not in by_id["p2"]
    assert "causality_limited_study" not in by_id["p3"]
    assert len(aligned["alignment_decisions"]) == 2
    assert aligned["diagnostics"]["blocking_symbol_drift"] is False
    assert aligned["diagnostics"]["query_predicate_connected"] is True


def test_v023_observational_discussion_example_passes_without_fallback(tmp_path):
    good = GeneratedProofWriterOutput(
        reasoning_steps=[
            GeneratedReasoningStep(
                id="s1",
                text="Study_S is causality_limited.",
                depends_on=["p1", "p2", "p4"],
            ),
            GeneratedReasoningStep(
                id="s2",
                text="Study_S is not causation_establishing.",
                depends_on=["s1", "p3", "p4"],
            ),
        ],
        final_answer="False",
        answer_depends_on=["s2"],
    )
    client = _HybridSequenceClient([good])
    result = run_custom_test(
        {
            "context": (
                "Study_S is observational.\n"
                "All observational studies are causality_limited.\n"
                "All causality_limited studies are not causation_establishing."
            ),
            "question": "Study_S is causation_establishing.",
            "gold_answer": "False",
            "input_mode": "general_science",
            "model": "gpt-5.6",
            "use_llm_formalizer": True,
            "use_premise_grounder": False,
            "max_repair_iterations": 0,
            "prefer_z3": False,
        },
        output_root=tmp_path,
        client=client,
    )
    assert result["schema_version"] == "0.23.0"
    assert result["summary"]["context_label"] == "False"
    assert result["summary"]["initial_pass"] is True
    assert result["summary"]["final_pass"] is True
    assert result["summary"]["formalization_fallback_used"] is False
    assert result["summary"]["api_call_count"] == 1
    preflight = result["formalization_preflight"]
    assert preflight["summary"]["connectivity"]["query_predicate_connected"] is True
    prompt = client.responses.calls[0]["input"][1]["content"]
    assert "p4: Study_S is a study." in prompt
    assert "If something is a study and it is observational" in prompt

from vrg.discussion_graph import (
    DiscussionGraphOutput,
    DiscussionNode,
    DiscussionEdge,
    DiscussionIssue,
    analyze_structured_discussion,
    generate_discussion_graph,
)


def _discussion_fixture():
    return DiscussionGraphOutput(
        paragraph_summary="Inflammation is presented as unnecessary but also as the exclusive mechanism.",
        overall_assessment="clear_conflict",
        nodes=[
            DiscussionNode(
                id="claim-a", sentence_index=1,
                source_text="Treatment G reduced inflammatory activity and was followed by slower fibrosis progression.",
                plain_meaning="Inflammation decreased before fibrosis progression slowed.",
                role="observation", assertion_type="temporal", polarity="positive", certainty="observed",
                subject="Treatment G", predicate="precedes", object="slower fibrosis progression",
                why_it_matters="This is temporal evidence, not proof of mediation.",
            ),
            DiscussionNode(
                id="claim-b", sentence_index=2,
                source_text="The antifibrotic effect was observed equally with and without an inflammatory response.",
                plain_meaning="Inflammation reduction was not necessary for benefit.",
                role="evidence", assertion_type="necessity", polarity="negative", certainty="observed",
                subject="inflammation reduction", predicate="necessary for", object="antifibrotic benefit",
                why_it_matters="This challenges a necessary or exclusive inflammatory mechanism.",
            ),
            DiscussionNode(
                id="claim-c", sentence_index=3,
                source_text="The results prove that Treatment G prevents fibrosis exclusively through suppression of inflammation.",
                plain_meaning="Inflammation suppression is claimed as the exclusive mechanism.",
                role="conclusion", assertion_type="exclusivity", polarity="positive", certainty="proves",
                subject="Treatment G", predicate="exclusive mechanism", object="inflammation suppression",
                why_it_matters="This conclusion is stronger than the preceding evidence permits.",
            ),
        ],
        edges=[
            DiscussionEdge(id="edge-a", source="claim-a", target="claim-c", relation="supports", rationale="Temporal order is used as support."),
            DiscussionEdge(id="edge-b", source="claim-b", target="claim-c", relation="contradicts", rationale="Benefit without inflammatory response conflicts with exclusivity."),
        ],
        issues=[
            DiscussionIssue(
                id="issue-a", issue_type="necessity_violation", severity="high",
                title="Benefit without the proposed mediator",
                node_ids=["claim-b", "claim-c"],
                explanation="The paragraph says inflammation reduction is unnecessary, then calls it exclusive.",
                logical_pattern="not necessary for benefit + exclusive mechanism",
                suggested_revision="Describe inflammation suppression as one possible pathway rather than the exclusive mechanism.",
            )
        ],
    )


def test_v025_discussion_graph_normalizes_ids_and_marks_issue_nodes():
    result = analyze_structured_discussion({"structured_output": _discussion_fixture().model_dump()})
    assert result["schema_version"] == "0.27.0"
    assert [n["id"] for n in result["nodes"]] == ["d1", "d2", "d3"]
    assert result["edges"][1]["source"] == "d2"
    assert result["edges"][1]["target"] == "d3"
    assert result["issues"][0]["node_ids"] == ["d2", "d3"]
    assert result["nodes"][1]["has_issue"] is True
    assert result["nodes"][0]["has_issue"] is False
    assert result["summary"]["high_severity_count"] == 1
    assert result["issues"][0]["validation_status"] == "formal_conflict"
    assert result["summary"]["formal_conflict_count"] == 1


def test_v025_discussion_generation_uses_one_structured_api_call():
    client = _HybridSequenceClient([_discussion_fixture()])
    result = generate_discussion_graph(
        "Treatment G reduced inflammation but the benefit also occurred without inflammation reduction.",
        model="gpt-5.6", reasoning_effort="low", client=client,
        architecture_mode="legacy_single_pass",
    )
    assert len(client.responses.calls) == 1
    assert client.responses.calls[0]["text_format"] is DiscussionGraphOutput
    assert result["overall_assessment"] == "formal_conflict"
    assert result["summary"]["issue_count"] == 1
    assert result["usage"]["total_tokens"] == 140


def test_v025_discussion_and_readable_detail_pages_exist():
    root = ROOT
    discussion = (root / "static" / "discussion_lab.html").read_text(encoding="utf-8")
    test_lab = (root / "static" / "test_lab.html").read_text(encoding="utf-8")
    browser = (root / "static" / "case_browser.html").read_text(encoding="utf-8")
    assert "Discussion Reasoning Lab · v028" in discussion
    assert "고급 정보 · 원시 JSON" in discussion
    assert "Node / Edge 설명" in test_lab
    assert "Node / Edge 설명" in browser
