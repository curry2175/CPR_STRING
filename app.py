from __future__ import annotations

import csv
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from vrg.batch import BatchOptions, evaluate_cases, parse_jsonl, rows_to_csv, errors_to_csv
from vrg.benchmark import BenchmarkOptions, evaluate_incremental_scenarios, load_builtin_scenarios
from vrg.verifier import verify_case, verify_case_incremental
from vrg.preflight import preflight_case
from vrg.mutation import mutation_test_case
from vrg.audit import build_audit_package
from vrg.ingest import build_case_from_raw
from vrg.dataset import DatasetOptions, parse_dataset_jsonl, adapt_records, evaluate_dataset_records, save_dataset_outputs
from vrg.audit_verify import verify_audit_package_bytes
from vrg.atomize import atomize_case
from vrg.gold_eval import GoldEvalOptions, parse_gold_jsonl, evaluate_gold_records, rows_to_csv as gold_rows_to_csv
from vrg.proofwriter import analyze_proofwriter
from vrg.openai_runner import generate_and_analyze_proofwriter, openai_status
from vrg.hybrid_runner import run_hybrid_proofwriter, run_hybrid_batch
from vrg.operational_batch import OperationalBatchManager
from vrg.dataset_download import download_proofwriter_600, proofwriter_download_status
from vrg.reverify_run import reverify_run_directory
from vrg.experiment import (
    run_fault_injection_experiment, run_natural_repair_experiment,
    list_experiments, archive_experiment, FAULT_TYPES, DIFFICULTIES,
)
from vrg.test_lab import preview_custom_input, run_custom_test, list_custom_tests, load_custom_test
from vrg.discussion_graph import (
    discussion_sample, run_discussion_lab, list_discussion_runs, load_discussion_run,
)
from vrg.evaluation_suite import run_three_dataset_evaluation, list_evaluation_runs
from vrg.comparative_evaluation import run_comparative_evaluation, list_comparative_runs
from vrg.lightweight_uplift import run_lightweight_uplift, list_lightweight_uplift_runs, ALL_VARIANTS


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DOWNLOADED_DATA_DIR = DATA_DIR / "downloaded"
DOWNLOADED_DATA_DIR.mkdir(parents=True, exist_ok=True)
OPERATIONAL_BATCH = OperationalBatchManager(OUTPUT_DIR)
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
TEST_LAB_DIR = OUTPUT_DIR / "test_lab"
TEST_LAB_DIR.mkdir(parents=True, exist_ok=True)
DISCUSSION_LAB_DIR = OUTPUT_DIR / "discussion_lab"
DISCUSSION_LAB_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_SUITE_DIR = OUTPUT_DIR / "evaluation_suite"
EVALUATION_SUITE_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DATA_DIR = DOWNLOADED_DATA_DIR / "evaluation_suite"
EVALUATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
COMPARATIVE_EVALUATION_DIR = OUTPUT_DIR / "comparative_evaluation"
COMPARATIVE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
LIGHTWEIGHT_UPLIFT_DIR = OUTPUT_DIR / "lightweight_uplift"
LIGHTWEIGHT_UPLIFT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Interactive Verified Reasoning Graph MVP", version="0.50.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/batch")
def batch_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "batch.html")


@app.get("/benchmark")
def benchmark_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "benchmark.html")


@app.get("/preflight")
def preflight_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "preflight.html")


@app.get("/mutation")
def mutation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "mutation.html")




@app.get("/ingest")
def ingest_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ingest.html")


@app.get("/dataset")
def dataset_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "dataset.html")


@app.get("/audit-verify")
def audit_verify_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "audit_verify.html")


@app.get("/quality")
def quality_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "quality.html")


@app.get("/atomize")
def atomize_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "atomize.html")


@app.get("/evaluation")
def evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "evaluation.html")


@app.get("/suite-evaluation")
def suite_evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "suite_evaluation.html")


@app.get("/comparative-evaluation")
def comparative_evaluation_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "comparative_evaluation.html")


@app.get("/lightweight-uplift")
def lightweight_uplift_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "lightweight_uplift.html")


@app.get("/proofwriter")
def proofwriter_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "proofwriter.html")


@app.get("/hybrid")
def hybrid_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "hybrid.html")


@app.get("/test-lab")
def test_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "test_lab.html")


@app.get("/discussion-lab")
def discussion_lab_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "discussion_lab.html")


@app.get("/hybrid-batch")
def hybrid_batch_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "hybrid_batch.html")


@app.get("/case-browser")
def case_browser_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "case_browser.html")


@app.get("/experiment")
def experiment_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "experiment.html")


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(DATA_DIR.glob("sample_*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result[path.stem] = json.load(handle)
    return result


@app.post("/api/verify")
def verify(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        case = payload.get("case", payload)
        prefer_z3 = bool(payload.get("prefer_z3", True)) if "case" in payload else True
        compute_counterfactuals = bool(payload.get("compute_counterfactuals", True)) if "case" in payload else True
        result = verify_case(
            case,
            prefer_z3=prefer_z3,
            compute_counterfactuals=compute_counterfactuals,
        )
        output_path = OUTPUT_DIR / "latest_result.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # visible error for iterative local debugging
        raise HTTPException(status_code=500, detail=f"Verifier error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/incremental-verify")
def incremental_verify(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        previous_case = payload.get("previous_case")
        updated_case = payload.get("updated_case")
        previous_result = payload.get("previous_result")
        if not isinstance(previous_case, dict) or not isinstance(updated_case, dict) or not isinstance(previous_result, dict):
            raise ValueError("previous_case, updated_case, and previous_result must be JSON objects")
        result = verify_case_incremental(
            previous_case,
            updated_case,
            previous_result,
            prefer_z3=bool(payload.get("prefer_z3", True)),
            validate_against_full=bool(payload.get("validate_against_full", True)),
        )
        output_path = OUTPUT_DIR / "latest_result.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Incremental verifier error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/latest-result")
def latest_result() -> FileResponse:
    path = OUTPUT_DIR / "latest_result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No verification has been run yet")
    return FileResponse(path, media_type="application/json", filename="latest_result.json")


@app.get("/api/batch-sample")
def batch_sample() -> PlainTextResponse:
    path = DATA_DIR / "batch_sample.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Built-in batch sample is missing")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-ndjson")


@app.post("/api/batch-verify")
def batch_verify(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        if isinstance(payload.get("cases"), list):
            cases = payload["cases"]
        else:
            cases = parse_jsonl(str(payload.get("jsonl") or ""))
        options = BatchOptions(
            prefer_z3=bool(payload.get("prefer_z3", True)),
            compute_counterfactuals=bool(payload.get("compute_counterfactuals", False)),
        )
        result = evaluate_cases(cases, options)
        (OUTPUT_DIR / "batch_latest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (OUTPUT_DIR / "batch_summary.json").write_text(
            json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (OUTPUT_DIR / "batch_cases.csv").write_text(rows_to_csv(result["cases"]), encoding="utf-8-sig")
        (OUTPUT_DIR / "batch_errors.csv").write_text(errors_to_csv(result["errors"]), encoding="utf-8-sig")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch verifier error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/batch-latest")
def batch_latest() -> FileResponse:
    path = OUTPUT_DIR / "batch_latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No batch verification has been run yet")
    return FileResponse(path, media_type="application/json", filename="batch_latest.json")


@app.get("/api/batch-summary")
def batch_summary() -> FileResponse:
    path = OUTPUT_DIR / "batch_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No batch verification has been run yet")
    return FileResponse(path, media_type="application/json", filename="batch_summary.json")


@app.get("/api/batch-cases-csv")
def batch_cases_csv() -> FileResponse:
    path = OUTPUT_DIR / "batch_cases.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No batch verification has been run yet")
    return FileResponse(path, media_type="text/csv", filename="batch_cases.csv")


@app.get("/api/batch-errors-csv")
def batch_errors_csv() -> FileResponse:
    path = OUTPUT_DIR / "batch_errors.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No batch verification has been run yet")
    return FileResponse(path, media_type="text/csv", filename="batch_errors.csv")


@app.post("/api/incremental-benchmark")
def incremental_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        scenarios = payload.get("scenarios")
        if not isinstance(scenarios, list):
            scenarios = load_builtin_scenarios(DATA_DIR)
        result = evaluate_incremental_scenarios(
            scenarios,
            BenchmarkOptions(
                prefer_z3=bool(payload.get("prefer_z3", True)),
                repetitions=int(payload.get("repetitions", 5)),
            ),
        )
        (OUTPUT_DIR / "incremental_benchmark.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Incremental benchmark error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/incremental-benchmark-latest")
def incremental_benchmark_latest() -> FileResponse:
    path = OUTPUT_DIR / "incremental_benchmark.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No incremental benchmark has been run yet")
    return FileResponse(path, media_type="application/json", filename="incremental_benchmark.json")


@app.post("/api/preflight")
def preflight(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        case = payload.get("case", payload)
        result = preflight_case(case)
        (OUTPUT_DIR / "preflight_latest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preflight error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/mutation-test")
def mutation_test(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        case = payload.get("case", payload)
        result = mutation_test_case(
            case,
            prefer_z3=bool(payload.get("prefer_z3", True)) if "case" in payload else True,
            max_nodes=int(payload.get("max_nodes", 20)) if "case" in payload else 20,
        )
        (OUTPUT_DIR / "mutation_latest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mutation test error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/audit-package")
def audit_package(payload: dict[str, Any]) -> FileResponse:
    try:
        case = payload.get("case")
        result = payload.get("result")
        if not isinstance(case, dict):
            raise ValueError("case must be a JSON object")
        if not isinstance(result, dict):
            result = verify_case(
                case,
                prefer_z3=bool(payload.get("prefer_z3", True)),
                compute_counterfactuals=bool(payload.get("compute_counterfactuals", False)),
            )
        path = build_audit_package(case, result, OUTPUT_DIR)
        return FileResponse(path, media_type="application/zip", filename=path.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit package error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/raw-ingest-sample")
def raw_ingest_sample() -> dict[str, Any]:
    path = DATA_DIR / "raw_ingest_sample.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Built-in raw ingest sample is missing")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/ingest")
def ingest_raw(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = build_case_from_raw(payload)
        (OUTPUT_DIR / "ingested_case.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/dataset-sample")
def dataset_sample() -> PlainTextResponse:
    path = DATA_DIR / "dataset_sample.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Built-in dataset sample is missing")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-ndjson")


@app.post("/api/dataset-adapt")
def dataset_adapt(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        records = payload.get("records")
        if not isinstance(records, list):
            records = parse_dataset_jsonl(str(payload.get("jsonl") or ""))
        result = adapt_records(records)
        (OUTPUT_DIR / "dataset_adaptation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset adaptation error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/dataset-evaluate")
def dataset_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        records = payload.get("records")
        if not isinstance(records, list):
            records = parse_dataset_jsonl(str(payload.get("jsonl") or ""))
        result = evaluate_dataset_records(
            records,
            DatasetOptions(
                prefer_z3=bool(payload.get("prefer_z3", True)),
                compute_counterfactuals=bool(payload.get("compute_counterfactuals", False)),
            ),
        )
        save_dataset_outputs(result, OUTPUT_DIR)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Dataset evaluation error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/dataset-evaluation-latest")
def dataset_evaluation_latest() -> FileResponse:
    path = OUTPUT_DIR / "dataset_evaluation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No dataset evaluation has been run yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/dataset-cases-csv")
def dataset_cases_csv() -> FileResponse:
    path = OUTPUT_DIR / "dataset_cases.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No dataset evaluation has been run yet")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/dataset-report")
def dataset_report() -> FileResponse:
    path = OUTPUT_DIR / "dataset_report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No dataset evaluation has been run yet")
    return FileResponse(path, media_type="text/html", filename=path.name)


@app.get("/api/adapted-cases-jsonl")
def adapted_cases_jsonl() -> FileResponse:
    path = OUTPUT_DIR / "adapted_cases.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No dataset evaluation has been run yet")
    return FileResponse(path, media_type="application/x-ndjson", filename=path.name)


@app.post("/api/audit-verify")
async def audit_verify(request: Request, rerun_smt: bool = True) -> dict[str, Any]:
    try:
        payload = await request.body()
        result = verify_audit_package_bytes(payload, rerun_smt=rerun_smt)
        (OUTPUT_DIR / "audit_verification.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Audit verification error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/atomize")
def atomize(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        case = payload.get("case", payload)
        result = atomize_case(case, apply=bool(payload.get("apply", True)) if "case" in payload else True)
        (OUTPUT_DIR / "atomized_case.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Atomization error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/gold-benchmark-sample")
def gold_benchmark_sample() -> PlainTextResponse:
    path = DATA_DIR / "gold_reasoning_benchmark.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Built-in gold benchmark is missing")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="application/x-ndjson")


@app.post("/api/gold-evaluate")
def gold_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        records = payload.get("records")
        if not isinstance(records, list):
            records = parse_gold_jsonl(str(payload.get("jsonl") or ""))
        result = evaluate_gold_records(records, GoldEvalOptions(
            prefer_z3=bool(payload.get("prefer_z3", True)),
            compute_counterfactuals=bool(payload.get("compute_counterfactuals", False)),
        ))
        (OUTPUT_DIR / "gold_evaluation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (OUTPUT_DIR / "gold_cases.csv").write_text(gold_rows_to_csv(result["cases"]), encoding="utf-8-sig")
        (OUTPUT_DIR / "gold_nodes.csv").write_text(gold_rows_to_csv(result["nodes"]), encoding="utf-8-sig")
        (OUTPUT_DIR / "review_queue.csv").write_text(gold_rows_to_csv(result["review_queue"]), encoding="utf-8-sig")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gold evaluation error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/gold-evaluation-latest")
def gold_evaluation_latest() -> FileResponse:
    path = OUTPUT_DIR / "gold_evaluation.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No gold evaluation has been run yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/gold-cases-csv")
def gold_cases_csv() -> FileResponse:
    path = OUTPUT_DIR / "gold_cases.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No gold evaluation has been run yet")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/gold-nodes-csv")
def gold_nodes_csv() -> FileResponse:
    path = OUTPUT_DIR / "gold_nodes.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No gold evaluation has been run yet")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.get("/api/review-queue-csv")
def review_queue_csv() -> FileResponse:
    path = OUTPUT_DIR / "review_queue.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No gold evaluation has been run yet")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@app.post("/api/evaluation-suite/run")
def evaluation_suite_run(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        datasets = payload.get("datasets") or ["proofwriter", "legalbench", "pubmedqa"]
        legal_tasks = payload.get("legal_tasks") or []
        if isinstance(legal_tasks, str):
            legal_tasks = [x.strip() for x in legal_tasks.split(",") if x.strip()]
        return run_three_dataset_evaluation(
            output_root=EVALUATION_SUITE_DIR,
            data_root=EVALUATION_DATA_DIR,
            datasets=[str(x) for x in datasets],
            limit_per_dataset=max(0, int(payload.get("limit_per_dataset", 5))),
            legal_tasks=[str(x) for x in legal_tasks],
            model=str(payload.get("model") or "gpt-5.6"),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            max_output_tokens=int(payload.get("max_output_tokens") or 3500),
            max_repair_iterations=max(0, min(3, int(payload.get("max_repair_iterations", 0)))),
            refresh_datasets=bool(payload.get("refresh_datasets", False)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation suite error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/evaluation-suite/runs")
def evaluation_suite_runs() -> dict[str, Any]:
    return {"runs": list_evaluation_runs(EVALUATION_SUITE_DIR)}


@app.get("/api/evaluation-suite/latest")
def evaluation_suite_latest() -> FileResponse:
    path = EVALUATION_SUITE_DIR / "latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No three-dataset evaluation has been run yet")
    return FileResponse(path, media_type="application/json", filename="three_dataset_evaluation_latest.json")


@app.get("/api/evaluation-suite/report/{run_id}")
def evaluation_suite_report(run_id: str) -> FileResponse:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "", run_id)
    path = EVALUATION_SUITE_DIR / safe / "report.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation report was not found")
    return FileResponse(path, media_type="text/html")


@app.get("/api/proofwriter-sample")
def proofwriter_sample() -> dict[str, Any]:
    path = DATA_DIR / "proofwriter_real_sample.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Built-in ProofWriter sample is missing")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/proofwriter-analyze")
def proofwriter_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = analyze_proofwriter(payload)
        (OUTPUT_DIR / "proofwriter_analysis.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (OUTPUT_DIR / "proofwriter_adapted_case.json").write_text(
            json.dumps(result["adapter"]["adapted_case"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (OUTPUT_DIR / "proofwriter_verified_graph.json").write_text(
            json.dumps(result["verified_graph"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ProofWriter analysis error: {type(exc).__name__}: {exc}") from exc




@app.get("/api/openai-status")
def api_openai_status() -> dict[str, Any]:
    return openai_status(ROOT)


@app.post("/api/proofwriter-gpt-analyze")
def proofwriter_gpt_analyze(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = generate_and_analyze_proofwriter(payload)
        (OUTPUT_DIR / "proofwriter_gpt_run.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if result.get("runs"):
            first = result["runs"][0]
            (OUTPUT_DIR / "proofwriter_gpt_output.json").write_text(
                json.dumps(first["generation"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            analysis = first["analysis"]
            (OUTPUT_DIR / "proofwriter_analysis.json").write_text(
                json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (OUTPUT_DIR / "proofwriter_adapted_case.json").write_text(
                json.dumps(analysis["adapter"]["adapted_case"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (OUTPUT_DIR / "proofwriter_verified_graph.json").write_text(
                json.dumps(analysis["verified_graph"], indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OpenAI automation error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/proofwriter-gpt-run-latest")
def proofwriter_gpt_run_latest() -> FileResponse:
    path = OUTPUT_DIR / "proofwriter_gpt_run.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No GPT automated ProofWriter run has been completed yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/proofwriter-analysis-latest")
def proofwriter_analysis_latest() -> FileResponse:
    path = OUTPUT_DIR / "proofwriter_analysis.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No ProofWriter analysis has been run yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/proofwriter-adapted-case")
def proofwriter_adapted_case() -> FileResponse:
    path = OUTPUT_DIR / "proofwriter_adapted_case.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No ProofWriter analysis has been run yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/proofwriter-verified-graph")
def proofwriter_verified_graph() -> FileResponse:
    path = OUTPUT_DIR / "proofwriter_verified_graph.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No ProofWriter analysis has been run yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.post("/api/hybrid-proofwriter")
def hybrid_proofwriter(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run_hybrid_proofwriter(payload)
        (OUTPUT_DIR / "hybrid_run.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        (OUTPUT_DIR / "hybrid_universal_graph.json").write_text(json.dumps(result["final_universal_graph"], indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hybrid verifier error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/hybrid-batch")
def hybrid_batch(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = run_hybrid_batch(payload)
        (OUTPUT_DIR / "hybrid_batch.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Hybrid batch error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/hybrid-run-latest")
def hybrid_run_latest() -> FileResponse:
    path = OUTPUT_DIR / "hybrid_run.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No hybrid run has been completed yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/hybrid-universal-graph")
def hybrid_universal_graph() -> FileResponse:
    path = OUTPUT_DIR / "hybrid_universal_graph.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No hybrid universal graph has been completed yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/hybrid-batch-latest")
def hybrid_batch_latest() -> FileResponse:
    path = OUTPUT_DIR / "hybrid_batch.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No hybrid batch has been completed yet")
    return FileResponse(path, media_type="application/json", filename=path.name)


@app.get("/api/proofwriter-dataset/status")
def proofwriter_dataset_status() -> dict[str, Any]:
    return proofwriter_download_status(DOWNLOADED_DATA_DIR)


@app.post("/api/proofwriter-dataset/download")
def proofwriter_dataset_download(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        payload = payload or {}
        result = download_proofwriter_600(DOWNLOADED_DATA_DIR, refresh=bool(payload.get("refresh", False)))
        return {**result.as_dict(), "available": True}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ProofWriter download error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/proofwriter-dataset/file")
def proofwriter_dataset_file() -> FileResponse:
    status = proofwriter_download_status(DOWNLOADED_DATA_DIR)
    if not status.get("available"):
        raise HTTPException(status_code=404, detail="ProofWriter dataset has not been downloaded yet")
    path = Path(str(status["dataset_path"]))
    return FileResponse(path, media_type="application/x-ndjson", filename=path.name)


@app.post("/api/hybrid-batch-job/start-proofwriter-600")
def hybrid_batch_job_start_proofwriter_600(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = download_proofwriter_600(DOWNLOADED_DATA_DIR, refresh=bool(payload.pop("refresh_dataset", False)))
        payload = dict(payload)
        payload["dataset_text"] = result.dataset_path.read_text(encoding="utf-8-sig")
        payload["dataset_source"] = {
            "dataset_id": result.source_dataset,
            "config": result.config,
            "split": result.split,
            "row_count": result.row_count,
            "cached": result.cached,
        }
        return OPERATIONAL_BATCH.start(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"ProofWriter automatic run error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/hybrid-batch-job/start")
def hybrid_batch_job_start(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return OPERATIONAL_BATCH.start(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Operational batch error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/hybrid-batch-job/upload")
async def hybrid_batch_job_upload(file: UploadFile = File(...), settings_json: str = Form("{}")) -> dict[str, Any]:
    try:
        settings = json.loads(settings_json or "{}")
        if not isinstance(settings, dict):
            raise ValueError("settings_json must be an object")
        raw = await file.read()
        settings["dataset_text"] = raw.decode("utf-8-sig")
        return OPERATIONAL_BATCH.start(settings)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Dataset file must be UTF-8 encoded") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Operational batch upload error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/hybrid-batch-jobs")
def hybrid_batch_jobs() -> dict[str, Any]:
    return {"runs": OPERATIONAL_BATCH.list_runs()}


@app.get("/api/hybrid-batch-job/{run_id}")
def hybrid_batch_job_status(run_id: str) -> dict[str, Any]:
    try:
        return OPERATIONAL_BATCH.status(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/hybrid-batch-job/{run_id}/pause")
def hybrid_batch_job_pause(run_id: str) -> dict[str, Any]:
    try:
        return OPERATIONAL_BATCH.pause(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/hybrid-batch-job/{run_id}/file/{relative_name:path}")
def hybrid_batch_job_file(run_id: str, relative_name: str) -> FileResponse:
    try:
        path = OPERATIONAL_BATCH.file_path(run_id, relative_name)
        return FileResponse(path, filename=path.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/hybrid-batch-job/{run_id}/archive")
def hybrid_batch_job_archive(run_id: str) -> FileResponse:
    try:
        path = OPERATIONAL_BATCH.archive(run_id)
        return FileResponse(path, media_type="application/zip", filename=path.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc



def _case_rows_for_run(run_id: str) -> tuple[Path, list[dict[str, Any]]]:
    run_dir = (OUTPUT_DIR / "hybrid_runs" / run_id).resolve()
    runs_root = (OUTPUT_DIR / "hybrid_runs").resolve()
    if runs_root not in run_dir.parents or not run_dir.exists():
        raise ValueError(f"Unknown run_id: {run_id}")
    rows: list[dict[str, Any]] = []
    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        index_path = run_dir / "index.jsonl"
        if index_path.exists():
            rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return run_dir, rows


@app.get("/api/case-browser/{run_id}/cases")
def case_browser_cases(run_id: str, filter: str = "all", q: str = "") -> dict[str, Any]:
    try:
        _run_dir, rows = _case_rows_for_run(run_id)
        def truth(value: Any) -> bool:
            return str(value).strip().lower() in {"1", "true", "yes"}
        selected = []
        query = q.strip().lower()
        for row in rows:
            if query and query not in str(row.get("record_id", "")).lower():
                continue
            initial_pass = truth(row.get("initial_pass"))
            final_pass = truth(row.get("final_pass"))
            correct = truth(row.get("final_answer_correct"))
            repair_count = int(float(row.get("repair_count") or 0))
            if filter == "initial_fail" and initial_pass:
                continue
            if filter == "final_fail" and final_pass:
                continue
            if filter == "repair" and repair_count <= 0:
                continue
            if filter == "wrong" and correct:
                continue
            selected.append(row)
        return {"run_id": run_id, "filter": filter, "count": len(selected), "cases": selected}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/case-browser/{run_id}/case/{case_index}")
def case_browser_case(run_id: str, case_index: int) -> dict[str, Any]:
    try:
        run_dir, rows = _case_rows_for_run(run_id)
        row = next((x for x in rows if int(float(x.get("index") or 0)) == case_index), None)
        if not row:
            raise ValueError(f"Case index not found: {case_index}")
        case_file = str(row.get("case_file") or "")
        if not case_file:
            candidates = sorted((run_dir / "cases").glob(f"{case_index:06d}_*.json"))
            if not candidates:
                raise ValueError(f"Case JSON not found: {case_index}")
            path = candidates[0]
        else:
            path = (run_dir / case_file).resolve()
        if run_dir not in path.parents or not path.exists():
            raise ValueError("Invalid or missing case file")
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/case-browser/{run_id}/reverify-v019")
def case_browser_reverify(run_id: str) -> dict[str, Any]:
    try:
        source = OPERATIONAL_BATCH._run_dir(run_id)
        dest = reverify_run_directory(source, OPERATIONAL_BATCH.runs_root)
        return {"source_run_id": run_id, "run_id": dest.name, "new_api_calls": 0, "status": "completed"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Offline re-verification error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/case-browser/import-reverify")
async def case_browser_import_reverify(file: UploadFile = File(...)) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="vrg_import_"))
    try:
        raw = await file.read()
        zip_path = temp_root / "run.zip"
        zip_path.write_bytes(raw)
        extract_root = temp_root / "extracted"
        extract_root.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            root = extract_root.resolve()
            for member in zf.infolist():
                target = (extract_root / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            zf.extractall(extract_root)
        candidates = [x for x in extract_root.iterdir() if x.is_dir() and (x / "cases").exists()]
        if not candidates and (extract_root / "cases").exists():
            candidates = [extract_root]
        if len(candidates) != 1:
            raise ValueError("ZIP must contain exactly one run directory with cases/*.json")
        dest = reverify_run_directory(candidates[0], OPERATIONAL_BATCH.runs_root)
        return {"source_filename": file.filename, "run_id": dest.name, "new_api_calls": 0, "status": "completed"}
    except (ValueError, zipfile.BadZipFile) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import/re-verification error: {type(exc).__name__}: {exc}") from exc
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _source_run_dir(run_id: str) -> Path:
    path = (OUTPUT_DIR / "hybrid_runs" / run_id).resolve()
    root = (OUTPUT_DIR / "hybrid_runs").resolve()
    if root not in path.parents or not path.exists():
        raise ValueError(f"Unknown source run: {run_id}")
    return path


@app.get("/api/test-lab/sample")
def test_lab_sample() -> dict[str, Any]:
    return {
        "context": (
            "Study_S is observational.\n"
            "All observational studies are causality_limited.\n"
            "All causality_limited studies are not causation_establishing."
        ),
        "question": "Study_S is causation_establishing.",
        "gold_answer": "False",
        "input_mode": "general_science",
    }


@app.post("/api/test-lab/preview")
def test_lab_preview(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return preview_custom_input(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/test-lab/run")
def test_lab_run(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_custom_test(payload, output_root=TEST_LAB_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test Lab error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/test-lab/runs")
def test_lab_runs() -> dict[str, Any]:
    return {"runs": list_custom_tests(TEST_LAB_DIR)}


@app.get("/api/test-lab/run/{run_id}")
def test_lab_result(run_id: str) -> dict[str, Any]:
    try:
        return load_custom_test(TEST_LAB_DIR, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc




@app.post("/api/comparative-evaluation/run")
def comparative_evaluation_run(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        datasets = payload.get("datasets") or ["proofwriter", "legalbench", "pubmedqa"]
        legal_tasks = payload.get("legal_tasks") or []
        if isinstance(legal_tasks, str):
            legal_tasks = [x.strip() for x in legal_tasks.split(",") if x.strip()]
        return run_comparative_evaluation(
            output_root=COMPARATIVE_EVALUATION_DIR,
            data_root=EVALUATION_DATA_DIR,
            discussion_benchmark=DATA_DIR / "discussion_audit_benchmark_v028.jsonl",
            datasets=list(datasets),
            limit_per_dataset=int(payload.get("limit_per_dataset") or 5),
            legal_tasks=list(legal_tasks),
            audit_cases=int(payload.get("audit_cases") or 5),
            run_answer_comparison=bool(payload.get("run_answer_comparison", True)),
            run_reasoning_audit=bool(payload.get("run_reasoning_audit", True)),
            run_discussion_audit=bool(payload.get("run_discussion_audit", True)),
            model=payload.get("model"),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            max_output_tokens=int(payload.get("max_output_tokens") or 3500),
            seed=int(payload.get("seed") or 2028),
            refresh_datasets=bool(payload.get("refresh_datasets", False)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparative evaluation error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/comparative-evaluation/runs")
def comparative_evaluation_runs() -> dict[str, Any]:
    return {"runs": list_comparative_runs(COMPARATIVE_EVALUATION_DIR)}


@app.get("/api/comparative-evaluation/latest")
def comparative_evaluation_latest() -> FileResponse:
    path = COMPARATIVE_EVALUATION_DIR / "latest_comparative.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No comparative evaluation run yet")
    return FileResponse(path)


@app.get("/api/comparative-evaluation/report/{run_id}")
def comparative_evaluation_report(run_id: str) -> FileResponse:
    path = (COMPARATIVE_EVALUATION_DIR / run_id / "report.html").resolve()
    if COMPARATIVE_EVALUATION_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Unknown comparative evaluation run")
    return FileResponse(path)


@app.post("/api/lightweight-uplift/run")
def lightweight_uplift_run(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        variants = payload.get("variants") or list(ALL_VARIANTS)
        if isinstance(variants, str):
            variants = [x.strip() for x in variants.split(",") if x.strip()]
        return run_lightweight_uplift(
            benchmark_path=DATA_DIR / "discussion_uplift_benchmark_v029.jsonl",
            output_root=LIGHTWEIGHT_UPLIFT_DIR,
            small_model=str(payload.get("small_model") or "gpt-5.4-nano"),
            reference_model=str(payload.get("reference_model") or "gpt-5.4-mini"),
            include_reference=bool(payload.get("include_reference", True)),
            variants=list(variants),
            limit=int(payload.get("limit") or 12),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            max_output_tokens=int(payload.get("max_output_tokens") or 2200),
            seed=int(payload.get("seed") or 2029),
            include_graph_structure_ablation=bool(payload.get("include_graph_structure_ablation", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lightweight uplift error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/lightweight-uplift/runs")
def lightweight_uplift_runs() -> dict[str, Any]:
    return {"runs": list_lightweight_uplift_runs(LIGHTWEIGHT_UPLIFT_DIR)}


@app.get("/api/lightweight-uplift/report/{run_id}")
def lightweight_uplift_report(run_id: str) -> FileResponse:
    path = (LIGHTWEIGHT_UPLIFT_DIR / run_id / "report.html").resolve()
    if LIGHTWEIGHT_UPLIFT_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Unknown lightweight uplift run")
    return FileResponse(path)


@app.get("/api/discussion-lab/sample")
def api_discussion_lab_sample() -> dict[str, Any]:
    return {"text": discussion_sample()}


@app.post("/api/discussion-lab/run")
def api_discussion_lab_run(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_discussion_lab(payload, output_root=DISCUSSION_LAB_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Discussion Lab error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/discussion-lab/runs")
def api_discussion_lab_runs() -> dict[str, Any]:
    return {"runs": list_discussion_runs(DISCUSSION_LAB_DIR)}


@app.get("/api/discussion-lab/run/{run_id}")
def api_discussion_lab_result(run_id: str) -> dict[str, Any]:
    try:
        return load_discussion_run(DISCUSSION_LAB_DIR, run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiment/config")
def experiment_config() -> dict[str, Any]:
    runs = []
    runs_root = OUTPUT_DIR / "hybrid_runs"
    if runs_root.exists():
        for path in sorted(runs_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if path.is_dir() and (path / "summary.json").exists():
                try:
                    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
                except Exception:
                    summary = {}
                runs.append({"run_id": path.name, "processed_records": summary.get("processed_records"), "final_pass_count": summary.get("final_pass_count")})
    return {"schema_version": "0.27.0", "source_runs": runs, "fault_types": list(FAULT_TYPES), "difficulties": list(DIFFICULTIES), "repair_modes": ["no_repair", "blind", "guided", "cascade"], "experiments": list_experiments(EXPERIMENT_DIR)}


@app.post("/api/experiment/fault-injection")
def experiment_fault_injection(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        source = _source_run_dir(str(payload.get("run_id") or "proofwriter_600_v019_reverified"))
        return run_fault_injection_experiment(
            source, EXPERIMENT_DIR,
            sample_count=int(payload.get("sample_count") or 100),
            seed=int(payload.get("seed") or 2026),
            fault_types=payload.get("fault_types") or FAULT_TYPES,
            difficulties=payload.get("difficulties") or DIFFICULTIES,
            prefer_z3=bool(payload.get("prefer_z3", True)),
            max_reasoning_steps=int(payload.get("max_reasoning_steps") or 8),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Fault-injection experiment error: {type(exc).__name__}: {exc}") from exc


@app.post("/api/experiment/natural-repair")
def experiment_natural_repair(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        source = _source_run_dir(str(payload.get("run_id") or "proofwriter_600_v019_reverified"))
        return run_natural_repair_experiment(
            source, EXPERIMENT_DIR,
            modes=payload.get("modes") or ["no_repair", "blind", "guided", "cascade"],
            max_cases=int(payload.get("max_cases") or 0),
            model=str(payload.get("model") or "gpt-5.6"),
            reasoning_effort=str(payload.get("reasoning_effort") or "low"),
            max_output_tokens=int(payload.get("max_output_tokens") or 5000),
            prefer_z3=bool(payload.get("prefer_z3", True)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Natural-repair experiment error: {type(exc).__name__}: {exc}") from exc


@app.get("/api/experiments")
def experiments_list() -> dict[str, Any]:
    return {"experiments": list_experiments(EXPERIMENT_DIR)}


@app.get("/api/experiment/{experiment_id}")
def experiment_result(experiment_id: str) -> dict[str, Any]:
    path = (EXPERIMENT_DIR / experiment_id / "summary.json").resolve()
    if EXPERIMENT_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="Unknown experiment")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/experiment/{experiment_id}/file/{relative_name:path}")
def experiment_file(experiment_id: str, relative_name: str) -> FileResponse:
    root = (EXPERIMENT_DIR / experiment_id).resolve()
    path = (root / relative_name).resolve()
    if root not in path.parents or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Experiment file not found")
    media = "text/csv" if path.suffix == ".csv" else ("application/json" if path.suffix in {".json", ".jsonl"} else "text/plain")
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/experiment/{experiment_id}/archive")
def experiment_archive(experiment_id: str) -> FileResponse:
    try:
        path = archive_experiment(EXPERIMENT_DIR, experiment_id)
        return FileResponse(path, media_type="application/zip", filename=path.name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
