from __future__ import annotations

import csv
import html
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .batch import BatchOptions, evaluate_cases
from .ingest import build_case_from_raw, normalize_yes_no, split_llm_response
from .preflight import preflight_case


@dataclass
class DatasetOptions:
    prefer_z3: bool = True
    compute_counterfactuals: bool = False


def parse_dataset_jsonl(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(str(text or "").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {line_no}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_no}: each record must be a JSON object")
        records.append(value)
    if not records:
        raise ValueError("No JSONL records were found")
    return records


def _flatten_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return list(value.values())
    if isinstance(value, list):
        return value
    return [value]


def _sentences(text: str) -> list[str]:
    return [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if piece.strip()]


def _premises_from_record(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    if isinstance(record.get("premises"), list):
        source = record["premises"]
        adapter = "native"
    elif isinstance(record.get("context"), list):
        source = record["context"]
        adapter = "context_list"
    elif isinstance(record.get("theory"), str):
        source = _sentences(record["theory"])
        adapter = "theory_text"
    elif record.get("triples") is not None or record.get("rules") is not None:
        source = _flatten_items(record.get("triples")) + _flatten_items(record.get("rules"))
        adapter = "proofwriter_like"
    elif isinstance(record.get("context"), str):
        source = _sentences(record["context"])
        adapter = "context_text"
    else:
        raise ValueError("Could not find premises/context/theory/triples+rules")

    premises: list[dict[str, str]] = []
    for index, item in enumerate(source, 1):
        if isinstance(item, str):
            text = item
            node_id = f"p{index}"
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("statement") or item.get("value") or "")
            node_id = str(item.get("id") or f"p{index}")
        else:
            text = str(item)
            node_id = f"p{index}"
        if text.strip():
            premises.append({"id": node_id, "text": text.strip()})
    if not premises:
        raise ValueError("No non-empty premises were found")
    return premises, adapter


def _question(record: dict[str, Any]) -> str:
    value = record.get("question", record.get("query", record.get("hypothesis")))
    text = str(value or "").strip()
    if not text:
        raise ValueError("Could not find question/query/hypothesis")
    return text


def _gold(record: dict[str, Any]) -> str:
    for key in ("gold_answer", "gold", "label", "answer"):
        if key in record and not isinstance(record.get(key), dict):
            return normalize_yes_no(record.get(key), "gold answer")
    raise ValueError("Could not find a strict Yes/No gold label")


def _native_llm_output(record: dict[str, Any]) -> dict[str, Any] | None:
    output = record.get("llm_output")
    if isinstance(output, dict):
        answer = normalize_yes_no(output.get("answer"), "predicted answer")
        raw_steps = output.get("reasoning_steps") or []
        if not isinstance(raw_steps, list):
            raise ValueError("llm_output.reasoning_steps must be a list")
        steps: list[dict[str, str]] = []
        for index, item in enumerate(raw_steps, 1):
            if isinstance(item, str):
                steps.append({"id": f"s{index}", "text": item})
            elif isinstance(item, dict):
                step = {"id": str(item.get("id") or f"s{index}"), "text": str(item.get("text") or "")}
                if item.get("depends_on") is not None:
                    step["depends_on"] = item.get("depends_on")
                steps.append(step)
            else:
                raise ValueError(f"Unsupported reasoning step {index}")
        result = {"reasoning_steps": steps, "answer": answer}
        if output.get("answer_depends_on") is not None:
            result["answer_depends_on"] = output.get("answer_depends_on")
        return result
    return None


def _llm_output(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    native = _native_llm_output(record)
    if native is not None:
        return native, {"strategy": "native_llm_output", "step_count": len(native["reasoning_steps"]), "warnings": []}

    if isinstance(record.get("reasoning_steps"), list):
        answer_value = record.get("predicted_answer", record.get("prediction"))
        answer = normalize_yes_no(answer_value, "predicted answer")
        steps = []
        for index, item in enumerate(record["reasoning_steps"], 1):
            if isinstance(item, str):
                steps.append({"id": f"s{index}", "text": item})
            else:
                step = {"id": str(item.get("id") or f"s{index}"), "text": str(item.get("text") or "")}
                if item.get("depends_on") is not None:
                    step["depends_on"] = item.get("depends_on")
                steps.append(step)
        output = {"reasoning_steps": steps, "answer": answer}
        if record.get("answer_depends_on") is not None:
            output["answer_depends_on"] = record.get("answer_depends_on")
        return output, {"strategy": "separate_reasoning_steps", "step_count": len(steps), "warnings": []}

    response = None
    response_key = None
    for key in ("llm_response", "model_output", "response", "completion", "generated_text"):
        if record.get(key) is not None:
            response = record.get(key)
            response_key = key
            break
    if response is None:
        raise ValueError("Could not find llm_output, reasoning_steps, or raw LLM response text")
    explicit = record.get("predicted_answer", record.get("prediction"))
    extracted = split_llm_response(str(response), explicit_answer=explicit)
    metadata = dict(extracted["extraction"])
    metadata["response_field"] = response_key
    return {"reasoning_steps": extracted["reasoning_steps"], "answer": extracted["answer"]}, metadata


def adapt_record(record: dict[str, Any], index: int = 1) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Dataset record must be a JSON object")
    premises, adapter = _premises_from_record(record)
    output, extraction = _llm_output(record)
    case_id = str(record.get("id") or record.get("case_id") or record.get("qid") or f"dataset_case_{index}")
    case = {
        "id": case_id,
        "premises": premises,
        "question": _question(record),
        "llm_output": output,
        "gold_answer": _gold(record),
    }
    if isinstance(record.get("semantic_relations"), list):
        case["semantic_relations"] = record["semantic_relations"]
    preflight = preflight_case(case)
    return {
        "case": case,
        "adapter": adapter,
        "extraction": extraction,
        "preflight": preflight,
    }


def adapt_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    adapted: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        record_id = str(record.get("id") or record.get("case_id") or record.get("qid") or f"line_{index}")
        try:
            item = adapt_record(record, index)
            adapted.append(item["case"])
            pre = item["preflight"]
            rows.append({
                "record_index": index,
                "case_id": item["case"]["id"],
                "adapter": item["adapter"],
                "extraction_strategy": item["extraction"].get("strategy"),
                "reasoning_step_count": len(item["case"]["llm_output"]["reasoning_steps"]),
                "parser_coverage_percent": pre["summary"]["parser_coverage_percent"],
                "ready_for_verification": pre["ready_for_verification"],
                "preflight_errors": len(pre["errors"]),
                "preflight_warnings": len(pre["warnings"]),
            })
            diagnostics.append({
                "case_id": item["case"]["id"],
                "adapter": item["adapter"],
                "extraction": item["extraction"],
                "preflight": pre,
            })
        except Exception as exc:
            errors.append({
                "record_index": index,
                "case_id": record_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    adapter_counts = Counter(row["adapter"] for row in rows)
    extraction_counts = Counter(row["extraction_strategy"] for row in rows)
    parser_error_counts: Counter[str] = Counter()
    extraction_warning_count = 0
    for diagnostic in diagnostics:
        extraction_warning_count += len(diagnostic.get("extraction", {}).get("warnings") or [])
        for statement in diagnostic.get("preflight", {}).get("statements", []):
            if statement.get("parse_status") == "untranslatable":
                parser_error_counts[str(statement.get("error") or "unknown")] += 1
        question = diagnostic.get("preflight", {}).get("question", {})
        if question.get("parse_status") == "untranslatable":
            parser_error_counts[str(question.get("error") or "unsupported question")] += 1
    ready_count = sum(bool(row["ready_for_verification"]) for row in rows)
    avg_coverage = round(sum(float(row["parser_coverage_percent"]) for row in rows) / len(rows), 2) if rows else 0.0
    return {
        "schema_version": "0.15.0",
        "summary": {
            "total_records": len(records),
            "adapted_records": len(adapted),
            "failed_records": len(errors),
            "ready_for_verification_count": ready_count,
            "average_parser_coverage_percent": avg_coverage,
            "adapter_distribution": dict(sorted(adapter_counts.items())),
            "extraction_strategy_distribution": dict(sorted(extraction_counts.items())),
            "parser_error_distribution": dict(sorted(parser_error_counts.items())),
            "extraction_warning_count": extraction_warning_count,
        },
        "cases": adapted,
        "rows": rows,
        "diagnostics": diagnostics,
        "errors": errors,
    }


def evaluate_dataset_records(records: list[dict[str, Any]], options: DatasetOptions) -> dict[str, Any]:
    adaptation = adapt_records(records)
    batch = evaluate_cases(
        adaptation["cases"],
        BatchOptions(
            prefer_z3=options.prefer_z3,
            compute_counterfactuals=options.compute_counterfactuals,
        ),
    ) if adaptation["cases"] else {
        "schema_version": "0.15.0", "summary": {"total_cases": 0, "completed_cases": 0, "failed_cases": 0}, "cases": [], "errors": []
    }
    verification_rows = {row["case_id"]: row for row in batch.get("cases", [])}
    reasoning_proof_statuses: Counter[str] = Counter()
    reasoning_chain_statuses: Counter[str] = Counter()
    for full_result in batch.get("results", []):
        for node in full_result.get("nodes", []):
            if node.get("kind") == "reasoning":
                reasoning_proof_statuses[str(node.get("proof_status") or "unknown")] += 1
                reasoning_chain_statuses[str(node.get("chain_status") or "unknown")] += 1
    combined_rows = []
    for row in adaptation["rows"]:
        combined = dict(row)
        combined.update(verification_rows.get(row["case_id"], {}))
        combined_rows.append(combined)
    return {
        "schema_version": "0.15.0",
        "adaptation": adaptation,
        "verification": batch,
        "research_summary": {
            "input_records": adaptation["summary"]["total_records"],
            "adapted_records": adaptation["summary"]["adapted_records"],
            "adapter_failures": adaptation["summary"]["failed_records"],
            "average_parser_coverage_percent": adaptation["summary"]["average_parser_coverage_percent"],
            "verified_cases": batch.get("summary", {}).get("completed_cases", 0),
            "verification_failures": batch.get("summary", {}).get("failed_cases", 0),
            "answer_accuracy_percent": batch.get("summary", {}).get("answer_accuracy_percent", 0),
            "final_proof_valid_percent": batch.get("summary", {}).get("final_proof_valid_percent", 0),
            "final_chain_valid_percent": batch.get("summary", {}).get("final_chain_valid_percent", 0),
            "valid_answer_but_invalid_reasoning_percent": batch.get("summary", {}).get("valid_answer_but_invalid_reasoning_percent", 0),
            "adapter_distribution": adaptation.get("summary", {}).get("adapter_distribution", {}),
            "extraction_strategy_distribution": adaptation.get("summary", {}).get("extraction_strategy_distribution", {}),
            "parser_error_distribution": adaptation.get("summary", {}).get("parser_error_distribution", {}),
            "reasoning_proof_status_distribution": dict(sorted(reasoning_proof_statuses.items())),
            "reasoning_chain_status_distribution": dict(sorted(reasoning_chain_statuses.items())),
        },
        "rows": combined_rows,
    }


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    fields = [
        "record_index", "case_id", "adapter", "extraction_strategy", "reasoning_step_count",
        "parser_coverage_percent", "ready_for_verification", "answer_correct", "final_proof_status",
        "final_chain_status", "all_reasoning_proof_valid", "all_reasoning_chain_valid",
        "valid_answer_but_invalid_reasoning", "root_error_nodes", "runtime_ms", "engine",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        clean = dict(row)
        if isinstance(clean.get("root_error_nodes"), list):
            clean["root_error_nodes"] = ",".join(clean["root_error_nodes"])
        writer.writerow(clean)
    return buffer.getvalue()


def report_html(result: dict[str, Any]) -> str:
    summary = result.get("research_summary", {})
    rows_html = []
    for row in result.get("rows", []):
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('case_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('adapter', '')))}</td>"
            f"<td>{html.escape(str(row.get('parser_coverage_percent', '')))}</td>"
            f"<td>{html.escape(str(row.get('answer_correct', '')))}</td>"
            f"<td>{html.escape(str(row.get('final_proof_status', '')))}</td>"
            f"<td>{html.escape(str(row.get('final_chain_status', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>VRG Dataset Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172033}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.card{{border:1px solid #d8deea;border-radius:10px;padding:12px;background:#f8fafc}}table{{width:100%;border-collapse:collapse;margin-top:18px}}th,td{{border:1px solid #d8deea;padding:7px;text-align:left}}th{{background:#eef3fa}}</style></head><body>
<h1>Verified Reasoning Graph Dataset Evaluation</h1><div class='cards'>
<div class='card'><small>Adapted</small><h2>{summary.get('adapted_records', 0)} / {summary.get('input_records', 0)}</h2></div>
<div class='card'><small>Parser coverage</small><h2>{summary.get('average_parser_coverage_percent', 0)}%</h2></div>
<div class='card'><small>Answer accuracy</small><h2>{summary.get('answer_accuracy_percent', 0)}%</h2></div>
<div class='card'><small>Valid answer, invalid reasoning</small><h2>{summary.get('valid_answer_but_invalid_reasoning_percent', 0)}%</h2></div>
</div><table><thead><tr><th>Case</th><th>Adapter</th><th>Parser coverage</th><th>Answer correct</th><th>Final proof</th><th>Final chain</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table>
<p>Generated by Verified Reasoning Graph MVP v015.</p></body></html>"""


def save_dataset_outputs(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "dataset_evaluation.json",
        "csv": output_dir / "dataset_cases.csv",
        "html": output_dir / "dataset_report.html",
        "adapted_jsonl": output_dir / "adapted_cases.jsonl",
    }
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["csv"].write_text(rows_to_csv(result.get("rows", [])), encoding="utf-8-sig")
    paths["html"].write_text(report_html(result), encoding="utf-8")
    cases = result.get("adaptation", {}).get("cases", [])
    paths["adapted_jsonl"].write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases), encoding="utf-8")
    return paths
