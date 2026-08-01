from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        clean = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            clean[field] = value
        writer.writerow(clean)
    return buffer.getvalue().encode("utf-8-sig")


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value or "case"


def _report_markdown(case: dict[str, Any], result: dict[str, Any]) -> str:
    summary = result.get("summary", {})
    lines = [
        f"# Verified Reasoning Graph Audit Report: {result.get('case_id', 'case')}",
        "",
        "## Verdict",
        "",
        f"- Question: {result.get('question')}",
        f"- Predicted / Gold: {result.get('predicted_answer')} / {result.get('gold_answer')}",
        f"- Answer correct: {result.get('answer_correct')}",
        f"- Final proof status: {summary.get('final_proof_status')}",
        f"- Final chain status: {summary.get('final_chain_status')}",
        f"- Engine: {result.get('engine')}",
        "",
        "## Reasoning nodes",
        "",
        "| ID | Text | Proof | Chain | Selected proof dependencies |",
        "|---|---|---|---|---|",
    ]
    for node in result.get("nodes", []):
        if node.get("kind") not in {"reasoning", "answer"}:
            continue
        text = str(node.get("text") or "").replace("|", "\\|")
        lines.append(
            f"| {node.get('id')} | {text} | {node.get('proof_status')} | {node.get('chain_status')} | "
            f"{node.get('reasoning_role') or '-'} | {node.get('local_support_status') or '-'} | "
            f"{node.get('final_proof_necessity') or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility notes",
            "",
            "- Each available SMT-LIB query is included under `smt2/`.",
            "- `manifest.json` contains SHA-256 hashes for every archived payload file.",
            "- A selected unsat core is a sufficient/minimized support set from this run, not an enumeration of every possible proof.",
            "- Semantic `related_to` edges are advisory and are not proof-usable.",
            "",
            "## Input case",
            "",
            "```json",
            json.dumps(case, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    return "\n".join(lines)


def _report_html(markdown_text: str, result: dict[str, Any]) -> str:
    # Small self-contained report; no external markdown dependency.
    summary = result.get("summary", {})
    rows = []
    for node in result.get("nodes", []):
        if node.get("kind") not in {"reasoning", "answer"}:
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(node.get('id')))}</td>"
            f"<td>{html.escape(str(node.get('text') or ''))}</td>"
            f"<td>{html.escape(str(node.get('proof_status')))}</td>"
            f"<td>{html.escape(str(node.get('chain_status')))}</td>"
            f"<td>{html.escape(str(node.get('reasoning_role') or '-'))}</td>"
            f"<td>{html.escape(str(node.get('local_support_status') or '-'))}</td>"
            f"<td>{html.escape(str(node.get('final_proof_necessity') or '-'))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>VRG Audit Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;color:#172033}}h1,h2{{color:#152a52}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{border:1px solid #d8deea;border-radius:10px;padding:14px;background:#f8fafc}}table{{width:100%;border-collapse:collapse;margin-top:14px}}th,td{{border:1px solid #d8deea;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef3fa}}code,pre{{font-family:Consolas,monospace}}.note{{color:#5d6778;font-size:13px}}</style></head>
<body><h1>Verified Reasoning Graph Audit Report</h1>
<p><strong>Case:</strong> {html.escape(str(result.get('case_id')))}<br><strong>Question:</strong> {html.escape(str(result.get('question')))}</p>
<div class="cards"><div class="card"><small>Final proof</small><h2>{html.escape(str(summary.get('final_proof_status')))}</h2></div><div class="card"><small>Final chain</small><h2>{html.escape(str(summary.get('final_chain_status')))}</h2></div><div class="card"><small>Predicted / gold</small><h2>{html.escape(str(result.get('predicted_answer')))} / {html.escape(str(result.get('gold_answer')))}</h2></div></div>
<h2>Reasoning and answer nodes</h2><table><thead><tr><th>ID</th><th>Text</th><th>Proof</th><th>Chain</th><th>Role</th><th>Local support</th><th>Final necessity</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Audit notes</h2><ul><li>SMT-LIB queries are included in the package under <code>smt2/</code>.</li><li>File hashes are recorded in <code>manifest.json</code>.</li><li>Selected cores do not enumerate all possible proofs.</li></ul>
<details><summary>Plain-text report source</summary><pre>{html.escape(markdown_text)}</pre></details>
<p class="note">Generated by Verified Reasoning Graph MVP v015.</p></body></html>"""


def build_audit_package(
    case: dict[str, Any],
    result: dict[str, Any],
    output_dir: Path,
) -> Path:
    case_id = _safe_filename(str(result.get("case_id") or case.get("id") or "case"))
    generated_at = datetime.now(timezone.utc).isoformat()
    files: dict[str, bytes] = {}
    files["input_case.json"] = _json_bytes(case)
    files["verified_graph.json"] = _json_bytes(result)

    node_fields = [
        "id", "kind", "order", "text", "proof_status", "chain_status", "formal", "parse_error",
        "proof_dependencies", "reasoning_dependencies", "chain_dependency_source",
        "declared_reasoning_dependencies", "inferred_reasoning_dependencies",
        "blocking_parent_nodes", "upstream_error_nodes",
        "reasoning_role", "reasoning_error_type", "local_support_status",
        "dependency_confidence", "dependency_candidate_count", "declared_dependency_sufficient",
        "minimal_proof_paths", "minimal_proof_count", "final_proof_necessity",
        "atomicity_status", "atomic_claim_count_estimate",
        "consistency_check_result", "entailment_check_result", "verification_origin",
    ]
    edge_fields = ["source", "target", "relation"]
    files["nodes.csv"] = _csv_bytes(result.get("nodes", []), node_fields)
    files["edges.csv"] = _csv_bytes(result.get("edges", []), edge_fields)

    report_md = _report_markdown(case, result)
    files["report.md"] = report_md.encode("utf-8")
    files["report.html"] = _report_html(report_md, result).encode("utf-8")

    smt_index: list[dict[str, str]] = []
    for node in result.get("nodes", []):
        node_id = _safe_filename(str(node.get("id") or "node"))
        for label, key in (
            ("consistency", "consistency_query_smtlib"),
            ("entailment", "entailment_query_smtlib"),
        ):
            query = node.get(key)
            if not query:
                continue
            name = f"smt2/{node_id}_{label}.smt2"
            files[name] = str(query).encode("utf-8")
            smt_index.append({"node_id": str(node.get("id")), "check": label, "path": name})
    files["smt2/index.json"] = _json_bytes(smt_index)

    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in sorted(files.items())}
    manifest = {
        "schema_version": "0.15.0",
        "package_type": "verified_reasoning_graph_audit",
        "generated_at_utc": generated_at,
        "case_id": result.get("case_id"),
        "engine": result.get("engine"),
        "final_proof_status": result.get("summary", {}).get("final_proof_status"),
        "final_chain_status": result.get("summary", {}).get("final_chain_status"),
        "answer_correct": result.get("answer_correct"),
        "file_sha256": hashes,
        "limitations": [
            "Controlled-English parser coverage is limited.",
            "Unsat cores identify one sufficient/minimized support set, not every possible proof.",
            "Advisory related_to semantic relations are not proof-usable.",
        ],
    }
    files["manifest.json"] = _json_bytes(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_id}_audit_package.zip"
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    return output_path
