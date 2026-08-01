from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .hybrid_runner import run_hybrid_proofwriter
from .proofwriter import split_context
from .scientific_text import preview_record_items


def _context_items(text: Any) -> list[str]:
    if isinstance(text, list):
        items = [str(x.get("text") if isinstance(x, dict) else x).strip() for x in text]
        return [x for x in items if x]
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Context를 입력하세요.")
    # Prefer one premise/rule per line. If the user pasted a paragraph, reuse the
    # ProofWriter sentence splitter.
    lines = [line.strip().lstrip("-*• ").strip() for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [x["text"] for x in split_context(raw)]


def build_custom_record(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    context = _context_items(payload.get("context"))
    question = str(payload.get("question") or "").strip()
    if not question:
        raise ValueError("Question을 입력하세요.")
    gold_raw = str(payload.get("gold_answer") or "").strip()
    gold_provided = bool(gold_raw and gold_raw.lower() not in {"auto", "none", "not_provided"})
    gold = gold_raw if gold_provided else "Unknown"
    record_id = str(payload.get("id") or f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    input_mode = str(payload.get("input_mode") or "general_science")
    if input_mode not in {"general_science", "controlled"}:
        raise ValueError("input_mode must be general_science or controlled")
    record = {
        "id": record_id,
        "context": context,
        "question": question,
        "answer": gold,
        "source": "individual_test_lab",
        "input_mode": input_mode,
    }
    return record, gold_provided



def preview_custom_input(payload: dict[str, Any]) -> dict[str, Any]:
    record, gold_provided = build_custom_record(payload)
    preview = preview_record_items(
        [str(x) for x in record["context"]],
        str(record["question"]),
        mode=str(record.get("input_mode") or "general_science"),
    )
    preview["record"] = record
    preview["gold_provided"] = gold_provided
    return preview

def run_custom_test(payload: dict[str, Any], *, output_root: Path, client: Any = None) -> dict[str, Any]:
    record, gold_provided = build_custom_record(payload)
    result = run_hybrid_proofwriter({
        "record": record,
        "model": payload.get("model"),
        "reasoning_effort": payload.get("reasoning_effort", "low"),
        "max_output_tokens": payload.get("max_output_tokens", 5000),
        "max_repair_iterations": payload.get("max_repair_iterations", 1),
        "repair_mode": payload.get("repair_mode", "blind"),
        "use_llm_formalizer": payload.get("use_llm_formalizer", True),
        "use_premise_grounder": payload.get("use_premise_grounder", True),
        "allow_external_premises": payload.get("allow_external_premises", False),
        "custom_instruction": payload.get("custom_instruction", ""),
        "prefer_z3": payload.get("prefer_z3", True),
    }, client=client)
    result["record"] = record
    result["test_lab"] = {
        "gold_provided": gold_provided,
        "input_mode": record.get("input_mode", "general_science"),
        "evaluation_basis": "user_gold_and_context" if gold_provided else "context_derived_only",
        "note": "Gold가 비어 있으면 정답 일치 대신 Context-derived label과 Graph validity를 사용합니다.",
        "deterministic_preview": preview_record_items(
            [str(x) for x in record["context"]],
            str(record["question"]),
            mode=str(record.get("input_mode") or "general_science"),
        ),
    }
    if not gold_provided:
        result["summary"]["gold_label"] = None
        result["summary"]["final_answer_correct"] = None
        for attempt in result.get("attempts") or []:
            attempt.get("analysis", {}).get("classification", {})["gold_label"] = None
            attempt.get("analysis", {}).get("classification", {})["answer_correct"] = None
        result.get("final_universal_graph", {}).setdefault("proofwriter", {})["three_way_gold_label"] = None
    run_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result["test_lab"]["run_id"] = run_id
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_custom_tests(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return rows
    for path in sorted(output_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        result_path = path / "result.json"
        if not path.is_dir() or not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.append({
            "run_id": path.name,
            "record_id": data.get("record_id"),
            "question": (data.get("record") or {}).get("question"),
            "final_answer": (data.get("summary") or {}).get("final_answer"),
            "context_label": (data.get("summary") or {}).get("context_label"),
            "final_pass": (data.get("summary") or {}).get("final_pass"),
            "repair_count": (data.get("summary") or {}).get("repair_count"),
            "input_mode": (data.get("test_lab") or {}).get("input_mode"),
            "formalization_fallback_used": (data.get("summary") or {}).get("formalization_fallback_used"),
        })
    return rows


def load_custom_test(output_root: Path, run_id: str) -> dict[str, Any]:
    path = (output_root / run_id / "result.json").resolve()
    root = output_root.resolve()
    if root not in path.parents or not path.exists():
        raise ValueError("Unknown Test Lab run")
    return json.loads(path.read_text(encoding="utf-8"))
