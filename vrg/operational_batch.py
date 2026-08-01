from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hybrid_runner import run_hybrid_proofwriter

SCHEMA_VERSION = "0.19.0"
TERMINAL_STATUSES = {"completed", "completed_with_errors", "paused", "stopped", "error"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_component(value: Any, fallback: str = "item") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return (text or fallback)[:100]


def parse_records_input(value: Any) -> list[dict[str, Any]]:
    """Accept a Python list, JSON array, wrapper object, single record, or JSONL text."""
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(value.get(key), list):
                records = value[key]
                break
        else:
            records = [value]
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("Dataset input is empty")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            records = []
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} must be an object")
                records.append(row)
        else:
            return parse_records_input(parsed)
    clean: list[dict[str, Any]] = []
    for index, row in enumerate(records, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index} must be a JSON object")
        clean.append(row)
    if not clean:
        raise ValueError("Dataset contains no records")
    return clean


def canonical_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in records) + "\n"


def dataset_fingerprint(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_jsonl(records).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_index(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["index"])] = row
    return rows


def _response_ids(value: Any, found: set[str] | None = None) -> set[str]:
    if found is None:
        found = set()
    if isinstance(value, dict):
        response_id = value.get("response_id")
        if response_id:
            found.add(str(response_id))
        for child in value.values():
            _response_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            _response_ids(child, found)
    return found


def _compact_result(index: int, record: dict[str, Any], result: dict[str, Any], case_file: str, elapsed_ms: float) -> dict[str, Any]:
    summary = result.get("summary") or {}
    usage = summary.get("total_usage") or {}
    return {
        "index": index,
        "record_id": str(record.get("id") or f"record_{index}"),
        "status": "completed",
        "final_pass": bool(summary.get("final_pass")),
        "initial_pass": bool(summary.get("initial_pass")),
        "initial_answer": summary.get("initial_answer"),
        "final_answer": summary.get("final_answer"),
        "context_label": summary.get("context_label"),
        "gold_label": summary.get("gold_label"),
        "final_answer_correct": bool(summary.get("final_answer_correct")),
        "repair_count": int(summary.get("repair_count") or 0),
        "attempt_count": int(summary.get("attempt_count") or 0),
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "api_calls": len(_response_ids(result)),
        "elapsed_ms": round(elapsed_ms, 3),
        "case_file": case_file,
        "completed_at": _utc_now(),
    }


def _compact_error(index: int, record: dict[str, Any], exc: Exception, retry_attempts: int, error_file: str, elapsed_ms: float) -> dict[str, Any]:
    return {
        "index": index,
        "record_id": str(record.get("id") or f"record_{index}"),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "retry_attempts": retry_attempts,
        "elapsed_ms": round(elapsed_ms, 3),
        "error_file": error_file,
        "completed_at": _utc_now(),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }


def _summary_from_rows(total_records: int, rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows.values() if row.get("status") == "completed"]
    failed = [row for row in rows.values() if row.get("status") == "failed"]
    return {
        "total_records": total_records,
        "processed_records": len(rows),
        "remaining_records": max(0, total_records - len(rows)),
        "completed_records": len(completed),
        "failed_records": len(failed),
        "final_pass_count": sum(bool(row.get("final_pass")) for row in completed),
        "initial_pass_count": sum(bool(row.get("initial_pass")) for row in completed),
        "repair_success_count": sum(bool(row.get("repair_count") and row.get("final_pass")) for row in completed),
        "answer_correct_count": sum(bool(row.get("final_answer_correct")) for row in completed),
        "total_input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows.values()),
        "total_output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows.values()),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows.values()),
        "total_api_calls": sum(int(row.get("api_calls") or 0) for row in rows.values()),
        "mean_elapsed_ms": round(sum(float(row.get("elapsed_ms") or 0) for row in rows.values()) / len(rows), 3) if rows else 0,
    }


def _write_summary_files(run_dir: Path, state: dict[str, Any], rows: dict[int, dict[str, Any]]) -> None:
    ordered = [rows[key] for key in sorted(rows)]
    summary = _summary_from_rows(int(state["total_records"]), rows)
    summary["run_id"] = state["run_id"]
    summary["status"] = state["status"]
    summary["dataset_fingerprint"] = state["dataset_fingerprint"]
    summary["updated_at"] = _utc_now()
    _atomic_write_json(run_dir / "summary.json", summary)
    fieldnames = [
        "index", "record_id", "status", "initial_pass", "final_pass", "initial_answer", "final_answer",
        "context_label", "gold_label", "final_answer_correct", "repair_count", "attempt_count", "input_tokens",
        "output_tokens", "total_tokens", "api_calls", "elapsed_ms", "retry_attempts", "error_type", "error",
        "case_file", "error_file", "completed_at",
    ]
    with (run_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _selection(total: int, mode: str, pilot_count: int, start_index: int, end_index: int | None) -> list[int]:
    if mode == "pilot":
        return list(range(1, min(total, max(1, pilot_count)) + 1))
    if mode == "range":
        start = max(1, start_index)
        end = min(total, end_index or total)
        if start > end:
            raise ValueError("start_index must not exceed end_index")
        return list(range(start, end + 1))
    if mode == "full":
        return list(range(1, total + 1))
    raise ValueError("mode must be pilot, range, or full")


@dataclass
class RuntimeJob:
    run_id: str
    thread: threading.Thread | None = None
    pause_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class OperationalBatchManager:
    """Local, checkpointed runner for pilot-then-full ProofWriter evaluation."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.runs_root = self.output_root / "hybrid_runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, RuntimeJob] = {}
        self._manager_lock = threading.Lock()

    def _run_dir(self, run_id: str) -> Path:
        return self.runs_root / _safe_component(run_id, "run")

    def list_runs(self) -> list[dict[str, Any]]:
        result = []
        for state_path in sorted(self.runs_root.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            state = _read_json(state_path, {})
            if state:
                result.append(state)
        return result[:50]

    def status(self, run_id: str) -> dict[str, Any]:
        run_dir = self._run_dir(run_id)
        state = _read_json(run_dir / "state.json")
        if not state:
            raise ValueError(f"Unknown run_id: {run_id}")
        rows = _read_index(run_dir / "index.jsonl")
        state = {**state, "summary": _summary_from_rows(int(state["total_records"]), rows)}
        state["download_files"] = {
            "summary_json": f"/api/hybrid-batch-job/{run_id}/file/summary.json",
            "summary_csv": f"/api/hybrid-batch-job/{run_id}/file/summary.csv",
            "predictions_jsonl": f"/api/hybrid-batch-job/{run_id}/file/predictions.jsonl",
            "archive_zip": f"/api/hybrid-batch-job/{run_id}/archive",
        }
        return state

    def _prepare_run(self, payload: dict[str, Any]) -> tuple[str, Path, list[dict[str, Any]], dict[str, Any]]:
        requested_run_id = str(payload.get("run_id") or "").strip()
        if requested_run_id:
            run_id = _safe_component(requested_run_id, "run")
            run_dir = self._run_dir(run_id)
            if not run_dir.exists():
                raise ValueError(f"Run does not exist: {run_id}")
            records = parse_records_input((run_dir / "dataset.jsonl").read_text(encoding="utf-8"))
            state = _read_json(run_dir / "state.json", {})
            supplied = payload.get("records") if isinstance(payload.get("records"), list) else payload.get("dataset_text", payload.get("jsonl"))
            if supplied:
                supplied_records = parse_records_input(supplied)
                if dataset_fingerprint(supplied_records) != state.get("dataset_fingerprint"):
                    raise ValueError("The supplied dataset does not match the existing run checkpoint")
            return run_id, run_dir, records, state

        raw = payload.get("records") if isinstance(payload.get("records"), list) else payload.get("dataset_text", payload.get("jsonl"))
        records = parse_records_input(raw)
        fingerprint = dataset_fingerprint(records)
        run_id = _safe_component(
            payload.get("new_run_id") or f"proofwriter_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fingerprint[:8]}",
            "proofwriter_run",
        )
        run_dir = self._run_dir(run_id)
        if run_dir.exists():
            run_id = f"{run_id}_{uuid.uuid4().hex[:6]}"
            run_dir = self._run_dir(run_id)
        (run_dir / "cases").mkdir(parents=True, exist_ok=True)
        (run_dir / "errors").mkdir(parents=True, exist_ok=True)
        (run_dir / "dataset.jsonl").write_text(canonical_jsonl(records), encoding="utf-8")
        state = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "status": "created",
            "total_records": len(records),
            "dataset_fingerprint": fingerprint,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "phase_history": [],
            "message": "Run created",
            "dataset_source": payload.get("dataset_source"),
        }
        _atomic_write_json(run_dir / "state.json", state)
        return run_id, run_dir, records, state

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id, run_dir, records, state = self._prepare_run(payload)
        with self._manager_lock:
            existing = self._jobs.get(run_id)
            if existing and existing.thread and existing.thread.is_alive():
                latest_state = _read_json(run_dir / "state.json", {})
                if latest_state.get("status") in TERMINAL_STATUSES:
                    existing.thread.join(timeout=1.0)
                if existing.thread.is_alive():
                    raise ValueError(f"Run {run_id} is already running")
            runtime = existing or RuntimeJob(run_id=run_id)
            runtime.pause_event.clear()
            self._jobs[run_id] = runtime

        mode = str(payload.get("mode") or "pilot").lower()
        settings = {
            "mode": mode,
            "pilot_count": max(1, int(payload.get("pilot_count") or 10)),
            "start_index": max(1, int(payload.get("start_index") or 1)),
            "end_index": int(payload["end_index"]) if payload.get("end_index") not in (None, "", 0, "0") else None,
            "max_workers": max(1, min(4, int(payload.get("max_workers") or 1))),
            "max_retries": max(0, min(5, int(payload.get("max_retries") or 2))),
            "retry_failed": bool(payload.get("retry_failed", True)),
            "max_total_tokens": max(0, int(payload.get("max_total_tokens") or 0)),
            "max_failures": max(0, int(payload.get("max_failures") or 0)),
            "model": str(payload.get("model") or "gpt-5.6"),
            "reasoning_effort": str(payload.get("reasoning_effort") or "low"),
            "max_output_tokens": int(payload.get("max_output_tokens") or 5000),
            "max_repair_iterations": int(payload.get("max_repair_iterations") or 1),
            "repair_mode": str(payload.get("repair_mode") or "blind"),
            "use_llm_formalizer": bool(payload.get("use_llm_formalizer", True)),
            "use_premise_grounder": bool(payload.get("use_premise_grounder", True)),
            "allow_external_premises": bool(payload.get("allow_external_premises", False)),
            "prefer_z3": bool(payload.get("prefer_z3", True)),
            "custom_instruction": str(payload.get("custom_instruction") or ""),
            "dataset_source": payload.get("dataset_source") or state.get("dataset_source"),
        }
        selection = _selection(len(records), mode, settings["pilot_count"], settings["start_index"], settings["end_index"])
        state.update({
            "status": "queued",
            "updated_at": _utc_now(),
            "active_mode": mode,
            "selected_records": len(selection),
            "settings": settings,
            "message": f"Queued {mode} phase",
            "pause_requested": False,
        })
        state.setdefault("phase_history", []).append({"mode": mode, "started_at": _utc_now(), "selected_records": len(selection)})
        _atomic_write_json(run_dir / "state.json", state)
        _atomic_write_json(run_dir / "settings.json", settings)

        runtime.thread = threading.Thread(
            target=self._run_phase,
            args=(runtime, run_dir, records, selection, settings),
            daemon=True,
            name=f"vrg-{run_id}",
        )
        runtime.thread.start()
        return self.status(run_id)

    def pause(self, run_id: str) -> dict[str, Any]:
        runtime = self._jobs.get(run_id)
        run_dir = self._run_dir(run_id)
        state = _read_json(run_dir / "state.json")
        if not state:
            raise ValueError(f"Unknown run_id: {run_id}")
        if runtime:
            runtime.pause_event.set()
        state["pause_requested"] = True
        state["message"] = "Pause requested; currently running records will finish first"
        state["updated_at"] = _utc_now()
        _atomic_write_json(run_dir / "state.json", state)
        return self.status(run_id)

    def archive(self, run_id: str) -> Path:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            raise ValueError(f"Unknown run_id: {run_id}")
        archive_path = run_dir.parent / f"{run_dir.name}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(run_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(run_dir.parent))
        return archive_path

    def file_path(self, run_id: str, relative_name: str) -> Path:
        run_dir = self._run_dir(run_id).resolve()
        candidate = (run_dir / relative_name).resolve()
        if run_dir not in candidate.parents and candidate != run_dir:
            raise ValueError("Invalid file path")
        if not candidate.exists() or not candidate.is_file():
            raise ValueError(f"File not found: {relative_name}")
        return candidate

    def _process_one(self, index: int, record: dict[str, Any], settings: dict[str, Any], run_dir: Path) -> dict[str, Any]:
        started = time.perf_counter()
        last_exc: Exception | None = None
        for retry in range(settings["max_retries"] + 1):
            try:
                result = run_hybrid_proofwriter({**settings, "record": record})
                filename = f"{index:06d}_{_safe_component(record.get('id'), f'record_{index}')}.json"
                path = run_dir / "cases" / filename
                _atomic_write_json(path, result)
                return _compact_result(index, record, result, str(path.relative_to(run_dir)), (time.perf_counter() - started) * 1000)
            except Exception as exc:  # per-record API/parser/verifier failure
                last_exc = exc
                if retry < settings["max_retries"]:
                    time.sleep(min(30.0, 2.0 ** retry))
        assert last_exc is not None
        filename = f"{index:06d}_{_safe_component(record.get('id'), f'record_{index}')}.json"
        error_path = run_dir / "errors" / filename
        error_payload = {
            "index": index,
            "record_id": record.get("id"),
            "error_type": type(last_exc).__name__,
            "error": str(last_exc),
            "retry_attempts": settings["max_retries"],
            "record": record,
            "failed_at": _utc_now(),
        }
        _atomic_write_json(error_path, error_payload)
        return _compact_error(index, record, last_exc, settings["max_retries"], str(error_path.relative_to(run_dir)), (time.perf_counter() - started) * 1000)

    def _run_phase(self, runtime: RuntimeJob, run_dir: Path, records: list[dict[str, Any]], selection: list[int], settings: dict[str, Any]) -> None:
        state_path = run_dir / "state.json"
        index_path = run_dir / "index.jsonl"
        state = _read_json(state_path, {})
        state.update({"status": "running", "message": f"Running {settings['mode']} phase", "updated_at": _utc_now(), "phase_started_at": _utc_now()})
        _atomic_write_json(state_path, state)
        rows = _read_index(index_path)
        pending: list[int] = []
        for index in selection:
            row = rows.get(index)
            if not row:
                pending.append(index)
            elif row.get("status") == "failed" and settings["retry_failed"]:
                pending.append(index)
        state["pending_at_phase_start"] = len(pending)
        _atomic_write_json(state_path, state)

        try:
            cursor = 0
            while cursor < len(pending):
                if runtime.pause_event.is_set():
                    state = _read_json(state_path, state)
                    state.update({"status": "paused", "message": "Paused at checkpoint", "updated_at": _utc_now(), "pause_requested": False})
                    _atomic_write_json(state_path, state)
                    break

                current_rows = _read_index(index_path)
                summary = _summary_from_rows(len(records), current_rows)
                if settings["max_total_tokens"] and summary["total_tokens"] >= settings["max_total_tokens"]:
                    state = _read_json(state_path, state)
                    state.update({"status": "stopped", "stop_reason": "max_total_tokens", "message": "Stopped at token limit", "updated_at": _utc_now()})
                    _atomic_write_json(state_path, state)
                    break
                if settings["max_failures"] and summary["failed_records"] >= settings["max_failures"]:
                    state = _read_json(state_path, state)
                    state.update({"status": "stopped", "stop_reason": "max_failures", "message": "Stopped at failure limit", "updated_at": _utc_now()})
                    _atomic_write_json(state_path, state)
                    break

                batch_indices = pending[cursor: cursor + settings["max_workers"]]
                cursor += len(batch_indices)
                if settings["max_workers"] == 1:
                    completed_rows = [self._process_one(batch_indices[0], records[batch_indices[0] - 1], settings, run_dir)]
                else:
                    completed_rows = []
                    with ThreadPoolExecutor(max_workers=settings["max_workers"], thread_name_prefix="vrg-worker") as pool:
                        future_map = {
                            pool.submit(self._process_one, idx, records[idx - 1], settings, run_dir): idx for idx in batch_indices
                        }
                        for future in as_completed(future_map):
                            completed_rows.append(future.result())
                with runtime.lock:
                    for row in sorted(completed_rows, key=lambda x: int(x["index"])):
                        _append_jsonl(index_path, row)
                        rows[int(row["index"])] = row
                    state = _read_json(state_path, state)
                    state.update({
                        "updated_at": _utc_now(),
                        "last_completed_index": max(int(row["index"]) for row in completed_rows),
                        "message": f"Processed {len(rows)} / {len(records)} records",
                    })
                    _atomic_write_json(state_path, state)
                    _write_summary_files(run_dir, state, rows)

            else:
                state = _read_json(state_path, state)
                rows = _read_index(index_path)
                selected_done = all(index in rows and (rows[index].get("status") == "completed" or not settings["retry_failed"]) for index in selection)
                if settings["mode"] == "pilot":
                    state["pilot_completed"] = selected_done
                elif settings["mode"] == "full":
                    state["full_completed"] = len(rows) >= len(records)
                state.update({
                    "status": "completed_with_errors" if any(row.get("status") == "failed" for row in rows.values()) else "completed",
                    "message": f"{settings['mode'].capitalize()} phase completed",
                    "updated_at": _utc_now(),
                    "phase_completed_at": _utc_now(),
                })
                if state.get("phase_history"):
                    state["phase_history"][-1]["completed_at"] = _utc_now()
                    state["phase_history"][-1]["status"] = state["status"]
                _atomic_write_json(state_path, state)
                _write_summary_files(run_dir, state, rows)
        except Exception as exc:
            state = _read_json(state_path, state)
            state.update({
                "status": "error",
                "message": f"Runner error: {type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "updated_at": _utc_now(),
            })
            _atomic_write_json(state_path, state)
            rows = _read_index(index_path)
            _write_summary_files(run_dir, state, rows)


def run_operational_batch_sync(payload: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """CLI/test helper: start a job and block until its phase reaches a terminal state."""
    manager = OperationalBatchManager(output_root)
    status = manager.start(payload)
    run_id = status["run_id"]
    while True:
        current = manager.status(run_id)
        if current["status"] in TERMINAL_STATUSES:
            return current
        time.sleep(0.05)
