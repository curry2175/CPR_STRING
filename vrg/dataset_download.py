from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DATASET_ID = "renma/ProofWriter"
CONFIG = "default"
SPLIT = "validation"
EXPECTED_ROWS = 600
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DATASET_PAGE = "https://huggingface.co/datasets/renma/ProofWriter"
SCHEMA_VERSION = "0.19.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_fetch_json(url: str, *, timeout: float = 60.0, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Hybrid-VeriCoT-VRG/0.18 (+local research runner)",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Hugging Face API returned a non-object response")
            return payload
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 1.0 * (2**attempt)))
    raise RuntimeError(f"ProofWriter download failed after {retries + 1} attempts: {last_error}")


def _rows_url(offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def _validate_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    required = ("id", "context", "question", "answer")
    missing = [name for name in required if not str(record.get(name, "")).strip()]
    if missing:
        raise ValueError(f"Downloaded row {index} is missing required fields: {', '.join(missing)}")
    answer = str(record.get("answer")).strip().upper()
    if answer not in {"A", "B", "C", "TRUE", "FALSE", "UNKNOWN"}:
        raise ValueError(f"Downloaded row {index} has an unsupported answer label: {record.get('answer')!r}")
    return record


def canonical_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n"


@dataclass(frozen=True)
class DownloadResult:
    dataset_path: Path
    metadata_path: Path
    row_count: int
    cached: bool
    source_dataset: str = DATASET_ID
    config: str = CONFIG
    split: str = SPLIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset_path": str(self.dataset_path),
            "metadata_path": str(self.metadata_path),
            "row_count": self.row_count,
            "expected_row_count": EXPECTED_ROWS,
            "cached": self.cached,
            "source_dataset": self.source_dataset,
            "config": self.config,
            "split": self.split,
            "dataset_page": DATASET_PAGE,
        }


def download_proofwriter_600(
    target_dir: Path,
    *,
    refresh: bool = False,
    batch_size: int = 100,
    fetch_json: Callable[[str], dict[str, Any]] | None = None,
) -> DownloadResult:
    """Download the 600-row renma/ProofWriter validation split through the HF Dataset Viewer API.

    The function intentionally uses the public /rows endpoint and only the Python standard library,
    so users do not need Git, git-lfs, pyarrow, or the Hugging Face datasets package.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / "renma_ProofWriter_validation_600.jsonl"
    metadata_path = target_dir / "renma_ProofWriter_validation_600.metadata.json"

    if dataset_path.exists() and not refresh:
        records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        for index, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ValueError(f"Cached dataset row {index} is not a JSON object")
            _validate_record(record, index)
        return DownloadResult(dataset_path, metadata_path, len(records), True)

    fetch = fetch_json or _default_fetch_json
    batch_size = max(1, min(100, int(batch_size)))
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_hint: int | None = None
    offset = 0

    while True:
        payload = fetch(_rows_url(offset, batch_size))
        if total_hint is None:
            raw_total = payload.get("num_rows_total")
            if isinstance(raw_total, int) and raw_total >= 0:
                total_hint = raw_total
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("Hugging Face /rows response is missing the rows list")
        if not rows:
            break
        for item in rows:
            record = item.get("row") if isinstance(item, dict) and isinstance(item.get("row"), dict) else item
            if not isinstance(record, dict):
                raise ValueError(f"Downloaded row {len(records) + 1} is not a JSON object")
            record = _validate_record(dict(record), len(records) + 1)
            record_id = str(record["id"])
            if record_id in seen_ids:
                raise ValueError(f"Duplicate record id returned by the dataset API: {record_id}")
            seen_ids.add(record_id)
            records.append(record)
        offset += len(rows)
        if total_hint is not None and offset >= total_hint:
            break
        if len(rows) < batch_size:
            break

    if not records:
        raise ValueError("ProofWriter download returned zero rows")
    if total_hint is not None and len(records) != total_hint:
        raise ValueError(f"ProofWriter download was incomplete: expected {total_hint}, received {len(records)}")

    temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    temporary.write_text(canonical_jsonl(records), encoding="utf-8")
    os.replace(temporary, dataset_path)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "downloaded_at": _utc_now(),
        "source_dataset": DATASET_ID,
        "dataset_page": DATASET_PAGE,
        "api_endpoint": ROWS_ENDPOINT,
        "config": CONFIG,
        "split": SPLIT,
        "row_count": len(records),
        "expected_row_count_at_build_time": EXPECTED_ROWS,
        "columns": sorted({key for record in records for key in record}),
        "notes": "This is the 600-row validation split used by the current ProofWriter adapter, not the 845k-row expanded tasksource corpus.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return DownloadResult(dataset_path, metadata_path, len(records), False)


def proofwriter_download_status(target_dir: Path) -> dict[str, Any]:
    target_dir = Path(target_dir)
    dataset_path = target_dir / "renma_ProofWriter_validation_600.jsonl"
    metadata_path = target_dir / "renma_ProofWriter_validation_600.metadata.json"
    if not dataset_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "source_dataset": DATASET_ID,
            "config": CONFIG,
            "split": SPLIT,
            "expected_row_count": EXPECTED_ROWS,
        }
    try:
        row_count = sum(1 for line in dataset_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    except OSError:
        row_count = 0
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
        "row_count": row_count,
        "source_dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "metadata": metadata,
    }
