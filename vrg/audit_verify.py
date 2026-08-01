from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from typing import Any

MAX_PACKAGE_BYTES = 25 * 1024 * 1024
MAX_MEMBER_BYTES = 5 * 1024 * 1024
REQUIRED_FILES = {"manifest.json", "input_case.json", "verified_graph.json", "nodes.csv", "edges.csv", "smt2/index.json"}


def _safe_names(archive: zipfile.ZipFile) -> list[str]:
    names: list[str] = []
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"Unsafe ZIP path: {info.filename}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(f"ZIP member is too large: {info.filename}")
        names.append(name)
    return names


def _rerun_smt2(text: str) -> tuple[str | None, str | None]:
    try:
        import z3  # type: ignore
    except ImportError:
        return None, "z3-solver is not installed"
    try:
        cleaned = re.sub(r"\(check-sat\)\s*", "", text)
        solver = z3.Solver()
        solver.from_string(cleaned)
        status = str(solver.check()).lower()
        return status, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def verify_audit_package_bytes(payload: bytes, *, rerun_smt: bool = True) -> dict[str, Any]:
    if not payload:
        raise ValueError("Audit ZIP is empty")
    if len(payload) > MAX_PACKAGE_BYTES:
        raise ValueError("Audit ZIP exceeds the 25 MB safety limit")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = _safe_names(archive)
        name_set = set(names)
        missing_required = sorted(REQUIRED_FILES - name_set)
        manifest = json.loads(archive.read("manifest.json")) if "manifest.json" in name_set else {}
        expected_hashes = manifest.get("file_sha256") if isinstance(manifest, dict) else {}
        if not isinstance(expected_hashes, dict):
            expected_hashes = {}

        hash_rows: list[dict[str, Any]] = []
        for name, expected in sorted(expected_hashes.items()):
            if name not in name_set:
                hash_rows.append({"path": name, "expected": expected, "actual": None, "match": False, "error": "missing"})
                continue
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            hash_rows.append({"path": name, "expected": expected, "actual": actual, "match": actual == expected, "error": None})

        extra_files = sorted(name_set - set(expected_hashes) - {"manifest.json"})
        graph = json.loads(archive.read("verified_graph.json")) if "verified_graph.json" in name_set else {}
        input_case = json.loads(archive.read("input_case.json")) if "input_case.json" in name_set else {}
        metadata_checks = {
            "case_id_matches": manifest.get("case_id") == graph.get("case_id"),
            "engine_matches": manifest.get("engine") == graph.get("engine"),
            "final_proof_matches": manifest.get("final_proof_status") == graph.get("summary", {}).get("final_proof_status"),
            "final_chain_matches": manifest.get("final_chain_status") == graph.get("summary", {}).get("final_chain_status"),
            "answer_correct_matches": manifest.get("answer_correct") == graph.get("answer_correct"),
            "input_case_id_matches": str(input_case.get("id") or input_case.get("case_id") or "") == str(graph.get("case_id") or ""),
        }

        node_lookup = {str(node.get("id")): node for node in graph.get("nodes", []) if isinstance(node, dict)}
        smt_index = json.loads(archive.read("smt2/index.json")) if "smt2/index.json" in name_set else []
        replay_rows: list[dict[str, Any]] = []
        if rerun_smt and isinstance(smt_index, list):
            for item in smt_index:
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("node_id") or "")
                check = str(item.get("check") or "")
                path = str(item.get("path") or "")
                expected_key = "consistency_check_result" if check == "consistency" else "entailment_check_result"
                expected = node_lookup.get(node_id, {}).get(expected_key)
                if path not in name_set:
                    replay_rows.append({"node_id": node_id, "check": check, "path": path, "expected": expected, "observed": None, "match": False, "error": "missing SMT2 file"})
                    continue
                observed, error = _rerun_smt2(archive.read(path).decode("utf-8", errors="replace"))
                match = observed == expected if observed is not None else None
                replay_rows.append({"node_id": node_id, "check": check, "path": path, "expected": expected, "observed": observed, "match": match, "error": error})

        hash_pass = bool(hash_rows) and all(row["match"] for row in hash_rows)
        metadata_pass = all(metadata_checks.values()) if metadata_checks else False
        replay_available = bool(replay_rows) and any(row["observed"] is not None for row in replay_rows)
        replay_pass = all(row["match"] is True for row in replay_rows if row["observed"] is not None) if replay_available else None
        integrity_pass = not missing_required and hash_pass and metadata_pass
        full_replay_pass = (integrity_pass and replay_pass) if replay_available else None
        return {
            "schema_version": "0.15.0",
            "package": {
                "case_id": manifest.get("case_id"),
                "schema_version": manifest.get("schema_version"),
                "engine": manifest.get("engine"),
                "generated_at_utc": manifest.get("generated_at_utc"),
                "file_count": len(name_set),
            },
            "summary": {
                "integrity_pass": integrity_pass,
                "full_replay_pass": full_replay_pass,
                "required_files_present": not missing_required,
                "hash_match_count": sum(row["match"] is True for row in hash_rows),
                "hash_check_count": len(hash_rows),
                "metadata_match_count": sum(metadata_checks.values()),
                "metadata_check_count": len(metadata_checks),
                "smt_replay_available": replay_available,
                "smt_replay_match_count": sum(row["match"] is True for row in replay_rows),
                "smt_replay_check_count": len(replay_rows),
            },
            "missing_required_files": missing_required,
            "extra_unhashed_files": extra_files,
            "hash_checks": hash_rows,
            "metadata_checks": metadata_checks,
            "smt_replay": replay_rows,
        }
