from __future__ import annotations

import copy
import re
from typing import Any

from .parser import parse_statement

INFERENCE_PREFIX = re.compile(r"^(?:therefore|thus|hence|consequently|so)\s*,?\s+", re.I)


def _coerce_step(item: Any, index: int) -> dict[str, Any]:
    if isinstance(item, str):
        return {"id": f"s{index}", "text": item}
    if isinstance(item, dict):
        return copy.deepcopy(item)
    return {"id": f"s{index}", "text": ""}


def split_atomic_candidates(text: str) -> list[str]:
    """Conservatively split explicit sentence/semicolon boundaries.

    The splitter intentionally avoids guessing that every 'and' introduces a
    new logical claim. It only splits boundaries that are visible in the
    authored response and removes a leading inference marker from each part.
    """
    text = str(text or "").strip()
    if not text:
        return []
    raw = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\s*;\s*", text) if piece.strip()]
    result: list[str] = []
    for piece in raw:
        cleaned = INFERENCE_PREFIX.sub("", piece).strip()
        if cleaned and cleaned[-1] not in ".?!":
            cleaned += "."
        if cleaned:
            result.append(cleaned)
    return result


def _suffix(index: int) -> str:
    # Enough for realistic reasoning steps; after z use aa, ab, ...
    letters = ""
    n = index
    while True:
        letters = chr(ord("a") + (n % 26)) + letters
        n = n // 26 - 1
        if n < 0:
            return letters


def atomize_case(case: dict[str, Any], apply: bool = True) -> dict[str, Any]:
    cloned = copy.deepcopy(case)
    output = cloned.setdefault("llm_output", {})
    raw_steps = output.get("reasoning_steps") or cloned.get("reasoning_steps") or []
    steps = [_coerce_step(item, i) for i, item in enumerate(raw_steps, 1)]

    atomized_steps: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    warnings: list[str] = []
    id_map: dict[str, list[str]] = {}
    changed = 0

    for index, step in enumerate(steps, 1):
        step_id = str(step.get("id") or f"s{index}")
        text = str(step.get("text") or "")
        parts = split_atomic_candidates(text)
        parse_rows = [parse_statement(part) for part in parts]
        safe = len(parts) > 1 and all(row.formula is not None for row in parse_rows)

        if not safe:
            preserved = copy.deepcopy(step)
            preserved["origin_step_id"] = step_id
            preserved["atomization_status"] = "already_atomic" if len(parts) <= 1 else "manual_review_required"
            atomized_steps.append(preserved)
            id_map[step_id] = [step_id]
            if len(parts) > 1:
                errors = [row.error for row in parse_rows if row.formula is None]
                warnings.append(f"{step_id}: split candidates were not all parseable: {errors}")
            mappings.append({
                "origin_step_id": step_id,
                "status": preserved["atomization_status"],
                "original_text": text,
                "new_step_ids": [step_id],
                "parts": parts or [text],
            })
            continue

        changed += 1
        new_ids: list[str] = []
        original_deps = step.get("depends_on") or step.get("dependencies") or []
        if isinstance(original_deps, str):
            original_deps = [x.strip() for x in original_deps.split(",") if x.strip()]
        for part_index, part in enumerate(parts):
            new_id = f"{step_id}{_suffix(part_index)}"
            new_ids.append(new_id)
            new_step: dict[str, Any] = {
                "id": new_id,
                "text": part,
                "origin_step_id": step_id,
                "origin_part_index": part_index + 1,
                "atomization_status": "auto_split",
                "atomization_confidence": "high",
            }
            # Preserve explicitly authored parents only for the first atom.
            # Later atoms are left for Horn/dependency inference rather than
            # inventing a causal edge that the author did not state.
            if part_index == 0 and original_deps:
                new_step["depends_on"] = list(original_deps)
            atomized_steps.append(new_step)
        id_map[step_id] = new_ids
        mappings.append({
            "origin_step_id": step_id,
            "status": "auto_split",
            "original_text": text,
            "new_step_ids": new_ids,
            "parts": parts,
        })

    if apply:
        output["reasoning_steps"] = atomized_steps
        # A final dependency on a compound original step should point to its
        # last atomic claim, which is the closest authored conclusion.
        final_deps = output.get("answer_depends_on") or cloned.get("answer_depends_on")
        if isinstance(final_deps, str):
            final_deps = [x.strip() for x in final_deps.split(",") if x.strip()]
        if isinstance(final_deps, list):
            remapped: list[str] = []
            for dep in final_deps:
                mapped = id_map.get(str(dep), [str(dep)])
                remapped.append(mapped[-1])
            output["answer_depends_on"] = remapped

    return {
        "schema_version": "0.15.0",
        "case_id": str(case.get("id") or "case"),
        "summary": {
            "original_step_count": len(steps),
            "atomized_step_count": len(atomized_steps),
            "compound_steps_split": changed,
            "manual_review_required_count": sum(row["status"] == "manual_review_required" for row in mappings),
            "changed": changed > 0,
            "safe_to_apply": not warnings,
        },
        "mappings": mappings,
        "warnings": warnings,
        "atomized_case": cloned if apply else None,
        "preview_steps": atomized_steps,
    }
