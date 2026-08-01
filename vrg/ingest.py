from __future__ import annotations

import re
from typing import Any

from .preflight import preflight_case

YES_VALUES = {"yes", "true", "1", "entails", "entailed"}
NO_VALUES = {"no", "false", "0", "contradicts", "contradicted"}


def normalize_yes_no(value: Any, field_name: str = "answer") -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)) and value in {0, 1}:
        return "Yes" if int(value) == 1 else "No"
    text = str(value or "").strip().lower()
    if text in YES_VALUES:
        return "Yes"
    if text in NO_VALUES:
        return "No"
    raise ValueError(f"{field_name} must resolve to strict Yes or No; received: {value!r}")


def _clean_step(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(?:step\s*)?\d+\s*[.)\-:]\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*[-*•]\s*", "", text)
    text = re.sub(r"^(reasoning|analysis|proof)\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _extract_answer_line(text: str) -> tuple[str | None, tuple[int, int] | None]:
    patterns = [
        r"(?im)^\s*(?:final\s+answer|answer|final)\s*[:\-]\s*(yes|no)\b.*$",
        r"(?im)^\s*(yes|no)\s*[.!]?\s*$",
        r"(?i)\b(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+answer\s+is\s+)?(yes|no)\b[.!]?\s*$",
        r"(?i)\bthe\s+answer\s+is\s+(yes|no)\b[.!]?\s*$",
    ]
    best: tuple[int, int, str] | None = None
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            candidate = (match.start(), match.end(), match.group(1).capitalize())
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        return None, None
    return best[2], (best[0], best[1])


def split_llm_response(text: str, *, explicit_answer: Any = None) -> dict[str, Any]:
    raw = str(text or "").strip()
    warnings: list[str] = []
    if not raw:
        raise ValueError("LLM response is empty")

    extracted_answer, answer_span = _extract_answer_line(raw)
    if explicit_answer is not None and str(explicit_answer).strip() != "":
        answer = normalize_yes_no(explicit_answer, "predicted answer")
        if extracted_answer and extracted_answer != answer:
            warnings.append(
                f"Explicit predicted answer ({answer}) differs from response answer ({extracted_answer}); explicit value was used."
            )
    elif extracted_answer:
        answer = extracted_answer
    else:
        raise ValueError("Could not extract a strict Yes/No final answer from the LLM response")

    reasoning_text = raw
    if answer_span:
        reasoning_text = (raw[: answer_span[0]] + "\n" + raw[answer_span[1] :]).strip()

    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    structured = [
        line
        for line in lines
        if re.match(r"^\s*(?:(?:step\s*)?\d+\s*[.)\-:]|[-*•])\s*", line, flags=re.I)
    ]
    if structured:
        candidates = lines
        strategy = "numbered_or_bulleted_lines"
    elif len(lines) > 1:
        candidates = lines
        strategy = "nonempty_lines"
    else:
        # Conservative sentence splitting: only split after terminal punctuation followed by a capital/number.
        candidates = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", reasoning_text)
        strategy = "sentence_split"

    steps: list[dict[str, str]] = []
    ignored: list[str] = []
    for candidate in candidates:
        step = _clean_step(candidate)
        if not step:
            continue
        if re.fullmatch(r"(?i)(reasoning|analysis|proof|step-by-step reasoning)\s*:?", step):
            ignored.append(candidate)
            continue
        if re.search(r"(?i)\b(?:final\s+answer|the\s+answer\s+is)\b", step):
            ignored.append(candidate)
            continue
        steps.append({"id": f"s{len(steps) + 1}", "text": step})

    if not steps:
        warnings.append("No reasoning steps were extracted; the case will contain only the final answer.")
    return {
        "reasoning_steps": steps,
        "answer": answer,
        "extraction": {
            "strategy": strategy,
            "step_count": len(steps),
            "answer_source": "explicit_field" if explicit_answer is not None and str(explicit_answer).strip() else "response_text",
            "ignored_segments": ignored,
            "warnings": warnings,
        },
    }


def _coerce_premises(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+|\n+", value) if piece.strip()]
        return [{"id": f"p{i}", "text": text} for i, text in enumerate(pieces, 1)]
    if not isinstance(value, list):
        raise ValueError("premises/context must be a list or a text block")
    result: list[dict[str, str]] = []
    for index, item in enumerate(value, 1):
        if isinstance(item, str):
            result.append({"id": f"p{index}", "text": item.strip()})
        elif isinstance(item, dict):
            result.append({"id": str(item.get("id") or f"p{index}"), "text": str(item.get("text") or "").strip()})
        else:
            raise ValueError(f"Unsupported premise at index {index}: {type(item).__name__}")
    return result


def build_case_from_raw(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = str(payload.get("case_id") or payload.get("id") or "imported_case")
    premises = _coerce_premises(payload.get("premises", payload.get("context", [])))
    question = str(payload.get("question") or payload.get("query") or "").strip()
    if not question:
        raise ValueError("question is required")
    gold = normalize_yes_no(payload.get("gold_answer", payload.get("gold", payload.get("label"))), "gold answer")
    response = payload.get("llm_response", payload.get("response", payload.get("model_output", "")))
    extracted = split_llm_response(str(response or ""), explicit_answer=payload.get("predicted_answer"))
    case = {
        "id": case_id,
        "premises": premises,
        "question": question,
        "llm_output": {
            "reasoning_steps": extracted["reasoning_steps"],
            "answer": extracted["answer"],
        },
        "gold_answer": gold,
    }
    return {
        "schema_version": "0.15.0",
        "case": case,
        "extraction": extracted["extraction"],
        "preflight": preflight_case(case),
    }
