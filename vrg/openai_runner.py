from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .proofwriter import (
    analyze_proofwriter,
    extract_query_statement,
    normalize_proofwriter_label,
    split_context,
)


DEFAULT_MODEL = "gpt-5.6"
ALLOWED_REASONING_EFFORTS = {"low", "medium", "high"}


class GeneratedReasoningStep(BaseModel):
    id: str = Field(description="Sequential step id such as s1, s2, s3")
    text: str = Field(description="One atomic English fact or derived claim")
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of context premises or earlier reasoning steps directly used for this claim",
    )


class GeneratedProofWriterOutput(BaseModel):
    reasoning_steps: list[GeneratedReasoningStep]
    final_answer: Literal["True", "False", "Unknown"]
    answer_depends_on: list[str] = Field(
        default_factory=list,
        description="Reasoning step IDs that directly support the final classification",
    )


def _load_local_env(root: Path | None = None) -> str | None:
    """Load a small .env file without adding another runtime dependency.

    Existing process environment variables always win. Only OPENAI_API_KEY and
    OPENAI_MODEL are consumed by this application.
    """
    root = root or Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if not env_path.exists():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"\'')
            if key in {"OPENAI_API_KEY", "OPENAI_MODEL"} and value:
                os.environ.setdefault(key, value)
        return str(env_path)
    except OSError:
        return None


def openai_status(root: Path | None = None) -> dict[str, Any]:
    env_path = _load_local_env(root)
    configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    return {
        "configured": configured,
        "key_source": "environment_or_dotenv" if configured else None,
        "dotenv_found": bool(env_path),
        "default_model": os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
        "api": "Responses API",
        "structured_outputs": True,
        "key_exposed_to_browser": False,
    }


def build_proofwriter_prompt(record: dict[str, Any], custom_instruction: str = "") -> dict[str, Any]:
    premises = split_context(record.get("context", record.get("premises")))
    query = extract_query_statement(record.get("question", record.get("query")))

    premise_block = "\n".join(f"{item['id']}: {item['text']}" for item in premises)
    system = (
        "You solve ProofWriter-style formal reasoning tasks using only the supplied facts and rules. "
        "Use open-world semantics with explicit negation: return True only when the queried statement is derivable; "
        "return False only when its explicit opposite is derivable; otherwise return Unknown. "
        "Produce a concise, inspectable stated proof, not private hidden chain-of-thought. "
        "Each reasoning step must contain exactly one atomic English claim. "
        "Use sequential ids s1, s2, ... and list only direct parent ids in depends_on. "
        "Parent ids may be context ids p1, p2, ... or earlier step ids only. "
        "Do not use outside knowledge. Do not copy distractor facts unless they are required. "
        "For a False answer, derive the explicit opposite of the queried statement. "
        "For Unknown, do not invent a proof; answer_depends_on should normally be empty."
    )
    if custom_instruction.strip():
        system += "\nAdditional user instruction: " + custom_instruction.strip()
    user = (
        "Context with stable premise IDs:\n"
        f"{premise_block}\n\n"
        f"Queried statement: {query}\n\n"
        "Return the structured proof and final True/False/Unknown classification."
    )
    return {
        "system": system,
        "user": user,
        "premises": premises,
        "query_statement": query,
        "gold_answer_was_sent": False,
    }


def _parsed_from_response(response: Any) -> GeneratedProofWriterOutput:
    refusal_messages: list[str] = []
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            item_type = getattr(item, "type", None)
            if item_type == "refusal":
                refusal_messages.append(str(getattr(item, "refusal", "Model refused")))
                continue
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                if isinstance(parsed, GeneratedProofWriterOutput):
                    return parsed
                return GeneratedProofWriterOutput.model_validate(parsed)
    output_parsed = getattr(response, "output_parsed", None)
    if output_parsed is not None:
        if isinstance(output_parsed, GeneratedProofWriterOutput):
            return output_parsed
        return GeneratedProofWriterOutput.model_validate(output_parsed)
    if refusal_messages:
        raise ValueError("OpenAI model refusal: " + " | ".join(refusal_messages))
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if output_text:
        try:
            return GeneratedProofWriterOutput.model_validate_json(output_text)
        except Exception as exc:  # noqa: BLE001 - converted to a visible local error
            raise ValueError(f"OpenAI response did not match the required structured schema: {exc}") from exc
    raise ValueError("OpenAI response contained no parsed structured output")


def normalize_generated_output(
    generated: GeneratedProofWriterOutput | dict[str, Any],
    premise_ids: set[str],
) -> dict[str, Any]:
    parsed = generated if isinstance(generated, GeneratedProofWriterOutput) else GeneratedProofWriterOutput.model_validate(generated)
    warnings: list[str] = []
    id_map: dict[str, str] = {}
    raw_steps = list(parsed.reasoning_steps)
    for index, step in enumerate(raw_steps, 1):
        raw_id = str(step.id or f"s{index}").strip()
        normalized_id = f"s{index}"
        if raw_id in id_map:
            warnings.append(f"Duplicate model step id {raw_id!r}; remapped by order to {normalized_id}.")
        id_map[raw_id] = normalized_id

    normalized_steps: list[dict[str, Any]] = []
    available_steps: set[str] = set()
    for index, step in enumerate(raw_steps, 1):
        normalized_id = f"s{index}"
        text = re.sub(r"\s+", " ", str(step.text or "")).strip()
        if text and text[-1] not in ".?!":
            text += "."
        if not text:
            text = "[Empty generated step]."
            warnings.append(f"{normalized_id} had empty text.")
        normalized_parents: list[str] = []
        invalid_parents: list[str] = []
        for raw_parent in step.depends_on:
            parent = str(raw_parent).strip()
            mapped = id_map.get(parent, parent)
            if mapped in premise_ids or mapped in available_steps:
                if mapped not in normalized_parents:
                    normalized_parents.append(mapped)
            else:
                invalid_parents.append(parent)
        if invalid_parents:
            warnings.append(
                f"{normalized_id} referenced unavailable parents {invalid_parents}; they were retained as diagnostics but omitted from executable depends_on."
            )
        normalized_steps.append(
            {
                "id": normalized_id,
                "text": text,
                "depends_on": normalized_parents,
                "model_declared_id": str(step.id),
                "model_declared_dependencies": [str(x) for x in step.depends_on],
                "invalid_model_dependencies": invalid_parents,
                "generation_source": "openai_structured_output",
            }
        )
        available_steps.add(normalized_id)

    answer_dependencies: list[str] = []
    invalid_answer_dependencies: list[str] = []
    for raw_parent in parsed.answer_depends_on:
        parent = str(raw_parent).strip()
        mapped = id_map.get(parent, parent)
        if mapped in available_steps:
            if mapped not in answer_dependencies:
                answer_dependencies.append(mapped)
        else:
            invalid_answer_dependencies.append(parent)
    if invalid_answer_dependencies:
        warnings.append(
            f"Final answer referenced unavailable steps {invalid_answer_dependencies}; they were omitted from executable answer_depends_on."
        )

    return {
        "llm_output": {
            "reasoning_steps": normalized_steps,
            "answer": normalize_proofwriter_label(parsed.final_answer, "generated final answer"),
            "answer_depends_on": answer_dependencies,
        },
        "warnings": warnings,
        "invalid_answer_dependencies": invalid_answer_dependencies,
        "raw_structured_output": parsed.model_dump(),
    }


def format_generated_output_text(llm_output: dict[str, Any]) -> str:
    lines = []
    for index, step in enumerate(llm_output.get("reasoning_steps") or [], 1):
        parents = step.get("depends_on") or []
        suffix = f" [uses: {', '.join(parents)}]" if parents else ""
        lines.append(f"{index}. {step.get('text', '')}{suffix}")
    lines.append(f"Final answer: {llm_output.get('answer', 'Unknown')}")
    return "\n".join(lines)


def _usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    result: dict[str, Any] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result


def generate_proofwriter_output(
    record: dict[str, Any],
    *,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_output_tokens: int = 4000,
    custom_instruction: str = "",
    client: Any = None,
) -> dict[str, Any]:
    _load_local_env()
    model = str(model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:\-]+", model):
        raise ValueError("model contains unsupported characters")
    reasoning_effort = str(reasoning_effort or "low").lower().strip()
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    max_output_tokens = int(max_output_tokens)
    if max_output_tokens < 500 or max_output_tokens > 30000:
        raise ValueError("max_output_tokens must be between 500 and 30000")

    prompt = build_proofwriter_prompt(record, custom_instruction=custom_instruction)
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError(
                "OPENAI_API_KEY is not configured. Copy .env.example to .env and place your key there, then restart RUN_WINDOWS.bat."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ValueError("The openai Python package is missing. Run RUN_WINDOWS.bat again to install requirements.") from exc
        client = OpenAI()

    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ],
        text_format=GeneratedProofWriterOutput,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    parsed = _parsed_from_response(response)
    normalized = normalize_generated_output(parsed, {item["id"] for item in prompt["premises"]})
    return {
        "provider": "openai",
        "api": "Responses API",
        "model_requested": model,
        "model_returned": str(getattr(response, "model", model)),
        "response_id": str(getattr(response, "id", "")),
        "status": str(getattr(response, "status", "completed")),
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "latency_ms": latency_ms,
        "usage": _usage_dict(response),
        "prompt": {
            "prompt_version": "proofwriter_open_world_atomic_v1",
            "system": prompt["system"],
            "user": prompt["user"],
            "query_statement": prompt["query_statement"],
            "premise_count": len(prompt["premises"]),
            "gold_answer_was_sent": False,
            "custom_instruction": custom_instruction.strip(),
        },
        **normalized,
        "display_text": format_generated_output_text(normalized["llm_output"]),
    }


def generate_and_analyze_proofwriter(payload: dict[str, Any], *, client: Any = None) -> dict[str, Any]:
    record = payload.get("record", payload.get("case", payload))
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")
    repetitions = int(payload.get("repetitions", 1))
    if repetitions < 1 or repetitions > 5:
        raise ValueError("repetitions must be between 1 and 5")

    runs: list[dict[str, Any]] = []
    for index in range(repetitions):
        generation = generate_proofwriter_output(
            record,
            model=payload.get("model"),
            reasoning_effort=str(payload.get("reasoning_effort", "low")),
            max_output_tokens=int(payload.get("max_output_tokens", 4000)),
            custom_instruction=str(payload.get("custom_instruction") or ""),
            client=client,
        )
        analysis = analyze_proofwriter(
            {
                "record": record,
                "structured_llm_output": generation["llm_output"],
                "prefer_z3": bool(payload.get("prefer_z3", True)),
                "compute_counterfactuals": bool(payload.get("compute_counterfactuals", False)),
            }
        )
        runs.append(
            {
                "run_index": index + 1,
                "generation": generation,
                "analysis": analysis,
            }
        )

    labels: dict[str, int] = {}
    correct = 0
    context_matches = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for run in runs:
        label = run["analysis"]["classification"]["predicted_label"]
        labels[label] = labels.get(label, 0) + 1
        correct += int(bool(run["analysis"]["classification"]["answer_correct"]))
        context_matches += int(bool(run["analysis"]["classification"]["prediction_matches_context"]))
        usage = run["generation"].get("usage") or {}
        total_input_tokens += int(usage.get("input_tokens") or 0)
        total_output_tokens += int(usage.get("output_tokens") or 0)

    return {
        "schema_version": "0.15.0",
        "record_id": str(record.get("id") or "proofwriter_case"),
        "automation": "OpenAI Responses API -> Structured reasoning -> VRG verification",
        "summary": {
            "run_count": repetitions,
            "label_distribution": labels,
            "gold_correct_count": correct,
            "context_match_count": context_matches,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "gold_answer_sent_to_model": False,
        },
        "runs": runs,
    }
