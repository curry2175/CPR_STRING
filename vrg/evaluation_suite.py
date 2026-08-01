from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import time
import urllib.request
import urllib.parse
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .dataset_download import download_proofwriter_600
from .graph_metrics import calculate_graph_metrics
from .hybrid_runner import run_hybrid_proofwriter
from .openai_runner import ALLOWED_REASONING_EFFORTS, DEFAULT_MODEL, _load_local_env, _usage_dict
from .proofwriter import normalize_proofwriter_label


BINARY_DATASETS = ("proofwriter", "legalbench", "pubmedqa")
LEGALBENCH_ARCHIVE = "https://github.com/HazyResearch/legalbench/archive/refs/heads/main.zip"
PUBMEDQA_ARCHIVE = "https://github.com/pubmedqa/pubmedqa/archive/refs/heads/master.zip"
LEGALBENCH_HF_DATASET = "nguha/legalbench"
LEGALBENCH_DATASET_SERVER = "https://datasets-server.huggingface.co"


@dataclass
class BenchmarkCase:
    dataset: str
    case_id: str
    task: str
    instruction: str
    context: str
    question: str
    gold_answer: Literal["Yes", "No"]
    metadata: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PublicReasoningStep(BaseModel):
    id: str = Field(description="Sequential id such as s1, s2")
    text: str = Field(description="One concise public justification claim, not hidden chain-of-thought")
    depends_on: list[str] = Field(default_factory=list, description="Direct parent step ids")
    source_spans: list[str] = Field(default_factory=list, description="Short exact quotes from supplied context that support this step")


class BinaryReasoningOutput(BaseModel):
    reasoning_steps: list[PublicReasoningStep]
    final_answer: Literal["Yes", "No"]
    answer_depends_on: list[str] = Field(default_factory=list)


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _binary_label(value: Any) -> str | None:
    text = _norm(value).lower().strip(" .,:;()[]{}\"'")
    if text in {"yes", "true", "1", "entails", "entailed", "positive"}:
        return "Yes"
    if text in {"no", "false", "0", "contradicts", "contradicted", "negative"}:
        return "No"
    return None


def _download(url: str, target: Path, *, refresh: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not refresh:
        return target
    request = urllib.request.Request(url, headers={"User-Agent": "VRG-Evaluation-Suite/0.26"})
    tmp = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(tmp, target)
    return target


def _extract_archive(archive: Path, target_dir: Path, *, refresh: bool = False) -> Path:
    marker = target_dir / ".complete"
    if marker.exists() and not refresh:
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
    marker.write_text("ok", encoding="utf-8")
    return target_dir


def _find_single_root(target_dir: Path) -> Path:
    roots = [path for path in target_dir.iterdir() if path.is_dir() and path.name != "__MACOSX"]
    return roots[0] if len(roots) == 1 else target_dir


def load_proofwriter_binary_cases(
    data_root: Path,
    *,
    limit: int = 0,
    refresh: bool = False,
) -> list[BenchmarkCase]:
    downloaded = download_proofwriter_600(data_root / "proofwriter", refresh=refresh)
    cases: list[BenchmarkCase] = []
    for line in downloaded.dataset_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        label = normalize_proofwriter_label(record.get("answer"), "ProofWriter label")
        if label == "Unknown":
            continue
        cases.append(BenchmarkCase(
            dataset="proofwriter",
            case_id=str(record.get("id") or f"proofwriter_{len(cases)+1}"),
            task="open_world_binary_entailment",
            instruction=(
                "Using only the supplied facts and rules, answer Yes when the queried statement is derivable and No when its explicit opposite is derivable. "
                "Unknown cases have already been excluded."
            ),
            context=_norm(record.get("context")),
            question=_norm(record.get("question")),
            gold_answer="Yes" if label == "True" else "No",
            metadata={"original_label": label, "source_split": downloaded.split},
            raw_record=record,
        ))
        if limit > 0 and len(cases) >= limit:
            break
    return cases


def _legal_task_description(readme: Path) -> str:
    if not readme.exists():
        return "Answer the legal binary-classification item with Yes or No using only the task text."
    text = readme.read_text(encoding="utf-8-sig", errors="replace")
    lines = []
    capture = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.lower().startswith("## task description"):
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture and line and not line.startswith("!"):
            lines.append(re.sub(r"[`*_]", "", line))
    if not lines:
        match = re.search(r"^###\s+(.+)$", text, flags=re.M)
        if match:
            lines.append(match.group(1).strip())
    description = _norm(" ".join(lines))
    return description[:5000] or "Answer the legal binary-classification item with Yes or No using only the task text."


def _find_tabular_files(task_dir: Path) -> list[Path]:
    # Public GitHub copies sometimes expose only the few-shot train file. The
    # Hugging Face dataset-viewer path below is the primary evaluation source;
    # this scanner is retained as an offline fallback.
    candidates = []
    for pattern in ("**/test.tsv", "**/test.csv", "**/train.tsv", "**/train.csv", "**/data.tsv", "**/data.csv"):
        candidates.extend(task_dir.glob(pattern))
    return sorted(set(path for path in candidates if path.is_file()))


def _read_table(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def _row_text(row: dict[str, Any]) -> str:
    preferred = ["text", "question", "input", "prompt", "fact_pattern", "scenario"]
    parts = []
    for key in preferred:
        if _norm(row.get(key)):
            parts.append(_norm(row[key]))
    if parts:
        return "\n".join(dict.fromkeys(parts))
    ignored = {"answer", "label", "gold", "target", "slice", "id", "index", "idx"}
    return "\n".join(f"{key}: {_norm(value)}" for key, value in row.items() if key.lower() not in ignored and _norm(value))


def _json_request(url: str, *, timeout: int = 180) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "VRG-Evaluation-Suite/0.26", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _cached_json_request(url: str, cache_path: Path, *, refresh: bool = False) -> dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    payload = _json_request(url)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def _legalbench_readme(cache: Path, task: str, *, refresh: bool = False) -> str:
    target = cache / "task_readmes" / task / "README.md"
    url = f"https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks/{urllib.parse.quote(task)}/README.md"
    try:
        _download(url, target, refresh=refresh)
        return _legal_task_description(target)
    except Exception:
        return "Answer the legal binary-classification item with Yes or No using only the task text."


def _legalbench_cases_from_hf(
    cache: Path,
    *,
    limit: int,
    task_names: list[str] | None,
    refresh: bool,
) -> list[BenchmarkCase]:
    query = urllib.parse.urlencode({"dataset": LEGALBENCH_HF_DATASET})
    split_payload = _cached_json_request(
        f"{LEGALBENCH_DATASET_SERVER}/splits?{query}",
        cache / "hf_splits.json",
        refresh=refresh,
    )
    split_rows = split_payload.get("splits") or []
    by_config: dict[str, set[str]] = {}
    for row in split_rows:
        config = _norm(row.get("config"))
        split = _norm(row.get("split"))
        if config and split:
            by_config.setdefault(config, set()).add(split)
    requested = {name.strip() for name in task_names or [] if name.strip()}
    configs = sorted(config for config in by_config if not requested or config in requested)
    if requested:
        missing = sorted(requested - set(configs))
        if missing:
            raise ValueError(f"LegalBench task(s) not found in official dataset splits: {missing}")
    cases: list[BenchmarkCase] = []
    for task in configs:
        # LegalBench's Hugging Face release exposes a few-shot train split and an
        # evaluation/test split. Never use train when test is available.
        splits = by_config[task]
        split = "test" if "test" in splits else ("validation" if "validation" in splits else "train")
        instruction = _legalbench_readme(cache, task, refresh=refresh)
        offset = 0
        while True:
            remaining = 100 if limit <= 0 else min(100, max(0, limit - len(cases)))
            if remaining <= 0:
                return cases
            params = urllib.parse.urlencode({
                "dataset": LEGALBENCH_HF_DATASET,
                "config": task,
                "split": split,
                "offset": offset,
                "length": remaining,
            })
            payload = _cached_json_request(
                f"{LEGALBENCH_DATASET_SERVER}/rows?{params}",
                cache / "hf_rows" / task / f"{split}_{offset}_{remaining}.json",
                refresh=refresh,
            )
            wrapped_rows = payload.get("rows") or []
            if not wrapped_rows:
                break
            for wrapped in wrapped_rows:
                row = wrapped.get("row") if isinstance(wrapped, dict) else None
                if not isinstance(row, dict):
                    continue
                raw_label = row.get("answer", row.get("label", row.get("gold", row.get("target"))))
                raw_label_text = _norm(raw_label).lower().strip(" .,:;()[]{}\"'")
                if raw_label_text not in {"yes", "no", "true", "false"}:
                    continue
                label = _binary_label(raw_label)
                text = _row_text(row)
                if label is None or not text:
                    continue
                row_idx = wrapped.get("row_idx", offset) if isinstance(wrapped, dict) else offset
                cases.append(BenchmarkCase(
                    dataset="legalbench",
                    case_id=f"{task}_{split}_{row_idx}",
                    task=task,
                    instruction=instruction,
                    context=text,
                    question="Provide the correct classification for this legal item: Yes or No.",
                    gold_answer=label,
                    metadata={
                        "source": "huggingface_dataset_viewer",
                        "dataset_repository": LEGALBENCH_HF_DATASET,
                        "source_split": split,
                        "row_idx": row_idx,
                        "raw_label": raw_label,
                    },
                    raw_record=row,
                ))
                if limit > 0 and len(cases) >= limit:
                    return cases
            offset += len(wrapped_rows)
            if len(wrapped_rows) < remaining:
                break
    return cases


def _legalbench_cases_from_github_fallback(
    cache: Path,
    *,
    limit: int,
    task_names: list[str] | None,
    refresh: bool,
) -> list[BenchmarkCase]:
    archive = _download(LEGALBENCH_ARCHIVE, cache / "legalbench-main.zip", refresh=refresh)
    root = _find_single_root(_extract_archive(archive, cache / "repo", refresh=refresh))
    tasks_root = root / "tasks"
    if not tasks_root.exists():
        raise ValueError(f"LegalBench tasks directory was not found under {root}")
    requested = {name.strip() for name in task_names or [] if name.strip()}
    cases: list[BenchmarkCase] = []
    for task_dir in sorted(path for path in tasks_root.iterdir() if path.is_dir()):
        task = task_dir.name
        if requested and task not in requested:
            continue
        instruction = _legal_task_description(task_dir / "README.md")
        tables = _find_tabular_files(task_dir)
        # Prefer public evaluation files. A train file is only a last-resort
        # smoke-test fallback and is marked explicitly in metadata.
        tables.sort(key=lambda path: (0 if path.stem.lower() in {"test", "eval", "validation"} else 1, str(path)))
        for table in tables:
            for row_index, row in enumerate(_read_table(table), 1):
                raw_label = row.get("answer", row.get("label", row.get("gold", row.get("target"))))
                raw_label_text = _norm(raw_label).lower().strip(" .,:;()[]{}\"'")
                if raw_label_text not in {"yes", "no", "true", "false"}:
                    continue
                label = _binary_label(raw_label)
                text = _row_text(row)
                if label is None or not text:
                    continue
                cases.append(BenchmarkCase(
                    dataset="legalbench",
                    case_id=f"{task}_{table.stem}_{row_index}",
                    task=task,
                    instruction=instruction,
                    context=text,
                    question="Provide the correct classification for this legal item: Yes or No.",
                    gold_answer=label,
                    metadata={
                        "source": "github_fallback",
                        "source_file": str(table.relative_to(root)),
                        "source_split": table.stem.lower(),
                        "raw_label": raw_label,
                        "warning": "GitHub fallback may be a few-shot train split when no public test file is present.",
                    },
                    raw_record=row,
                ))
                if limit > 0 and len(cases) >= limit:
                    return cases
    return cases


def load_legalbench_yes_no_cases(
    data_root: Path,
    *,
    limit: int = 0,
    task_names: list[str] | None = None,
    refresh: bool = False,
) -> list[BenchmarkCase]:
    cache = data_root / "legalbench"
    primary_error: Exception | None = None
    try:
        cases = _legalbench_cases_from_hf(cache, limit=limit, task_names=task_names, refresh=refresh)
        if cases:
            return cases
    except Exception as exc:
        primary_error = exc
    try:
        cases = _legalbench_cases_from_github_fallback(cache, limit=limit, task_names=task_names, refresh=refresh)
        if cases:
            return cases
    except Exception as fallback_error:
        raise ValueError(
            f"LegalBench loading failed from the official Hugging Face evaluation release ({primary_error}) "
            f"and the GitHub fallback ({fallback_error})."
        ) from fallback_error
    if primary_error:
        raise ValueError(f"No explicit Yes/No LegalBench rows were found; Hugging Face error: {primary_error}")
    raise ValueError("No explicit Yes/No LegalBench rows were found in the selected tasks.")


def _pubmedqa_context(record: dict[str, Any]) -> str:
    contexts = record.get("CONTEXTS") or record.get("contexts") or []
    labels = record.get("LABELS") or record.get("labels") or []
    if isinstance(contexts, str):
        return _norm(contexts)
    lines = []
    for index, paragraph in enumerate(contexts, 1):
        label = _norm(labels[index - 1]) if isinstance(labels, list) and index - 1 < len(labels) else ""
        prefix = f"{label}: " if label else ""
        lines.append(f"[{index}] {prefix}{_norm(paragraph)}")
    return "\n".join(lines)


def load_pubmedqa_binary_cases(
    data_root: Path,
    *,
    limit: int = 0,
    refresh: bool = False,
) -> list[BenchmarkCase]:
    cache = data_root / "pubmedqa"
    archive = _download(PUBMEDQA_ARCHIVE, cache / "pubmedqa-master.zip", refresh=refresh)
    root = _find_single_root(_extract_archive(archive, cache / "repo", refresh=refresh))
    candidates = [root / "data" / "ori_pqal.json", root / "data" / "pqal_fold0" / "test_set.json"]
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        matches = list(root.glob("**/ori_pqal.json")) + list(root.glob("**/test_set.json"))
        source = matches[0] if matches else None
    if source is None:
        raise ValueError(f"PubMedQA PQA-L JSON was not found under {root}")
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    rows = payload.items() if isinstance(payload, dict) else enumerate(payload)
    cases: list[BenchmarkCase] = []
    for pmid, record in rows:
        if not isinstance(record, dict):
            continue
        label = _binary_label(record.get("final_decision", record.get("answer", record.get("label"))))
        # Strict policy excludes PubMedQA Maybe.
        if label is None:
            continue
        question = _norm(record.get("QUESTION", record.get("question")))
        context = _pubmedqa_context(record)
        if not question or not context:
            continue
        cases.append(BenchmarkCase(
            dataset="pubmedqa",
            case_id=str(pmid),
            task="pqa_labeled_yes_no",
            instruction=(
                "Answer the biomedical research question using only the supplied abstract sections. Return Yes or No. "
                "The original PubMedQA Maybe cases have been excluded."
            ),
            context=context,
            question=question,
            gold_answer=label,
            metadata={"pmid": str(pmid), "source_file": str(source.relative_to(root))},
            raw_record=record,
        ))
        if limit > 0 and len(cases) >= limit:
            break
    return cases


def build_binary_prompt(case: BenchmarkCase) -> dict[str, str]:
    system = (
        "You are being evaluated on a strict binary reasoning benchmark. Use only the supplied task instruction and input. "
        "Return Yes or No. Produce a short inspectable public justification graph, not private hidden chain-of-thought. "
        "Each reasoning step must be one atomic claim, use sequential ids s1, s2, ..., and list direct parent step ids only. "
        "source_spans must be short exact quotes copied from the supplied input; leave them empty when no exact quote supports the step. "
        "Do not invent statutes, medical facts, citations, assumptions, or missing evidence."
    )
    user = (
        f"Dataset: {case.dataset}\nTask: {case.task}\n\n"
        f"Task instruction:\n{case.instruction}\n\n"
        f"Supplied input/context:\n{case.context}\n\n"
        f"Question:\n{case.question}\n\n"
        "Return the structured public justification and final Yes/No answer."
    )
    return {"system": system, "user": user}


def _parse_binary_response(response: Any) -> BinaryReasoningOutput:
    for output in getattr(response, "output", []) or []:
        if getattr(output, "type", None) != "message":
            continue
        for item in getattr(output, "content", []) or []:
            parsed = getattr(item, "parsed", None)
            if parsed is not None:
                return parsed if isinstance(parsed, BinaryReasoningOutput) else BinaryReasoningOutput.model_validate(parsed)
    parsed = getattr(response, "output_parsed", None)
    if parsed is not None:
        return parsed if isinstance(parsed, BinaryReasoningOutput) else BinaryReasoningOutput.model_validate(parsed)
    text = _norm(getattr(response, "output_text", ""))
    if text:
        return BinaryReasoningOutput.model_validate_json(text)
    raise ValueError("OpenAI response contained no parsed binary reasoning output")


def _token_set(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _quote_is_exact(quote: str, context: str) -> bool:
    return bool(_norm(quote)) and _norm(quote).lower() in _norm(context).lower()


def _reasoning_graph(parsed: BinaryReasoningOutput, context: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    known: set[str] = set()
    declared_to_internal: dict[str, str] = {}
    exact_quotes = 0
    quote_count = 0
    invalid_dependencies: list[dict[str, str]] = []
    context_tokens = _token_set(context)
    grounding_scores: list[float] = []
    root_node_ids: list[str] = []
    grounded_root_ids: list[str] = []

    # Normalize arbitrary model ids to deterministic s1, s2, ... while keeping
    # dependencies resolvable through the model-declared ids.
    for index, step in enumerate(parsed.reasoning_steps, 1):
        internal_id = f"s{index}"
        declared_id = _norm(step.id) or internal_id
        if declared_id not in declared_to_internal:
            declared_to_internal[declared_id] = internal_id
        declared_to_internal.setdefault(internal_id, internal_id)

    for index, step in enumerate(parsed.reasoning_steps, 1):
        node_id = f"s{index}"
        declared_id = _norm(step.id) or node_id
        text = _norm(step.text)
        quotes = [_norm(q) for q in step.source_spans if _norm(q)]
        exact = [q for q in quotes if _quote_is_exact(q, context)]
        quote_count += len(quotes)
        exact_quotes += len(exact)
        step_tokens = _token_set(text)
        overlap = len(step_tokens & context_tokens) / max(1, len(step_tokens))
        grounding_scores.append(overlap)
        declared_parents = [_norm(parent) for parent in step.depends_on if _norm(parent)]
        resolved_parents: list[str] = []
        for parent_id in declared_parents:
            mapped = declared_to_internal.get(parent_id)
            if mapped in known:
                resolved_parents.append(mapped)
                edges.append({"id": f"e{len(edges)+1}", "source": mapped, "target": node_id, "relation": "supports", "confidence": 0.8})
            else:
                invalid_dependencies.append({"node_id": node_id, "parent_id": parent_id})
        is_root = not resolved_parents
        if is_root:
            root_node_ids.append(node_id)
            if exact:
                grounded_root_ids.append(node_id)
        nodes.append({
            "id": node_id,
            "model_declared_id": declared_id,
            "kind": "reasoning",
            "role": "evidence" if exact else "claim",
            "text": text,
            "plain_meaning": text,
            "source_text": exact[0] if exact else (quotes[0] if quotes else ""),
            "source_fidelity_status": "exact" if exact else ("unmatched" if quotes else "missing"),
            "source_span_exact": bool(exact),
            "confidence": round(min(1.0, 0.45 + overlap * 0.55), 3),
            "numeric_mentions": re.findall(r"(?<!\w)(?:\d+(?:\.\d+)?%?|p\s*[<=>]\s*0?\.\d+)(?!\w)", text, re.I),
            "inferred_details": [],
            "declared_dependencies": declared_parents,
            "resolved_dependencies": resolved_parents,
        })
        known.add(node_id)

    answer_id = "final"
    nodes.append({
        "id": answer_id, "kind": "answer", "role": "conclusion",
        "text": parsed.final_answer, "plain_meaning": f"Final answer: {parsed.final_answer}",
        "source_text": "", "source_fidelity_status": "not_checked", "source_span_exact": False,
        "confidence": 1.0, "numeric_mentions": [], "inferred_details": [],
    })
    valid_answer_parents: list[str] = []
    for parent in parsed.answer_depends_on:
        parent_id = _norm(parent)
        mapped = declared_to_internal.get(parent_id, parent_id if parent_id in known else None)
        if mapped in known:
            valid_answer_parents.append(mapped)
            edges.append({"id": f"e{len(edges)+1}", "source": mapped, "target": answer_id, "relation": "supports", "confidence": 0.9})
        else:
            invalid_dependencies.append({"node_id": answer_id, "parent_id": parent_id})

    missing_root_grounding = sorted(set(root_node_ids) - set(grounded_root_ids))
    structural_issues: list[dict[str, Any]] = []
    if invalid_dependencies:
        structural_issues.append({
            "verification_level": "formal_conflict",
            "node_ids": sorted({item["node_id"] for item in invalid_dependencies}),
            "issue_type": "invalid_dependency_reference",
        })
    if not valid_answer_parents:
        structural_issues.append({
            "verification_level": "rule_confirmed_unsupported",
            "node_ids": [answer_id],
            "issue_type": "answer_without_declared_support",
        })
    if missing_root_grounding:
        structural_issues.append({
            "verification_level": "rule_confirmed_unsupported",
            "node_ids": missing_root_grounding,
            "issue_type": "root_reasoning_without_exact_source_span",
        })
    graph_pass = not invalid_dependencies and bool(valid_answer_parents) and not missing_root_grounding
    diagnostics = {
        "quote_count": quote_count,
        "exact_quote_count": exact_quotes,
        "exact_quote_rate": round(exact_quotes / max(1, quote_count), 3),
        "mean_lexical_grounding": round(statistics.mean(grounding_scores), 3) if grounding_scores else 0.0,
        "invalid_dependencies": invalid_dependencies,
        "dependency_integrity": len(invalid_dependencies) == 0,
        "root_node_ids": root_node_ids,
        "grounded_root_ids": grounded_root_ids,
        "missing_root_grounding": missing_root_grounding,
        "valid_answer_parents": valid_answer_parents,
        "answer_grounded": bool(valid_answer_parents),
        "graph_pass": graph_pass,
        "graph_pass_type": "public_reasoning_structural",
        "structural_issues": structural_issues,
    }
    return nodes, edges, diagnostics


def evaluate_general_binary_case(
    case: BenchmarkCase,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    client: Any,
) -> dict[str, Any]:
    prompt = build_binary_prompt(case)
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        store=False,
        input=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}],
        text_format=BinaryReasoningOutput,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    parsed = _parse_binary_response(response)
    nodes, edges, diagnostics = _reasoning_graph(parsed, case.instruction + "\n" + case.context)
    graph_metrics = calculate_graph_metrics(nodes, edges, issues=diagnostics["structural_issues"])
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "task": case.task,
        "gold_answer": case.gold_answer,
        "predicted_answer": parsed.final_answer,
        "answer_correct": parsed.final_answer == case.gold_answer,
        "graph_pass": diagnostics["graph_pass"],
        "graph_pass_type": diagnostics["graph_pass_type"],
        "reasoning_steps": [step.model_dump() for step in parsed.reasoning_steps],
        "answer_depends_on": parsed.answer_depends_on,
        "graph": {"nodes": nodes, "edges": edges},
        "graph_metrics": graph_metrics,
        "diagnostics": diagnostics,
        "usage": _usage_dict(response),
        "latency_ms": latency_ms,
        "response_id": str(getattr(response, "id", "")),
        "metadata": case.metadata,
    }


def evaluate_proofwriter_case(
    case: BenchmarkCase,
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    max_repair_iterations: int,
    client: Any,
) -> dict[str, Any]:
    raw = dict(case.raw_record)
    result = run_hybrid_proofwriter({
        "record": raw,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "max_repair_iterations": max_repair_iterations,
        "repair_mode": "blind",
        "use_llm_formalizer": True,
        "use_premise_grounder": True,
        "allow_external_premises": False,
        "prefer_z3": True,
    }, client=client)
    graph = result.get("final_universal_graph") or {}
    predicted = "Yes" if result["summary"].get("final_answer") == "True" else "No"
    graph_metrics = calculate_graph_metrics(graph.get("nodes") or [], graph.get("edges") or [], issues=[])
    return {
        "dataset": case.dataset,
        "case_id": case.case_id,
        "task": case.task,
        "gold_answer": case.gold_answer,
        "predicted_answer": predicted,
        "answer_correct": predicted == case.gold_answer,
        "graph_pass": bool(result["summary"].get("final_pass")),
        "graph_pass_type": "formal_verifier",
        "context_match": bool(result["summary"].get("final_context_match")),
        "repair_count": int(result["summary"].get("repair_count") or 0),
        "graph_metrics": graph_metrics,
        "usage": result["summary"].get("total_usage") or {},
        "api_call_count": result["summary"].get("api_call_count") or 0,
        "result": result,
        "metadata": case.metadata,
    }


def _macro_f1(rows: list[dict[str, Any]]) -> float:
    f1s = []
    for label in ("Yes", "No"):
        tp = sum(1 for row in rows if row.get("gold_answer") == label and row.get("predicted_answer") == label)
        fp = sum(1 for row in rows if row.get("gold_answer") != label and row.get("predicted_answer") == label)
        fn = sum(1 for row in rows if row.get("gold_answer") == label and row.get("predicted_answer") != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(statistics.mean(f1s) * 100, 2)


def summarize_results(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, dict[str, Any]] = {}
    for dataset in BINARY_DATASETS:
        subset = [row for row in rows if row.get("dataset") == dataset]
        if not subset:
            continue
        correct = sum(bool(row.get("answer_correct")) for row in subset)
        input_tokens = sum(int((row.get("usage") or {}).get("input_tokens") or 0) for row in subset)
        output_tokens = sum(int((row.get("usage") or {}).get("output_tokens") or 0) for row in subset)
        metrics = [row.get("graph_metrics", {}).get("structure", {}) for row in subset]
        score_rows = [row.get("graph_metrics", {}).get("scores", {}) for row in subset]
        task_breakdown: dict[str, Any] = {}
        for task in sorted({str(row.get("task") or "unknown") for row in subset}):
            task_rows = [row for row in subset if str(row.get("task") or "unknown") == task]
            task_breakdown[task] = {
                "case_count": len(task_rows),
                "accuracy_percent": round(sum(bool(row.get("answer_correct")) for row in task_rows) / len(task_rows) * 100, 2),
                "macro_f1_percent": _macro_f1(task_rows),
                "graph_pass_percent": round(sum(bool(row.get("graph_pass")) for row in task_rows) / len(task_rows) * 100, 2),
            }
        datasets[dataset] = {
            "case_count": len(subset),
            "accuracy_percent": round(correct / len(subset) * 100, 2),
            "macro_f1_percent": _macro_f1(subset),
            "yes_count": sum(1 for row in subset if row.get("gold_answer") == "Yes"),
            "no_count": sum(1 for row in subset if row.get("gold_answer") == "No"),
            "prediction_yes_count": sum(1 for row in subset if row.get("predicted_answer") == "Yes"),
            "mean_graph_depth": round(statistics.mean(float(m.get("maximum_depth") or 0) for m in metrics), 3),
            "mean_graph_width": round(statistics.mean(float(m.get("maximum_width") or 0) for m in metrics), 3),
            "mean_complexity_score": round(statistics.mean(float(score.get("complexity") or 0) for score in score_rows), 2),
            "mean_grounding_score": round(statistics.mean(float(score.get("grounding") or 0) for score in score_rows), 2),
            "mean_integrity_score": round(statistics.mean(float(score.get("integrity") or 0) for score in score_rows), 2),
            "mean_fidelity_score": round(statistics.mean(float(score.get("fidelity") or 0) for score in score_rows), 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "graph_pass_percent": round(sum(bool(row.get("graph_pass")) for row in subset) / len(subset) * 100, 2),
            "graph_pass_type": "formal_verifier" if dataset == "proofwriter" else "public_reasoning_structural",
            "task_count": len(task_breakdown),
            "tasks": task_breakdown,
        }
    total = len(rows)
    correct = sum(bool(row.get("answer_correct")) for row in rows)
    return {
        "completed_cases": total,
        "failed_cases": len(failures),
        "overall_accuracy_percent": round(correct / total * 100, 2) if total else 0.0,
        "datasets": datasets,
        "strict_binary_policy": {
            "proofwriter_unknown_excluded": True,
            "pubmedqa_maybe_excluded": True,
            "legalbench_non_yes_no_labels_excluded": True,
        },
    }


def _html_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    rows = []
    task_sections = []
    for name, stats in summary.get("datasets", {}).items():
        rows.append(
            f"<tr><td>{name}</td><td>{stats['case_count']}</td><td>{stats['accuracy_percent']}%</td>"
            f"<td>{stats['macro_f1_percent']}%</td><td>{stats['graph_pass_percent']}%</td>"
            f"<td>{stats['mean_graph_depth']}</td><td>{stats['mean_graph_width']}</td>"
            f"<td>{stats['mean_complexity_score']}</td><td>{stats['mean_grounding_score']}</td>"
            f"<td>{stats['mean_integrity_score']}</td><td>{stats['mean_fidelity_score']}</td>"
            f"<td>{stats['total_tokens']}</td></tr>"
        )
        tasks = stats.get("tasks") or {}
        if tasks:
            task_rows = "".join(
                f"<tr><td>{task}</td><td>{task_stats['case_count']}</td>"
                f"<td>{task_stats['accuracy_percent']}%</td><td>{task_stats['macro_f1_percent']}%</td>"
                f"<td>{task_stats['graph_pass_percent']}%</td></tr>"
                for task, task_stats in tasks.items()
            )
            task_sections.append(
                f"<h3>{name} task breakdown</h3>"
                "<table><thead><tr><th>Task</th><th>N</th><th>Accuracy</th><th>Macro-F1</th><th>Graph PASS</th></tr></thead>"
                f"<tbody>{task_rows}</tbody></table>"
            )
    return f"""<!doctype html><meta charset='utf-8'><title>VRG Three-Dataset Evaluation</title>
<style>body{{font-family:Arial;margin:32px;color:#172033}}table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid #dbe2ec;padding:8px;text-align:left}}th{{background:#f1f5f9}}.card{{border:1px solid #dbe2ec;border-radius:12px;padding:16px;margin:14px 0}}.note{{color:#475569;line-height:1.5}}</style>
<h1>VRG Three-Dataset Evaluation</h1><div class='card'><b>Run:</b> {result['run_id']}<br><b>Model:</b> {result['settings']['model']}<br><b>Overall accuracy:</b> {summary['overall_accuracy_percent']}%<br><b>Completed:</b> {summary['completed_cases']} · <b>Failed:</b> {summary['failed_cases']}</div>
<table><thead><tr><th>Dataset</th><th>N</th><th>Accuracy</th><th>Macro-F1</th><th>Graph PASS</th><th>Depth</th><th>Width</th><th>Complexity</th><th>Grounding</th><th>Integrity</th><th>Fidelity</th><th>Tokens</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class='note'>Depth and width describe graph structure, not correctness. Complexity, grounding, integrity, and fidelity are reported separately. ProofWriter Graph PASS uses the formal verifier; LegalBench and PubMedQA use public-reasoning structural validation.</div>
{''.join(task_sections)}
<p>Policy: ProofWriter Unknown and PubMedQA Maybe cases were excluded; LegalBench includes only explicit Yes/No labels.</p>"""


def run_three_dataset_evaluation(
    *,
    output_root: Path,
    data_root: Path,
    datasets: list[str] | None = None,
    limit_per_dataset: int = 20,
    legal_tasks: list[str] | None = None,
    model: str | None = None,
    reasoning_effort: str = "low",
    max_output_tokens: int = 3500,
    max_repair_iterations: int = 0,
    refresh_datasets: bool = False,
    client: Any = None,
) -> dict[str, Any]:
    _load_local_env()
    selected = datasets or list(BINARY_DATASETS)
    unknown = sorted(set(selected) - set(BINARY_DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    model = _norm(model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)
    reasoning_effort = _norm(reasoning_effort).lower() or "low"
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    if client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is not configured")
        from openai import OpenAI
        client = OpenAI()
    data_root.mkdir(parents=True, exist_ok=True)
    loaders = {
        "proofwriter": lambda: load_proofwriter_binary_cases(data_root, limit=limit_per_dataset, refresh=refresh_datasets),
        "legalbench": lambda: load_legalbench_yes_no_cases(data_root, limit=limit_per_dataset, task_names=legal_tasks, refresh=refresh_datasets),
        "pubmedqa": lambda: load_pubmedqa_binary_cases(data_root, limit=limit_per_dataset, refresh=refresh_datasets),
    }
    all_cases: list[BenchmarkCase] = []
    dataset_counts = {}
    for dataset in selected:
        loaded = loaders[dataset]()
        all_cases.extend(loaded)
        dataset_counts[dataset] = len(loaded)

    run_id = f"three_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "schema_version": "0.27.0",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": selected,
        "limit_per_dataset": limit_per_dataset,
        "legal_tasks": legal_tasks or [],
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "max_repair_iterations": max_repair_iterations,
        "refresh_datasets": refresh_datasets,
        "dataset_case_counts": dataset_counts,
        "strict_binary_policy": {
            "proofwriter_unknown_excluded": True,
            "pubmedqa_maybe_excluded": True,
            "legalbench_non_yes_no_labels_excluded": True,
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    checkpoint = run_dir / "cases.jsonl"
    failures_path = run_dir / "failures.jsonl"
    checkpoint.touch()
    failures_path.touch()
    for index, case in enumerate(all_cases, 1):
        try:
            if case.dataset == "proofwriter":
                row = evaluate_proofwriter_case(
                    case, model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens, max_repair_iterations=max_repair_iterations, client=client,
                )
            else:
                row = evaluate_general_binary_case(
                    case, model=model, reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens, client=client,
                )
            row["case_index"] = index
            rows.append(row)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:
            failure = {"case_index": index, "dataset": case.dataset, "case_id": case.case_id, "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    result = {
        "schema_version": "0.27.0",
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "datasets": selected,
            "limit_per_dataset": limit_per_dataset,
            "legal_tasks": legal_tasks or [],
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "max_repair_iterations": max_repair_iterations,
            "dataset_case_counts": dataset_counts,
        },
        "summary": summarize_results(rows, failures),
        "cases": rows,
        "failures": failures,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_html_report(result), encoding="utf-8")
    (output_root / "latest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def list_evaluation_runs(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((p for p in output_root.glob("three_dataset_*") if p.is_dir()), reverse=True):
        summary_path = path / "summary.json"
        if not summary_path.exists():
            continue
        try:
            result = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append({
                "run_id": result.get("run_id", path.name),
                "created_at": result.get("created_at"),
                "overall_accuracy_percent": (result.get("summary") or {}).get("overall_accuracy_percent"),
                "completed_cases": (result.get("summary") or {}).get("completed_cases"),
                "datasets": (result.get("settings") or {}).get("datasets"),
            })
        except Exception:
            continue
    return rows
