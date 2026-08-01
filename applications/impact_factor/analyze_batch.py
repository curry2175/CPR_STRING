"""Analyze a publication corpus with the root STRING reasoning-graph engine.

corpus 의 논문마다 Typed Claim Graph 를 만들고, graph_metrics 를 평평한 표로 뽑는다.
논문 1편 = OpenAI 호출 1회 이상(긴 Discussion 은 chunk 수만큼).

resume 를 지원한다. 이미 kg/<paper_id>.json 이 있으면 건너뛴다.
중간에 죽어도 다시 돌리면 이어서 한다. 돈이 나가는 작업이라 이게 중요하다.

사용법
    python analyze_batch.py --corpus corpus/collected.jsonl
    python analyze_batch.py --corpus corpus/hcq_discussions.jsonl --effort medium
    python analyze_batch.py --corpus corpus/collected.jsonl --limit 5   :: 먼저 5건만
    python analyze_batch.py --rebuild-table                             :: 호출 없이 표만 다시
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
# Use the single, current STRING core at the repository root.
REPO_ROOT = HERE.parents[1]
DEFAULT_MODULE = REPO_ROOT


def _bootstrap(module_dir: Path) -> None:
    if not (module_dir / "vrg" / "discussion_graph.py").exists():
        raise SystemExit(
            f"STRING core를 찾을 수 없다: {module_dir}\n"
            f"  --module-dir 로 직접 지정하거나 `python lab\\sync_modules.py` 를 먼저 돌려라."
        )
    sys.path.insert(0, str(module_dir))
    # .env 는 모듈 폴더 기준으로 읽히므로 cwd 를 옮긴다.
    import os
    os.chdir(module_dir)


# 표로 뽑을 지표. graph_metrics 의 중첩 구조를 평평하게 만든다.
FLAT_METRICS: list[tuple[str, tuple[str, ...]]] = [
    ("node_count", ("size", "node_count")),
    ("edge_count", ("size", "edge_count")),
    ("root_count", ("size", "root_count")),
    ("leaf_count", ("size", "leaf_count")),
    ("conclusion_count", ("size", "conclusion_count")),
    ("component_count", ("size", "connected_component_count")),
    ("maximum_depth", ("structure", "maximum_depth")),
    ("mean_conclusion_depth", ("structure", "mean_conclusion_depth")),
    ("maximum_width", ("structure", "maximum_width")),
    ("mean_branching_factor", ("structure", "mean_branching_factor")),
    ("edge_node_ratio", ("structure", "edge_node_ratio")),
    ("density", ("structure", "density")),
    ("has_cycle", ("structure", "has_cycle")),
    ("evidence_to_conclusion_ratio", ("reasoning_quality", "evidence_to_conclusion_ratio")),
    ("limitation_to_conclusion_ratio", ("reasoning_quality", "limitation_to_conclusion_ratio")),
    ("conclusions_with_issue_ratio", ("reasoning_quality", "conclusions_with_issue_ratio")),
    ("grounded_edge_ratio", ("reasoning_quality", "grounded_edge_ratio")),
    ("weak_or_model_edge_ratio", ("reasoning_quality", "weak_or_model_edge_ratio")),
    ("source_alignment_rate", ("reasoning_quality", "source_alignment_rate")),
    ("numeric_preservation_rate", ("reasoning_quality", "numeric_preservation_rate")),
    ("unique_relation_type_count", ("reasoning_quality", "unique_relation_type_count")),
    ("score_complexity", ("scores", "complexity")),
    ("score_grounding", ("scores", "grounding")),
    ("score_integrity", ("scores", "integrity")),
    ("score_fidelity", ("scores", "fidelity")),
]

# corpus 에서 그대로 옮길 메타데이터
CARRY = ["paper_id", "journal", "journal_short", "jif", "jif_source", "tier", "year",
         "doi", "pmid", "pmcid", "title", "first_author", "text_source",
         "has_explicit_limitation_heading", "word_count", "char_count",
         "paragraph_count", "cited_by_count", "design", "blinding", "control",
         "n_randomized", "primary_endpoint", "endpoint_type", "powered",
         "early_termination"]

VERIFICATION_LEVELS = ["formal_conflict", "rule_confirmed_unsupported",
                       "structural_methodological_risk", "model_suggested_concern"]


def _dig(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def flatten(record: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """corpus 메타 + graph metrics + 이슈 분포를 한 줄로 만든다."""
    metrics = result.get("graph_metrics") or {}
    nodes = result.get("nodes") or []
    issues = result.get("issues") or []
    words = max(1, int(record.get("word_count") or 1))

    row: dict[str, Any] = {key: record.get(key) for key in CARRY}
    for name, path in FLAT_METRICS:
        row[name] = _dig(metrics, path)

    row["overall_assessment"] = result.get("overall_assessment")
    row["model_overall_assessment"] = result.get("model_overall_assessment")
    row["analysis_mode"] = result.get("analysis_mode")
    row["chunk_count"] = result.get("chunk_count")
    row["api_call_count"] = result.get("api_call_count")

    # --- 길이 보정. Discussion 길이가 교란변수라 원자값만 보면 안 된다. ---
    row["nodes_per_1k_words"] = round(len(nodes) / words * 1000, 2)
    row["issues_per_1k_words"] = round(len(issues) / words * 1000, 2)

    # --- 역할 분포 ---
    role_counts: dict[str, int] = {}
    certainty_counts: dict[str, int] = {}
    assertion_counts: dict[str, int] = {}
    for node in nodes:
        role_counts[str(node.get("role"))] = role_counts.get(str(node.get("role")), 0) + 1
        certainty_counts[str(node.get("certainty"))] = certainty_counts.get(str(node.get("certainty")), 0) + 1
        assertion_counts[str(node.get("assertion_type"))] = assertion_counts.get(str(node.get("assertion_type")), 0) + 1
    total_nodes = max(1, len(nodes))
    for role in ["observation", "evidence", "claim", "mechanism", "limitation",
                 "conclusion", "study_design", "analysis_method"]:
        row[f"role_{role}_n"] = role_counts.get(role, 0)
        row[f"role_{role}_frac"] = round(role_counts.get(role, 0) / total_nodes, 4)
    row["limitation_per_1k_words"] = round(role_counts.get("limitation", 0) / words * 1000, 2)

    # hedging: 단정(establishes/proves/concludes) 대 유보(may/suggests/likely)
    strong = sum(certainty_counts.get(key, 0) for key in ["establishes", "proves", "concludes"])
    hedged = sum(certainty_counts.get(key, 0) for key in ["may", "suggests", "likely", "uncertain"])
    row["certainty_strong_n"] = strong
    row["certainty_hedged_n"] = hedged
    row["hedging_ratio"] = round(hedged / max(1, strong + hedged), 4)

    # 인과 과잉주장의 원재료
    row["assertion_causal_n"] = assertion_counts.get("causal", 0)
    row["assertion_association_n"] = assertion_counts.get("association", 0)
    row["causal_to_association_ratio"] = round(
        assertion_counts.get("causal", 0) / max(1, assertion_counts.get("association", 0)), 4)

    # --- 이슈 분포 ---
    row["issue_total"] = len(issues)
    for level in VERIFICATION_LEVELS:
        row[f"issue_{level}_n"] = sum(1 for issue in issues
                                      if issue.get("verification_level") == level)
    for severity in ["high", "medium", "low"]:
        row[f"issue_severity_{severity}_n"] = sum(1 for issue in issues
                                                  if issue.get("severity") == severity)
    issue_types: dict[str, int] = {}
    for issue in issues:
        key = str(issue.get("issue_type"))
        issue_types[key] = issue_types.get(key, 0) + 1
    row["issue_types_json"] = json.dumps(issue_types, ensure_ascii=False)
    for issue_type in ["causal_overclaim", "scope_overreach", "unsupported_generalization",
                       "surrogate_to_clinical_overreach", "design_claim_mismatch",
                       "evidence_strength_mismatch", "magnitude_inflation",
                       "subgroup_significance_fallacy"]:
        row[f"issue_{issue_type}_n"] = issue_types.get(issue_type, 0)

    usage = result.get("usage") or {}
    row["input_tokens"] = usage.get("input_tokens")
    row["output_tokens"] = usage.get("output_tokens")
    return row


def build_table(kg_dir: Path, corpus: list[dict[str, Any]], out_csv: Path) -> int:
    by_id = {str(record.get("paper_id")): record for record in corpus}
    rows: list[dict[str, Any]] = []
    for path in sorted(kg_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(f"  ! 읽기 실패 {path.name}", file=sys.stderr)
            continue
        record = by_id.get(path.stem)
        if record is None:
            continue
        rows.append(flatten(record, result))
    if not rows:
        print("표로 만들 결과가 없다.", file=sys.stderr)
        return 0
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"표: {len(rows)} 행 x {len(fieldnames)} 열 -> {out_csv}")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(HERE / "corpus" / "collected.jsonl"))
    parser.add_argument("--module-dir", default=str(DEFAULT_MODULE))
    parser.add_argument("--kg-dir", default=str(HERE / "out" / "kg"))
    parser.add_argument("--table", default=str(HERE / "out" / "metrics.csv"))
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    parser.add_argument("--rebuild-table", action="store_true",
                        help="API 호출 없이 기존 kg 로 표만 다시 만든다")
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    kg_dir = Path(args.kg_dir).resolve()
    table_path = Path(args.table).resolve()
    kg_dir.mkdir(parents=True, exist_ok=True)

    corpus = [json.loads(line) for line in
              corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"corpus: {len(corpus)} 건  <- {corpus_path}")

    if args.rebuild_table:
        build_table(kg_dir, corpus, table_path)
        return 0

    _bootstrap(Path(args.module_dir).resolve())
    from vrg.discussion_graph import generate_discussion_graph  # noqa: E402

    pending = [record for record in corpus
               if not (kg_dir / f"{record['paper_id']}.json").exists()]
    if args.limit:
        pending = pending[: args.limit]
    done = len(corpus) - len([record for record in corpus
                              if not (kg_dir / f"{record['paper_id']}.json").exists()])
    print(f"이번에 돌릴 것: {len(pending)} 건 (이미 완료 {done} 건은 건너뛴다)")

    failures: list[dict[str, str]] = []
    started = time.time()
    for index, record in enumerate(pending, start=1):
        paper_id = str(record["paper_id"])
        label = f"[{index}/{len(pending)}] {paper_id}"
        try:
            result = generate_discussion_graph(
                record["discussion_text"],
                model=args.model,
                reasoning_effort=args.effort,
                max_output_tokens=args.max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"paper_id": paper_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{label}  FAIL {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=1)
            continue
        (kg_dir / f"{paper_id}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = result.get("graph_metrics", {})
        print(f"{label}  nodes={len(result.get('nodes') or []):>3} "
              f"issues={len(result.get('issues') or []):>2} "
              f"depth={_dig(metrics, ('structure', 'maximum_depth'))} "
              f"grounding={_dig(metrics, ('scores', 'grounding'))} "
              f"calls={result.get('api_call_count')}")

    elapsed = time.time() - started
    print(f"\n분석 완료 · {len(pending) - len(failures)} 성공 / {len(failures)} 실패 "
          f"· {elapsed / 60:.1f}분")
    if failures:
        path = kg_dir.parent / "analyze_failures.json"
        path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"실패 목록 -> {path}")

    build_table(kg_dir, corpus, table_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
