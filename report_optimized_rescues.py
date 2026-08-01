from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

RAW_METHOD_CANDIDATES = ("nano_raw_direct", "raw_direct", "small_direct_span")
GRAPH_METHOD_CANDIDATES = ("nano_dual_graph", "dual_graph", "small_claim_graph")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc
    return rows


def method(row: dict[str, Any], candidates: tuple[str, ...]) -> dict[str, Any]:
    methods = row.get("methods") or {}
    for name in candidates:
        value = methods.get(name)
        if isinstance(value, dict):
            return value
    # Fallback by method field.
    for value in methods.values():
        if not isinstance(value, dict):
            continue
        method_name = str(value.get("method") or "")
        if any(token in method_name for token in candidates):
            return value
    return {}


def pred_positive(m: dict[str, Any]) -> bool:
    scores = m.get("scores") or {}
    if "predicted_has_hallucination" in scores:
        return bool(scores.get("predicted_has_hallucination"))
    return bool(m.get("predicted_spans"))


def confusion(rows: list[dict[str, Any]], method_candidates: tuple[str, ...]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for row in rows:
        actual = bool(row.get("gold_has_hallucination") or row.get("gold_labels"))
        predicted = pred_positive(method(row, method_candidates))
        if actual and predicted:
            tp += 1
        elif actual and not predicted:
            fn += 1
        elif not actual and predicted:
            fp += 1
        else:
            tn += 1
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def response_metrics(cm: dict[str, int]) -> dict[str, float]:
    tp, fn, fp, tn = cm["TP"], cm["FN"], cm["FP"], cm["TN"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    return {
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def char_metrics(rows: list[dict[str, Any]], method_candidates: tuple[str, ...]) -> dict[str, float | int]:
    tp = fp = fn = 0
    for row in rows:
        scores = method(row, method_candidates).get("scores") or {}
        tp += int(scores.get("char_tp") or 0)
        fp += int(scores.get("char_fp") or 0)
        fn += int(scores.get("char_fn") or 0)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"TP_chars": tp, "FP_chars": fp, "FN_chars": fn, "precision": precision, "recall": recall, "f1": f1}


def find_latest(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    optimizer_root = root / "outputs" / "alignment_threshold_optimizer"
    gate_path = optimizer_root / "latest_best_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Missing {gate_path}. Run run_threshold_optimizer.bat first.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    original = Path(str(gate.get("source_cases_jsonl") or ""))
    if not original.exists():
        raise FileNotFoundError(f"Original cases file recorded by optimizer does not exist: {original}")
    latest_dir_file = optimizer_root / "LATEST_OUTPUT_DIR.txt"
    if latest_dir_file.exists():
        output_dir = Path(latest_dir_file.read_text(encoding="utf-8").strip())
    else:
        dirs = sorted(optimizer_root.glob("threshold_opt_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not dirs:
            raise FileNotFoundError("No threshold_opt_* output directory found.")
        output_dir = dirs[0]
    optimized = output_dir / "best_reprojected_cases.jsonl"
    if not optimized.exists():
        raise FileNotFoundError(f"Missing optimized cases file: {optimized}")
    return original, optimized, gate


def span_texts(m: dict[str, Any], response: str) -> str:
    parts: list[str] = []
    for span in m.get("predicted_spans") or []:
        try:
            start, end = int(span["start"]), int(span["end"])
        except Exception:
            continue
        text = str(span.get("text") or response[start:end]).replace("\n", " ")
        parts.append(text)
    return " | ".join(parts)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description="Report Raw vs optimized DualGraph confusion matrices and paired rescue cases.")
    parser.add_argument("--root", default=".", help="Project root containing outputs/alignment_threshold_optimizer")
    parser.add_argument("--original-cases", default="", help="Optional original cases.jsonl path")
    parser.add_argument("--optimized-cases", default="", help="Optional best_reprojected_cases.jsonl path")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    if args.original_cases and args.optimized_cases:
        original_path = Path(args.original_cases)
        optimized_path = Path(args.optimized_cases)
        gate_path = root / "outputs" / "alignment_threshold_optimizer" / "latest_best_gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.exists() else {}
    else:
        original_path, optimized_path, gate = find_latest(root)

    original_rows = load_jsonl(original_path)
    optimized_rows = load_jsonl(optimized_path)
    orig_by_id = {str(r.get("case_id")): r for r in original_rows}
    opt_by_id = {str(r.get("case_id")): r for r in optimized_rows}
    common_ids = [cid for cid in orig_by_id if cid in opt_by_id]
    if not common_ids:
        raise RuntimeError("No common case_id values between original and optimized files.")
    originals = [orig_by_id[cid] for cid in common_ids]
    optimized = [opt_by_id[cid] for cid in common_ids]

    raw_cm = confusion(originals, RAW_METHOD_CANDIDATES)
    opt_cm = confusion(optimized, GRAPH_METHOD_CANDIDATES)
    raw_resp = response_metrics(raw_cm)
    opt_resp = response_metrics(opt_cm)
    raw_char = char_metrics(originals, RAW_METHOD_CANDIDATES)
    opt_char = char_metrics(optimized, GRAPH_METHOD_CANDIDATES)

    paired: list[dict[str, Any]] = []
    categories = {"strict_hallucination_rescue": 0, "clean_rescue": 0, "hallucination_regression": 0, "clean_regression": 0}
    for cid in common_ids:
        o = orig_by_id[cid]
        n = opt_by_id[cid]
        actual = bool(o.get("gold_has_hallucination") or o.get("gold_labels"))
        raw_m = method(o, RAW_METHOD_CANDIDATES)
        opt_m = method(n, GRAPH_METHOD_CANDIDATES)
        raw_pos = pred_positive(raw_m)
        opt_pos = pred_positive(opt_m)
        category = "other"
        if actual and not raw_pos and opt_pos:
            category = "strict_hallucination_rescue"
        elif not actual and raw_pos and not opt_pos:
            category = "clean_rescue"
        elif actual and raw_pos and not opt_pos:
            category = "hallucination_regression"
        elif not actual and not raw_pos and opt_pos:
            category = "clean_regression"
        if category != "other":
            categories[category] += 1
            response = str(o.get("response") or "")
            paired.append({
                "category": category,
                "case_id": cid,
                "gold_label": "hallucinated" if actual else "clean",
                "raw_predicted_hallucination": raw_pos,
                "optimized_predicted_hallucination": opt_pos,
                "raw_spans": span_texts(raw_m, response),
                "optimized_spans": span_texts(opt_m, response),
                "response_preview": response[:250].replace("\n", " "),
            })

    out_root = root / "outputs" / "alignment_threshold_optimizer" / "rescue_report"
    out_root.mkdir(parents=True, exist_ok=True)
    report = {
        "original_cases": str(original_path),
        "optimized_cases": str(optimized_path),
        "n": len(common_ids),
        "gate_config": gate.get("gate_config"),
        "raw": {"confusion_matrix": raw_cm, "response_metrics": raw_resp, "character_metrics": raw_char},
        "optimized_dual_graph": {"confusion_matrix": opt_cm, "response_metrics": opt_resp, "character_metrics": opt_char},
        "paired_categories": categories,
    }
    (out_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_root / "paired_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["category", "case_id", "gold_label", "raw_predicted_hallucination", "optimized_predicted_hallucination", "raw_spans", "optimized_spans", "response_preview"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    print("\n================ RAW vs OPTIMIZED DUALGRAPH ================")
    print(f"Cases: {len(common_ids)}")
    print("\nRaw confusion matrix (actual rows, prediction columns)")
    print(f"  Hallucinated: predicted H={raw_cm['TP']:3d}, predicted clean={raw_cm['FN']:3d}")
    print(f"  Clean:        predicted H={raw_cm['FP']:3d}, predicted clean={raw_cm['TN']:3d}")
    print(f"  Response precision={pct(raw_resp['precision'])} sensitivity={pct(raw_resp['recall_sensitivity'])} specificity={pct(raw_resp['specificity'])} F1={pct(raw_resp['f1'])} accuracy={pct(raw_resp['accuracy'])}")
    print(f"  Character precision={pct(float(raw_char['precision']))} recall={pct(float(raw_char['recall']))} F1={pct(float(raw_char['f1']))}")

    print("\nOptimized DualGraph confusion matrix")
    print(f"  Hallucinated: predicted H={opt_cm['TP']:3d}, predicted clean={opt_cm['FN']:3d}")
    print(f"  Clean:        predicted H={opt_cm['FP']:3d}, predicted clean={opt_cm['TN']:3d}")
    print(f"  Response precision={pct(opt_resp['precision'])} sensitivity={pct(opt_resp['recall_sensitivity'])} specificity={pct(opt_resp['specificity'])} F1={pct(opt_resp['f1'])} accuracy={pct(opt_resp['accuracy'])}")
    print(f"  Character precision={pct(float(opt_char['precision']))} recall={pct(float(opt_char['recall']))} F1={pct(float(opt_char['f1']))}")

    print("\nPaired categories")
    for key, value in categories.items():
        print(f"  {key:30s} {value}")
    print(f"\nSaved: {out_root / 'summary.json'}")
    print(f"Saved: {out_root / 'paired_cases.csv'}")
    print("=============================================================")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
