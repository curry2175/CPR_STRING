"""Math solver: solution text -> six-agent graph -> revision -> re-check.

The graph is an auditor of the public solution, not a replacement for the
solution writer.  Each revision is therefore generated as prose again and is
then compiled by the same six-agent Discussion graph pipeline.
"""
from __future__ import annotations

import re
import argparse
import html
import json
import time
from pathlib import Path
from typing import Any, Callable
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from . import solver
    from .auditor import audit, summarize
    from .questioner import build as build_questions
except ImportError:
    import solver
    from auditor import audit, summarize
    from questioner import build as build_questions
from vrg.discussion_graph import generate_discussion_graph


MATH_SOLVE_PROMPT = (
    "Solve the following mathematics problem rigorously. Write a concise public solution "
    "in 5-10 sentences of plain prose, showing the key equations, deductions, and case "
    "checks. Do not use bullet points or headings. Do not guess. End with exactly one line "
    "of the form ANSWER: <integer>."
)

MATH_REVISION_PROMPT = (
    "You are revising a mathematical solution after a six-agent knowledge-graph audit. "
    "Re-solve the problem where necessary; do not merely edit wording. Address every "
    "structural question, repair invalid algebra, missing cases, unsupported inference, "
    "or incorrect final value. Use only the problem statement and the previous solution. "
    "Write 5-10 sentences of plain prose with equations inline, no headings or bullets. "
    "End with exactly one line: ANSWER: <integer>."
)

ANSWER_CHECK_PROMPT = (
    "Solve the problem independently and audit the candidate solution below. Do not use "
    "any reference answer, benchmark answer, or outside solution. Check the equations, "
    "case coverage, combinatorial counting, and final arithmetic. Return a concise critique "
    "and end with exactly one line of the form INDEPENDENT_ANSWER: <integer>."
)


def extract_answer(text: str) -> int | None:
    matches = re.findall(r"ANSWER\s*:\s*(-?\d+)", str(text or ""), flags=re.I)
    if matches:
        return int(matches[-1])
    match = re.search(r"(-?\d{1,6})\s*$", str(text or "").strip())
    return int(match.group(1)) if match else None


def reasoning_body(text: str) -> str:
    return re.sub(r"\n\s*ANSWER\s*:\s*-?\d+\s*$", "", str(text or ""), flags=re.I | re.S).strip()


def _json_for_html(value: Any) -> str:
    """Serialize data safely for an inline, read-only visualization."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _visualization_html(round_label: str, row: dict[str, Any]) -> str:
    """Create a dependency-free SVG/HTML view of one revision round."""
    graph = row.get("graph") or {}
    nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    edges = [e for e in graph.get("edges", []) if isinstance(e, dict)]
    defects = row.get("defects") or []
    issues = graph.get("issues") or []
    payload = {"round": row.get("round"), "reasoning": row.get("reasoning", ""),
               "answer": row.get("answer"), "nodes": nodes, "edges": edges,
               "issues": issues, "defects": defects, "questions": row.get("questions", []),
               "answer_check": row.get("answer_check", {}),
               "revision": row.get("revision", {})}
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Math self-revision: {html.escape(round_label)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 0; color: #172033; background: #f6f8fb; }}
header {{ padding: 18px 24px; background: #172033; color: white; }}
main {{ display: grid; grid-template-columns: minmax(620px, 2fr) minmax(320px, 1fr); gap: 16px; padding: 16px; }}
section {{ background: white; border: 1px solid #d9e0ea; border-radius: 10px; padding: 14px; }}
h2 {{ margin: 0 0 10px; font-size: 18px; }}
#graph {{ width: 100%; min-height: 560px; border: 1px solid #e1e7ef; border-radius: 8px; background: #fbfcfe; }}
.edge {{ stroke: #8492a6; stroke-width: 1.5; marker-end: url(#arrow); }}
.node {{ stroke: #334155; stroke-width: 1.5; }}
.node.issue {{ stroke: #c2410c; stroke-width: 3; }}
.node-label {{ font-size: 12px; pointer-events: none; }}
.badge {{ display: inline-block; padding: 3px 7px; border-radius: 999px; background: #e7edf6; margin: 2px; font-size: 12px; }}
.high {{ color: #991b1b; font-weight: 700; }} .medium {{ color: #9a3412; }} .low {{ color: #475569; }}
pre {{ white-space: pre-wrap; max-height: 260px; overflow: auto; background: #f8fafc; padding: 10px; border-radius: 6px; }}
li {{ margin: 8px 0; }} .muted {{ color: #64748b; }}
@media (max-width: 950px) {{ main {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<header><strong>Math self-revision — {html.escape(round_label)}</strong>
 <span id="meta" class="muted"></span></header>
<main><section><h2>Knowledge graph</h2><svg id="graph" viewBox="0 0 1000 560" role="img" aria-label="Knowledge graph"></svg>
<p class="muted">Orange outline = node referenced by a detected defect. Hover a node or edge for details.</p></section>
<section><h2>Detected logical issues</h2><div id="issues"></div><h2>Deterministic audit defects</h2><div id="defects"></div><h2>Revision questions</h2><ol id="questions"></ol><h2>Independent answer check</h2><pre id="answer-check"></pre><h2>Solution at this round</h2><pre id="reasoning"></pre><h2>Revision output</h2><pre id="revision"></pre></section></main>
<script>
const data = {_json_for_html(payload)};
const svg = document.getElementById('graph');
const NS = 'http://www.w3.org/2000/svg';
const issueNodes = new Set(data.defects.flatMap(x => x.node_ids || []).concat(data.issues.flatMap(x => x.node_ids || [])));
const cols = Math.max(1, Math.ceil(Math.sqrt(data.nodes.length || 1)));
const pos = new Map(data.nodes.map((n, i) => [n.id, {{x: 90 + (i % cols) * (850 / Math.max(1, cols - 1)), y: 70 + Math.floor(i / cols) * 115}}]));
function add(tag, attrs, parent=svg) {{ const x=document.createElementNS(NS, tag); for (const [k,v] of Object.entries(attrs)) x.setAttribute(k,v); parent.appendChild(x); return x; }}
add('defs', {{}}); const defs=svg.querySelector('defs'); const marker=add('marker', {{id:'arrow',viewBox:'0 0 10 10',refX:'9',refY:'5',markerWidth:'6',markerHeight:'6',orient:'auto-start-reverse'}}, defs); add('path', {{d:'M 0 0 L 10 5 L 0 10 z',fill:'#8492a6'}}, marker);
for (const e of data.edges) {{ const a=pos.get(e.source), b=pos.get(e.target); if (!a || !b) continue; const line=add('line', {{x1:a.x,y1:a.y,x2:b.x,y2:b.y,class:'edge'}}); line.appendChild(Object.assign(document.createElementNS(NS,'title'), {{textContent:`${{e.id||''}}: ${{e.relation||''}}`}})); }}
for (const n of data.nodes) {{ const p=pos.get(n.id); const g=add('g', {{}}); const c=add('circle', {{cx:p.x,cy:p.y,r:31,class:'node'+(issueNodes.has(n.id)?' issue':'')}} ,g); c.appendChild(Object.assign(document.createElementNS(NS,'title'), {{textContent:`${{n.id}} — ${{n.plain_meaning || n.source_text || ''}}`}})); const t=add('text', {{x:p.x,y:p.y+4,'text-anchor':'middle',class:'node-label'}},g); t.textContent=n.id; const sub=add('text', {{x:p.x,y:p.y+50,'text-anchor':'middle',class:'node-label'}},g); sub.textContent=(n.role||'claim').slice(0,16); }}
document.getElementById('meta').textContent=`  answer=${{data.answer ?? 'none'}} · nodes=${{data.nodes.length}} · edges=${{data.edges.length}}`;
function issueText(x) {{ return `<li class="${{x.severity||''}}"><strong>${{x.id||''}} ${{x.title||x.issue_type||x.kind||''}}</strong><br>${{x.explanation||x.summary||''}}<br><span class="muted">nodes: ${{(x.node_ids||[]).join(', ')}}</span></li>`; }}
document.getElementById('issues').innerHTML=data.issues.length ? `<ul>${{data.issues.map(issueText).join('')}}</ul>` : '<p class="muted">No graph issues reported.</p>';
document.getElementById('defects').innerHTML=data.defects.length ? `<ul>${{data.defects.map(issueText).join('')}}</ul>` : '<p class="muted">No deterministic defects.</p>';
document.getElementById('questions').innerHTML=data.questions.length ? data.questions.map(x=>`<li>${{x}}</li>`).join('') : '<li class="muted">No revision requested.</li>';
document.getElementById('answer-check').textContent=data.answer_check.feedback || '(not run or no feedback)';
document.getElementById('reasoning').textContent=data.reasoning;
document.getElementById('revision').textContent=data.revision.text || '(no revision generated)';
</script></body></html>'''


def write_visualizations(result: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Write one graph/error bundle per generated graph and return its manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    rounds = []
    for row in result.get("history", []):
        round_number = int(row.get("round", len(rounds)))
        has_revision = bool(row.get("revision"))
        phase = "before_revision" if round_number == 0 else "after_revision"
        label = f"round_{round_number:02d}_{phase}"
        graph_path = directory / f"{label}.graph.json"
        defects_path = directory / f"{label}.defects.json"
        round_path = directory / f"{label}.round.json"
        html_path = directory / f"{label}.html"
        graph_path.write_text(json.dumps(row.get("graph") or {}, ensure_ascii=False, indent=2), encoding="utf-8")
        defects_path.write_text(json.dumps({"round": round_number, "defects": row.get("defects", []),
                                            "questions": row.get("questions", [])}, ensure_ascii=False, indent=2), encoding="utf-8")
        # Complete provenance: solution -> graph -> defects/questions -> check -> revision.
        round_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        html_path.write_text(_visualization_html(label, row), encoding="utf-8")
        rounds.append({"round": round_number, "phase": phase, "has_revision": has_revision,
                       "html": str(html_path), "graph_json": str(graph_path),
                       "defects_json": str(defects_path), "round_json": str(round_path)})
    manifest = {"directory": str(directory), "rounds": rounds}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _issue_questions(graph: dict[str, Any], defects: list[Any]) -> list[str]:
    questions = build_questions(graph, defects)
    out = [str(question).strip() for question in questions if str(question).strip()]
    for issue in graph.get("issues") or []:
        issue_id = issue.get("id", "issue")
        title = issue.get("title", "Logical issue")
        explanation = issue.get("explanation", "")
        revision = issue.get("suggested_revision", "")
        out.append(f"[{issue_id}] {title}: {explanation} {revision}".strip())
    # Preserve order while preventing duplicate prompts from multiple auditors.
    return list(dict.fromkeys(out))


def _graph_instruction() -> str:
    return (
        "This is a mathematical solution, not a scientific Discussion paragraph. "
        "Represent each equation, definition, case split, implication, and final answer "
        "as a public claim node. Preserve exact mathematical expressions in source_text. "
        "Check arithmetic, quantifiers, case coverage, dependency direction, and whether "
        "the final integer follows from the preceding claims. Treat an unsupported result "
        "or skipped case as a high-priority logical issue."
    )


def solve_with_self_revision(
    problem: str,
    *,
    model: str = "gpt-5.4-nano",
    solve_effort: str = "low",
    graph_effort: str = "low",
    initial_reasoning: str | None = None,
    visualization_dir: Path | None = None,
    max_revision_rounds: int = 8,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Return the final answer, reasoning text, graph, and revision history.

    The loop stops early when the deterministic graph auditor raises no
    questions and the six-agent graph has no public issues.  This is a
    structural stopping condition; without a gold answer the system cannot
    prove mathematical correctness solely from its own output.
    """
    problem = str(problem or "").strip()
    if not problem:
        raise ValueError("problem must not be empty")
    if max_revision_rounds < 0:
        raise ValueError("max_revision_rounds must be non-negative")

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    if initial_reasoning:
        current_text = str(initial_reasoning).strip()
        initial_call = None
        report("initial solution: supplied existing reasoning")
    else:
        report(f"initial solution: calling solver ({model}, effort={solve_effort})")
        initial_call = solver._call(
            f"{problem}\n\n{MATH_SOLVE_PROMPT}",
            model=model, effort=solve_effort, max_output_tokens=3000, client=client,
        )
        current_text = initial_call["text"]
        report(
            f"initial solution: complete ({initial_call.get('latency_s', '?')}s, "
            f"answer={extract_answer(current_text)})"
        )

    history: list[dict[str, Any]] = []
    graph_result: dict[str, Any] | None = None
    final_defects: list[Any] = []
    final_questions: list[str] = []

    revision_round = 0
    revision_count = 0
    while True:
        report(f"round {revision_round}: building six-agent discussion graph")
        graph_result = generate_discussion_graph(
            current_text,
            model=model,
            reasoning_effort=graph_effort,
            max_output_tokens=9000,
            custom_instruction=_graph_instruction(),
            client=client,
            architecture_mode="graph_native_6agents_balanced",
        )
        final_defects = audit(graph_result)
        final_questions = _issue_questions(graph_result, final_defects)
        graph_questions_pending = bool(final_questions)
        candidate_answer = extract_answer(current_text)
        answer_check_call: dict[str, Any] = {}
        answer_check_text = ""
        independent_answer: int | None = None
        answer_check_passed = False
        if not graph_questions_pending:
            answer_check_call = solver._call(
                f"PROBLEM\n{problem}\n\nCANDIDATE SOLUTION\n{reasoning_body(current_text)}\n\n"
                f"{ANSWER_CHECK_PROMPT}",
                model=model, effort=solve_effort, max_output_tokens=2500, client=client,
            )
            answer_check_text = str(answer_check_call.get("text") or "").strip()
            independent_matches = re.findall(
                r"INDEPENDENT_ANSWER\s*:\s*(-?\d+)", answer_check_text, flags=re.I
            )
            independent_answer = int(independent_matches[-1]) if independent_matches else None
            answer_check_passed = (
                independent_answer is not None
                and candidate_answer is not None
                and independent_answer == candidate_answer
            )
        if not graph_questions_pending and not answer_check_passed:
            final_questions.append(
                "[ANSWER_CHECK] The final candidate must pass an independent solve check. "
                "Re-solve the problem from first principles, check every case and the final "
                "arithmetic, and do not preserve the previous answer merely because the "
                "knowledge graph looks structurally plausible."
            )
            final_questions = list(dict.fromkeys(final_questions))
        report(
            f"round {revision_round}: graph complete; "
            f"defects={len(final_defects)}, questions={len(final_questions)}, "
            f"answer_check={'PASS' if answer_check_passed else ('PENDING' if graph_questions_pending else 'FAIL/UNKNOWN')}"
        )
        row: dict[str, Any] = {
            "round": revision_round,
            "model": model,
            "source": "initial_solver" if revision_round == 0 else "revision_solver",
            "reasoning": current_text,
            "answer": extract_answer(current_text),
            "graph": graph_result,
            "defects": [defect.as_dict() for defect in final_defects],
            "defect_summary": summarize(final_defects),
            "questions": final_questions,
            "answer_check": {
                "candidate_answer": candidate_answer,
                "independent_answer": independent_answer,
                "passed": answer_check_passed,
                "feedback": answer_check_text,
                "usage": answer_check_call.get("usage", {}),
                "latency_s": answer_check_call.get("latency_s"),
            },
        }

        if not final_questions:
            row["status"] = "converged"
            history.append(row)
            if visualization_dir is not None:
                write_visualizations({"history": history}, visualization_dir)
            report(f"round {revision_round}: converged")
            break

        if revision_count >= max_revision_rounds:
            row["status"] = "max_revisions_reached"
            history.append(row)
            if visualization_dir is not None:
                write_visualizations({"history": history}, visualization_dir)
            report(f"round {revision_round}: maximum revisions reached")
            break

        report(f"round {revision_round}: revising solution")
        answer_check = ""
        if not answer_check_passed:
            answer_check = (
                f"\n\nANSWER VALIDATION\n{ANSWER_CHECK_PROMPT}\n"
                "The independent checker disagreed with the candidate. Its answer is not "
                "shown here; independently recompute the result and repair the reasoning.\n"
            )
        revised = solver._call(
            f"PROBLEM\n{problem}\n\nPREVIOUS SOLUTION\n{reasoning_body(current_text)}\n\n"
            f"STRUCTURAL QUESTIONS\n" + "\n".join(
                f"{index}. {question}" for index, question in enumerate(final_questions, 1)
            ) + f"\n\n{MATH_REVISION_PROMPT}{answer_check}",
            model=model, effort=solve_effort, max_output_tokens=3500, client=client,
        )
        next_text = revised["text"].strip()
        row["revision"] = {
            "source_round": revision_round,
            "feedback_questions": final_questions,
            "answer_check_feedback": answer_check_text,
            "text": next_text,
            "answer": extract_answer(next_text),
            "usage": revised.get("usage", {}),
            "latency_s": revised.get("latency_s"),
        }
        history.append(row)
        # Persist the graph immediately. This remains useful even when the
        # revision loop runs for an unbounded number of rounds.
        if visualization_dir is not None:
            write_visualizations({"history": history}, visualization_dir)
        if not next_text or next_text == current_text:
            row["status"] = "unchanged"
            if visualization_dir is not None:
                write_visualizations({"history": history}, visualization_dir)
            report(f"round {revision_round}: revision unchanged; stopping")
            break
        report(
            f"round {revision_round}: revision complete "
            f"({revised.get('latency_s', '?')}s, answer={extract_answer(next_text)})"
        )
        current_text = next_text
        revision_round += 1
        revision_count += 1

    assert graph_result is not None
    return {
        "problem": problem,
        "answer": extract_answer(current_text),
        "reasoning": current_text,
        "graph": graph_result,
        "defects": [defect.as_dict() for defect in final_defects],
        "questions": final_questions,
        "history": history,
        "revision_rounds_completed": revision_count,
        "answer_check_passed": bool(history and history[-1].get("answer_check", {}).get("passed")),
        "initial_generation": initial_call,
        "converged": bool(
            history
            and history[-1].get("status") == "converged"
            and bool(history[-1].get("answer_check", {}).get("passed"))
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the math six-agent self-revision pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--problem", help="Math problem text")
    source.add_argument("--file", type=Path, help="UTF-8 text file containing the math problem")
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument(
        "--initial-reasoning-file", type=Path, default=None,
        help="UTF-8 file containing a fixed initial solution, useful for auditing an observed wrong answer",
    )
    parser.add_argument("--solve-effort", default="low", choices=("low", "medium", "high"))
    parser.add_argument("--graph-effort", default="low", choices=("low", "medium", "high"))
    parser.add_argument("--output", type=Path, default=Path("math_self_revision_result.json"))
    parser.add_argument("--max-revision-rounds", type=int, default=8)
    parser.add_argument(
        "--visualization-dir", type=Path, default=None,
        help="Directory for per-round graph/error JSON and standalone HTML visualizations; "
             "defaults to <output-stem>_visualizations",
    )
    args = parser.parse_args()

    def show_progress(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    problem = args.problem if args.problem is not None else args.file.read_text(encoding="utf-8")
    visualization_dir = args.visualization_dir or args.output.with_name(
        f"{args.output.stem}_visualizations"
    )
    show_progress("started")
    initial_reasoning = (
        args.initial_reasoning_file.read_text(encoding="utf-8")
        if args.initial_reasoning_file is not None else None
    )
    result = solve_with_self_revision(
        problem,
        model=args.model,
        solve_effort=args.solve_effort,
        graph_effort=args.graph_effort,
        initial_reasoning=initial_reasoning,
        visualization_dir=visualization_dir,
        max_revision_rounds=args.max_revision_rounds,
        progress=show_progress,
    )
    result["visualizations"] = write_visualizations(result, visualization_dir)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"answer: {result['answer']}")
    print(f"converged: {result['converged']}")
    print(f"revision rounds: {result['revision_rounds_completed']}")
    print(f"visualizations: {visualization_dir.resolve()}")
    print("\nreasoning:\n" + result["reasoning"])
    print(f"\nfull result: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
