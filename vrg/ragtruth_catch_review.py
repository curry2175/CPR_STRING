from __future__ import annotations

import html
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .openai_runner import ALLOWED_REASONING_EFFORTS, _load_local_env
from .ragtruth_dual_graph import (
    ResponseClaimGraphOutput,
    SourceEvidenceGraphOutput,
    _predictions_from_alignment,
)
from .ragtruth_six_agent_dual_graph import (
    _empty_cache,
    _load_cache,
    _save_cache,
    _run_cross_graph_pipeline,
    _validate_graph_six_agents,
)


SCHEMA_VERSION = "0.48.0-catch-six-agent-visual-review"


def _latest_candidate_run(output_root: Path) -> Path:
    candidates: list[Path] = []
    if output_root.exists():
        for path in output_root.iterdir():
            if path.is_dir() and (path / "catch_candidates" / "index.json").exists():
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"No RAGQA run with catch_candidates was found under {output_root}. Run run_ragqa.bat first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_catch_index(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "catch_candidates" / "index.json"
    if not path.exists():
        raise FileNotFoundError(f"Catch index not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError(f"Invalid catch index: {path}")
    return payload


def resolve_run_dir(output_root: Path, run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        return run_dir.resolve()
    return _latest_candidate_run(output_root.resolve())


def list_catch_candidates(run_dir: Path) -> list[dict[str, Any]]:
    return list(load_catch_index(run_dir).get("candidates") or [])


def load_catch_candidate(run_dir: Path, case_id: str) -> dict[str, Any]:
    path = run_dir / "catch_candidates" / "cases" / f"{case_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Catch case not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid catch case: {path}")
    return payload


def _overlaps(span: dict[str, Any], other: dict[str, Any]) -> bool:
    try:
        return max(int(span["start"]), int(other["start"])) < min(int(span["end"]), int(other["end"]))
    except Exception:
        return False


def _node_span(node: dict[str, Any], response: str) -> dict[str, Any] | None:
    start, end = node.get("start"), node.get("end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(response):
        return {"start": start, "end": end, "text": response[start:end]}
    text = str(node.get("resolved_text") or node.get("text") or "")
    if text:
        start = response.find(text)
        if start >= 0:
            return {"start": start, "end": start + len(text), "text": text}
    return None


def _node_comparison_rows(candidate: dict[str, Any], six_result: dict[str, Any]) -> list[dict[str, Any]]:
    response = str(candidate.get("response") or "")
    gold = candidate.get("gold_labels") or []
    dual_spans = ((candidate.get("balanced_dual_graph") or {}).get("predicted_spans") or [])
    balanced = candidate.get("balanced_dual_graph") or {}
    original_response_graph = balanced.get("response_graph") or {}
    original_alignments = balanced.get("alignments") or []
    alignment_by_id = {str(row.get("response_node_id")): row for row in original_alignments}
    alignment_by_claim = {}
    for node in original_response_graph.get("nodes") or []:
        key = " ".join(str(node.get("normalized_claim") or node.get("text") or "").lower().split())
        if key and str(node.get("id")) in alignment_by_id:
            alignment_by_claim[key] = alignment_by_id[str(node.get("id"))]

    response_graph = six_result.get("validated_response_graph") or original_response_graph
    source_graph = six_result.get("validated_source_graph") or balanced.get("source_graph") or {}
    source_by_id = {str(node.get("id")): node for node in source_graph.get("nodes") or []}
    cross_verdict_by_id = {
        str(row.get("response_node_id")): row
        for row in ((six_result.get("cross_graph") or {}).get("final_verdicts") or [])
    }
    rows: list[dict[str, Any]] = []
    for node in response_graph.get("nodes") or []:
        node_id = str(node.get("id") or "")
        span = _node_span(node, response)
        claim_key = " ".join(str(node.get("normalized_claim") or node.get("text") or "").lower().split())
        alignment = alignment_by_id.get(node_id) or alignment_by_claim.get(claim_key) or {}
        six_verdict = cross_verdict_by_id.get(node_id) or {}
        source_ids = [str(x) for x in (six_verdict.get("source_node_ids") or alignment.get("source_node_ids") or [])]
        sources = [source_by_id[x] for x in source_ids if x in source_by_id]
        rows.append({
            "node_id": node_id,
            "sentence_id": node.get("sentence_id"),
            "text": node.get("resolved_text") or node.get("text"),
            "normalized_claim": node.get("normalized_claim") or node.get("text"),
            "span": span,
            "overlaps_gold": bool(span and any(_overlaps(span, x) for x in gold)),
            "overlaps_dual_prediction": bool(span and any(_overlaps(span, x) for x in dual_spans)),
            "balanced_relation": alignment.get("relation"),
            "balanced_confidence": alignment.get("confidence"),
            "balanced_explanation": alignment.get("explanation"),
            "source_node_ids": source_ids,
            "source_nodes": sources,
            "six_agent_verdict": six_verdict.get("verdict"),
            "six_agent_confidence": six_verdict.get("confidence"),
            "six_agent_explanation": six_verdict.get("explanation"),
            "six_agent_changed_dimensions": six_verdict.get("changed_dimensions") or [],
        })
    return rows

def _span_text(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return "none"
    return "<br>".join(
        f"<code>[{html.escape(str(x.get('start')))}:{html.escape(str(x.get('end')))}]</code> "
        f"{html.escape(str(x.get('text') or ''))}"
        for x in spans
    )


def _pretty_json(value: Any) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def _safe_script_json(value: Any) -> str:
    """Serialize JSON for embedding inside a script tag without allowing tag termination."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _report_html(result: dict[str, Any]) -> str:
    candidate = result["candidate"]
    rows = result["node_comparisons"]
    balanced = candidate.get("balanced_dual_graph") or {}
    direct_mode = str(candidate.get("catch_reason") or "") == "direct_case_id_review"
    report_heading = "Direct RAGTruth case · 6-Agent graph review" if direct_mode else "Raw-miss / DualGraph-catch · 6-Agent review"
    report_note = (
        "This case was loaded directly by id. The v043 Balanced Source/Response graphs and v046 alignment were created first, "
        "then the selected case was reviewed by the six-agent graph pipeline."
        if direct_mode else
        "The benchmark result below is the saved v046 Balanced Dual-Graph result. Six-Agent review was run only after this case was selected; it did not affect the full benchmark score."
    )
    row_html: list[str] = []
    for row in rows:
        source_html = "<br>".join(
            f"<b>{html.escape(str(node.get('id') or ''))}</b>: {html.escape(str(node.get('text') or ''))}"
            for node in row.get("source_nodes") or []
        ) or "—"
        classes = []
        if row.get("overlaps_gold"):
            classes.append("gold")
        if row.get("overlaps_dual_prediction"):
            classes.append("pred")
        row_html.append(
            f"<tr class='{' '.join(classes)}'>"
            f"<td><b>{html.escape(row['node_id'])}</b><br><small>{html.escape(str(row.get('sentence_id') or ''))}</small></td>"
            f"<td>{html.escape(str(row.get('text') or ''))}<hr><small>{html.escape(str(row.get('normalized_claim') or ''))}</small></td>"
            f"<td>{'YES' if row.get('overlaps_gold') else 'no'}</td>"
            f"<td>{'YES' if row.get('overlaps_dual_prediction') else 'no'}</td>"
            f"<td><b>{html.escape(str(row.get('balanced_relation') or '—'))}</b> "
            f"({html.escape(str(row.get('balanced_confidence') or '—'))})<br><small>{html.escape(str(row.get('balanced_explanation') or ''))}</small></td>"
            f"<td>{source_html}</td>"
            f"<td><b>{html.escape(str(row.get('six_agent_verdict') or '—'))}</b> "
            f"({html.escape(str(row.get('six_agent_confidence') or '—'))})<br>"
            f"<small>{html.escape(str(row.get('six_agent_explanation') or ''))}</small><br>"
            f"<small>{html.escape(', '.join(row.get('six_agent_changed_dimensions') or []))}</small></td>"
            "</tr>"
        )
    source_trace = result.get("source_six_agent") or {}
    response_trace = result.get("response_six_agent") or {}
    cross_trace = result.get("cross_graph") or {}
    graph_payload = {
        "candidateSource": balanced.get("source_graph") or {},
        "validatedSource": result.get("validated_source_graph") or {},
        "candidateResponse": balanced.get("response_graph") or {},
        "validatedResponse": result.get("validated_response_graph") or {},
        "balancedAlignments": balanced.get("alignments") or [],
        "sixAgentVerdicts": cross_trace.get("final_verdicts") or [],
        "nodeComparisons": rows,
        "evidenceUnits": ((candidate.get("evidence_card") or {}).get("units") or []),
    }
    graph_json = _safe_script_json(graph_payload)
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Catch review {html.escape(str(candidate.get('case_id')))}</title>
<style>
:root{{--blue:#2563eb;--line:#cbd5e1;--muted:#64748b;--ink:#172033;--panel:#ffffff;--bg:#f4f6fa;--green:#16a34a;--red:#dc2626;--amber:#d97706;--purple:#7c3aed}}*{{box-sizing:border-box}}body{{font-family:Inter,Arial,sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.45}}.page{{max-width:1780px;margin:auto;padding:22px}}h1,h2{{margin-top:28px}}table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff}}th,td{{border:1px solid #cbd5e1;padding:8px;vertical-align:top}}th{{background:#eff6ff}}tr.gold{{background:#fff7ed}}tr.pred td:first-child{{border-left:5px solid #2563eb}}code,pre{{background:#f8fafc}}pre{{padding:12px;border:1px solid #e2e8f0;border-radius:8px;white-space:pre-wrap;max-height:520px;overflow:auto}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.card{{background:#fff;border:1px solid #dbe3ee;border-radius:10px;padding:12px}}.note{{background:#eef6ff;border:1px solid #bfdbfe;border-radius:10px;padding:12px}}small{{color:#475569}}button,select{{border:1px solid #b8c1d1;border-radius:8px;padding:8px 11px;background:#fff;color:#334155;font-weight:700;cursor:pointer}}button.active{{background:#1d4ed8;color:#fff;border-color:#1d4ed8}}.graph-shell{{background:#fff;border:1px solid #d9deea;border-radius:13px;padding:14px}}.graph-toolbar{{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:10px}}.graph-tabs,.graph-controls{{display:flex;gap:7px;align-items:center;flex-wrap:wrap}}.graph-workspace{{display:grid;grid-template-columns:minmax(720px,1.55fr) minmax(330px,.7fr);gap:12px}}.graph-canvas{{height:720px;overflow:auto;border:1px solid var(--line);border-radius:10px;background:linear-gradient(#fff,#fbfcff)}}#caseGraph{{display:block}}.graph-detail{{min-height:720px;border:1px solid var(--line);border-radius:10px;padding:13px;background:#fff}}.detail-empty{{padding:30px 8px;text-align:center;color:var(--muted)}}.detail-title{{font-size:18px;font-weight:800;margin:8px 0}}.badge{{display:inline-block;border:1px solid #cad2df;border-radius:999px;padding:4px 8px;font-size:12px;background:#fff;margin:2px}}.badge.supported_by,.badge.safe_inference{{background:#dcfce7;color:#166534;border-color:#bbf7d0}}.badge.contradicted_by{{background:#fee2e2;color:#991b1b;border-color:#fecaca}}.badge.not_found_in_source,.badge.partially_supported_by{{background:#fef3c7;color:#92400e;border-color:#fde68a}}.badge.generic_advice,.badge.not_factual,.badge.uncertain{{background:#e0f2fe;color:#075985;border-color:#bae6fd}}.quote{{background:#f8fafc;border-left:4px solid #94a3b8;border-radius:6px;padding:10px;margin:10px 0}}.meaning{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:9px;padding:10px;margin:8px 0}}.kv{{display:grid;grid-template-columns:135px 1fr;border-top:1px solid #e5e7eb;margin-top:10px}}.kv div{{padding:7px 3px;border-bottom:1px solid #e5e7eb}}.kv .k{{font-size:12px;color:#64748b;font-weight:700}}.connection{{border:1px solid #e2e8f0;border-radius:8px;padding:8px;margin:6px 0;cursor:pointer}}.connection:hover{{background:#f8fafc}}.node{{cursor:pointer}}.node rect{{stroke-width:2}}.node:hover rect{{filter:drop-shadow(0 2px 4px #64748b55)}}.node.selected rect{{stroke-width:4}}.node text{{font-size:11px;pointer-events:none}}.edge{{fill:none;stroke:#64748b;stroke-width:2;opacity:.7;cursor:pointer}}.edge.cross{{stroke:#7c3aed;stroke-dasharray:7 5}}.edge:hover{{opacity:1;stroke-width:3}}.edge-label{{font-size:10px;fill:#475569;pointer-events:none}}.source-node{{fill:#dbeafe;stroke:#2563eb}}.response-node{{fill:#f1f5f9;stroke:#64748b}}.response-node.supported_by,.response-node.safe_inference{{fill:#dcfce7;stroke:#16a34a}}.response-node.contradicted_by{{fill:#fee2e2;stroke:#dc2626}}.response-node.not_found_in_source,.response-node.partially_supported_by{{fill:#fef3c7;stroke:#d97706}}.response-node.generic_advice,.response-node.not_factual,.response-node.uncertain{{fill:#e0f2fe;stroke:#0284c7}}.legend{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}.legend span{{border:1px solid #cad2df;border-radius:999px;padding:4px 8px;font-size:12px;background:#fff}}.section-title{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#64748b;font-weight:800;margin-top:14px}}.json{{white-space:pre-wrap;background:#0f172a;color:#dbeafe;border-radius:8px;padding:10px;max-height:300px;overflow:auto;font:12px ui-monospace,monospace}}@media(max-width:1100px){{.graph-workspace{{grid-template-columns:1fr}}.graph-detail{{min-height:auto}}.cards{{grid-template-columns:1fr}}}}
</style></head><body><div class='page'>
<h1>{html.escape(report_heading)}</h1>
<p class='note'>{html.escape(report_note)}</p>
<div class='cards'>
<div class='card'><b>Case</b><br>{html.escape(str(candidate.get('case_id')))}<br><small>{html.escape(str(candidate.get('catch_reason')))}</small></div>
<div class='card'><b>Raw F1</b><br>{html.escape(str((candidate.get('raw') or {}).get('scores',{}).get('char_f1')))}</div>
<div class='card'><b>Balanced F1</b><br>{html.escape(str((candidate.get('balanced_dual_graph') or {}).get('scores',{}).get('char_f1')))}</div>
</div>
<h2>Interactive Source / Response Graph</h2>
<div class='graph-shell'>
<div class='graph-toolbar'>
  <div class='graph-tabs'>
    <button type='button' class='active' data-view='source'>Source Graph</button>
    <button type='button' data-view='response'>Response Graph</button>
    <button type='button' data-view='cross'>Cross comparison</button>
  </div>
  <div class='graph-controls'><label for='graphStage'><small>Graph stage</small></label><select id='graphStage'><option value='validated'>6-Agent validated</option><option value='candidate'>Balanced candidate</option></select><button type='button' id='fitGraph'>Reset view</button></div>
</div>
<div class='legend'><span>Blue = Source evidence</span><span>Green = supported</span><span>Red = contradiction</span><span>Amber = unsupported/partial</span><span>Light blue = safe/non-factual/uncertain</span></div>
<div class='graph-workspace'><div class='graph-canvas'><svg id='caseGraph' width='1220' height='720' aria-label='Interactive Source and Response graph'></svg></div><div id='graphDetail' class='graph-detail'><div class='detail-empty'>Node 또는 Edge를 클릭하면 Discussion Lab처럼 상세 정보를 표시합니다.</div></div></div>
</div>
<h2>Task</h2><pre>{html.escape(str(candidate.get('task_instruction') or ''))}</pre>
<h2>Source evidence</h2><pre>{html.escape(str((candidate.get('evidence_card') or {}).get('text') or ''))}</pre>
<h2>Response</h2><pre>{html.escape(str(candidate.get('response') or ''))}</pre>
<div class='cards'>
<div class='card'><b>Gold</b><br>{_span_text(candidate.get('gold_labels') or [])}</div>
<div class='card'><b>Raw predictions</b><br>{_span_text((candidate.get('raw') or {}).get('predicted_spans') or [])}</div>
<div class='card'><b>Balanced predictions</b><br>{_span_text((candidate.get('balanced_dual_graph') or {}).get('predicted_spans') or [])}</div>
</div>
<h2>Node-by-node Source comparison</h2>
<p><small>Orange rows overlap the gold hallucination. Blue left border means the Balanced benchmark projected an error from that node.</small></p>
<table><thead><tr><th>Response node</th><th>Claim</th><th>Gold overlap</th><th>Dual span</th><th>Balanced alignment</th><th>Matched Source nodes</th><th>Selected 6-Agent verdict</th></tr></thead><tbody>{''.join(row_html)}</tbody></table>
<h2>Source Graph six-agent findings</h2><pre>{_pretty_json(source_trace)}</pre>
<h2>Response Graph six-agent findings</h2><pre>{_pretty_json(response_trace)}</pre>
<h2>Cross-Graph six-agent trace</h2><pre>{_pretty_json(cross_trace)}</pre>
<h2>Validated Source Graph</h2><pre>{_pretty_json(result.get('validated_source_graph'))}</pre>
<h2>Validated Response Graph</h2><pre>{_pretty_json(result.get('validated_response_graph'))}</pre>
<script>const GRAPH_DATA={graph_json};</script>
<script>
(() => {{
  const svg = document.getElementById('caseGraph');
  const detail = document.getElementById('graphDetail');
  const stageSelect = document.getElementById('graphStage');
  let activeView = 'source';
  let selectedKey = '';
  const NS = 'http://www.w3.org/2000/svg';
  const sourceTypeOrder = ['definition','constraint','source_fact','quantitative_fact','qualified_fact'];
  const responseTypeOrder = ['claim','quantitative_claim','comparison_claim','causal_claim','qualified_claim','conclusion'];
  const safe = v => String(v ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
  const balancedById = Object.fromEntries((GRAPH_DATA.balancedAlignments||[]).map(x => [String(x.response_node_id), x]));
  const sixById = Object.fromEntries((GRAPH_DATA.sixAgentVerdicts||[]).map(x => [String(x.response_node_id), x]));
  const comparisonById = Object.fromEntries((GRAPH_DATA.nodeComparisons||[]).map(x => [String(x.node_id), x]));
  const evidenceById = Object.fromEntries((GRAPH_DATA.evidenceUnits||[]).map(x => [String(x.id), x]));

  function currentGraphs() {{
    const validated = stageSelect.value === 'validated';
    return {{
      source: validated ? GRAPH_DATA.validatedSource : GRAPH_DATA.candidateSource,
      response: validated ? GRAPH_DATA.validatedResponse : GRAPH_DATA.candidateResponse,
    }};
  }}
  function verdictFor(id) {{ return (sixById[id]||{{}}).verdict || (balancedById[id]||{{}}).relation || ''; }}
  function words(text, max=31) {{
    const out=[]; let line='';
    for(const word of String(text||'').split(/\\s+/)) {{
      if((line+' '+word).trim().length>max) {{ if(line) out.push(line); line=word; }} else line=(line+' '+word).trim();
    }}
    if(line) out.push(line); if(out.length>3) return [out[0],out[1],out[2]+'…']; return out;
  }}
  function graphEdges(graph) {{
    const out=[];
    for(const e of (graph.edges||[])) for(const sid of (e.source_ids||[])) out.push({{id:`${{sid}}→${{e.target_id}}:${{e.relation}}`,source:String(sid),target:String(e.target_id),relation:e.relation,raw:e}});
    return out;
  }}
  function groupedLayout(nodes, kind) {{
    const order = kind==='source' ? sourceTypeOrder : responseTypeOrder;
    const groups = new Map();
    for(const n of nodes) {{ const type=n.node_type||'other'; if(!groups.has(type)) groups.set(type,[]); groups.get(type).push(n); }}
    const types=[...order.filter(x=>groups.has(x)), ...[...groups.keys()].filter(x=>!order.includes(x))];
    const pos={{}}; let maxY=0;
    types.forEach((type, col)=>{{ (groups.get(type)||[]).forEach((n,row)=>{{pos[String(n.id)]={{x:40+col*265,y:55+row*105}};maxY=Math.max(maxY,55+row*105);}}); }});
    return {{pos,width:Math.max(1050,80+types.length*265),height:Math.max(680,maxY+120)}};
  }}
  function crossLayout(sourceNodes,responseNodes) {{
    const pos={{}}; const gap=96;
    sourceNodes.forEach((n,i)=>pos['S:'+n.id]={{x:35,y:45+i*gap}});
    responseNodes.forEach((n,i)=>pos['R:'+n.id]={{x:675,y:45+i*gap}});
    return {{pos,width:1050,height:Math.max(680,90+Math.max(sourceNodes.length,responseNodes.length)*gap)}};
  }}
  function linePath(a,b) {{ const x1=a.x+230,y1=a.y+34,x2=b.x,y2=b.y+34,m=(x1+x2)/2; return `M${{x1}},${{y1}} C${{m}},${{y1}} ${{m}},${{y2}} ${{x2}},${{y2}}`; }}
  function addMarker() {{
    const defs=document.createElementNS(NS,'defs'); const marker=document.createElementNS(NS,'marker');
    marker.setAttribute('id','arrow');marker.setAttribute('markerWidth','8');marker.setAttribute('markerHeight','8');marker.setAttribute('refX','7');marker.setAttribute('refY','3');marker.setAttribute('orient','auto');
    const p=document.createElementNS(NS,'path');p.setAttribute('d','M0,0 L0,6 L7,3 z');p.setAttribute('fill','#64748b');marker.appendChild(p);defs.appendChild(marker);svg.appendChild(defs);
  }}
  function drawEdge(edge,a,b,isCross=false) {{
    const path=document.createElementNS(NS,'path'); path.setAttribute('d',linePath(a,b)); path.setAttribute('marker-end','url(#arrow)'); path.setAttribute('class','edge'+(isCross?' cross':''));
    path.addEventListener('click',()=>showEdge(edge,isCross)); svg.appendChild(path);
    const label=document.createElementNS(NS,'text');label.setAttribute('x',String((a.x+b.x+230)/2));label.setAttribute('y',String((a.y+b.y)/2+25));label.setAttribute('class','edge-label');label.textContent=edge.relation||'';svg.appendChild(label);
  }}
  function drawNode(node,p,kind,prefix='') {{
    const id=String(node.id); const key=prefix+id; const verdict=kind==='response'?verdictFor(id):'';
    const g=document.createElementNS(NS,'g');g.setAttribute('class','node'+(selectedKey===key?' selected':''));g.setAttribute('transform',`translate(${{p.x}},${{p.y}})`);
    const rect=document.createElementNS(NS,'rect');rect.setAttribute('width','230');rect.setAttribute('height','68');rect.setAttribute('rx','9');rect.setAttribute('class',kind==='source'?'source-node':`response-node ${{verdict}}`);g.appendChild(rect);
    const title=document.createElementNS(NS,'text');title.setAttribute('x','10');title.setAttribute('y','16');title.setAttribute('font-weight','700');title.textContent=`${{id}} · ${{node.node_type||kind}}`;g.appendChild(title);
    const text=document.createElementNS(NS,'text');text.setAttribute('x','10');text.setAttribute('y','32');words(node.normalized_claim||node.text||'',34).forEach((line,i)=>{{const t=document.createElementNS(NS,'tspan');t.setAttribute('x','10');t.setAttribute('dy',i===0?'0':'13');t.textContent=line;text.appendChild(t);}});g.appendChild(text);
    g.addEventListener('click',()=>{{document.querySelectorAll('.node').forEach(x=>x.classList.remove('selected'));g.classList.add('selected');showNode(node,kind,key);}});svg.appendChild(g);
  }}
  function render() {{
    const graphs=currentGraphs(); svg.innerHTML=''; addMarker(); selectedKey=''; detail.innerHTML='<div class="detail-empty">Node 또는 Edge를 클릭하면 상세 정보를 표시합니다.</div>';
    if(activeView==='cross') return renderCross(graphs);
    const kind=activeView; const graph=graphs[kind]||{{nodes:[],edges:[]}}; const nodes=graph.nodes||[]; const layout=groupedLayout(nodes,kind);svg.setAttribute('width',layout.width);svg.setAttribute('height',layout.height);
    for(const e of graphEdges(graph)) if(layout.pos[e.source]&&layout.pos[e.target]) drawEdge(e,layout.pos[e.source],layout.pos[e.target],false);
    for(const n of nodes) drawNode(n,layout.pos[String(n.id)],kind,'');
  }}
  function renderCross(graphs) {{
    const sourceNodes=graphs.source.nodes||[], responseNodes=graphs.response.nodes||[];const layout=crossLayout(sourceNodes,responseNodes);svg.setAttribute('width',layout.width);svg.setAttribute('height',layout.height);
    const alignments=stageSelect.value==='validated'?(GRAPH_DATA.sixAgentVerdicts||[]):(GRAPH_DATA.balancedAlignments||[]);
    for(const a of alignments) for(const sid of (a.source_node_ids||[])) {{ const sp=layout.pos['S:'+sid],rp=layout.pos['R:'+a.response_node_id];if(sp&&rp)drawEdge({{source:sid,target:a.response_node_id,relation:a.verdict||a.relation,raw:a}},sp,rp,true); }}
    for(const n of sourceNodes) drawNode(n,layout.pos['S:'+n.id],'source','S:');
    for(const n of responseNodes) drawNode(n,layout.pos['R:'+n.id],'response','R:');
  }}
  function connectionsFor(id,kind) {{
    const graph=currentGraphs()[kind]||{{}}; return graphEdges(graph).filter(e=>e.source===id||e.target===id);
  }}
  function showNode(node,kind,key) {{
    selectedKey=key; const id=String(node.id); const connections=connectionsFor(id,kind); const comparison=comparisonById[id]||{{}}; const bal=balancedById[id]||{{}};const six=sixById[id]||{{}};
    let evidence=''; if(kind==='source') evidence=(node.evidence_ids||[]).map(eid=>`<div class="connection"><b>${{safe(eid)}}</b> · ${{safe((evidenceById[eid]||{{}}).text||'')}}</div>`).join('')||'—';
    else evidence=(six.source_node_ids||bal.source_node_ids||[]).map(sid=>{{const source=(currentGraphs().source.nodes||[]).find(n=>String(n.id)===String(sid));return `<div class="connection"><b>${{safe(sid)}}</b> · ${{safe(source?.text||'')}}</div>`;}}).join('')||'—';
    detail.innerHTML=`<div><span class="badge">${{safe(kind.toUpperCase())}}</span><span class="badge">${{safe(node.node_type||'node')}}</span>${{kind==='response'?`<span class="badge ${{safe(verdictFor(id))}}">${{safe(verdictFor(id)||'unclassified')}}</span>`:''}}</div><div class="detail-title">${{safe(node.normalized_claim||node.text||id)}}</div><div class="section-title">Exact graph text</div><div class="quote">${{safe(node.text||'')}}</div>${{node.normalized_claim?`<div class="section-title">Normalized claim</div><div class="meaning">${{safe(node.normalized_claim)}}</div>`:''}}<div class="kv"><div class="k">Node ID</div><div>${{safe(id)}}</div><div class="k">Node type</div><div>${{safe(node.node_type||'')}}</div>${{node.sentence_id?`<div class="k">Sentence ID</div><div>${{safe(node.sentence_id)}}</div>`:''}}${{kind==='response'?`<div class="k">Balanced relation</div><div>${{safe(bal.relation||'—')}} (${{safe(bal.confidence??'—')}})</div><div class="k">6-Agent verdict</div><div>${{safe(six.verdict||'—')}} (${{safe(six.confidence??'—')}})</div><div class="k">Gold overlap</div><div>${{comparison.overlaps_gold?'YES':'no'}}</div><div class="k">Projected by DualGraph</div><div>${{comparison.overlaps_dual_prediction?'YES':'no'}}</div>`:''}}</div><div class="section-title">${{kind==='source'?'Evidence units':'Matched Source nodes'}}</div>${{evidence}}<div class="section-title">Relations</div>${{connections.map(e=>`<div class="connection">${{safe(e.source)}} → ${{safe(e.target)}} · <b>${{safe(e.relation)}}</b></div>`).join('')||'—'}}${{kind==='response'&&bal.explanation?`<div class="section-title">Balanced explanation</div><div class="meaning">${{safe(bal.explanation)}}</div>`:''}}${{kind==='response'&&six.explanation?`<div class="section-title">6-Agent explanation</div><div class="meaning">${{safe(six.explanation)}}</div>`:''}}<details><summary>Raw JSON</summary><pre class="json">${{safe(JSON.stringify({{node,balanced:bal,sixAgent:six,comparison}},null,2))}}</pre></details>`;
  }}
  function showEdge(edge,isCross) {{
    const raw=edge.raw||edge; detail.innerHTML=`<div><span class="badge">${{isCross?'CROSS EDGE':'EDGE'}}</span><span class="badge">${{safe(edge.relation||'')}}</span></div><div class="detail-title">${{safe(edge.source)}} → ${{safe(edge.target)}}</div><div class="meaning">${{safe(raw.explanation||raw.rationale||raw.match_type||'Graph relation')}}</div><pre class="json">${{safe(JSON.stringify(raw,null,2))}}</pre>`;
  }}
  document.querySelectorAll('[data-view]').forEach(btn=>btn.addEventListener('click',()=>{{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('active'));btn.classList.add('active');activeView=btn.dataset.view;render();}}));
  stageSelect.addEventListener('change',render);document.getElementById('fitGraph').addEventListener('click',()=>{{document.querySelector('.graph-canvas').scrollTo({{top:0,left:0,behavior:'smooth'}});}});
  render();
}})();
</script>
</div></body></html>"""

def run_selected_catch_review(
    *,
    benchmark_run_dir: Path,
    case_id: str,
    output_root: Path,
    cache_path: Path | None = None,
    model: str = "gpt-5.4-nano",
    reasoning_effort: str = "low",
    max_output_tokens: int = 2600,
    client: Any = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    _load_local_env()
    progress = progress or (lambda _message: None)
    if model != "gpt-5.4-nano":
        raise ValueError("Catch review is fixed to gpt-5.4-nano")
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium, or high")
    candidate = load_catch_candidate(benchmark_run_dir, case_id)
    balanced = candidate.get("balanced_dual_graph") or {}
    source_graph = SourceEvidenceGraphOutput.model_validate(balanced.get("source_graph") or {})
    response_graph = ResponseClaimGraphOutput.model_validate(balanced.get("response_graph") or {})
    evidence_card = candidate.get("evidence_card") or {}
    case = {
        "case_id": candidate.get("case_id"),
        "source_id": candidate.get("source_id"),
        "task_instruction": candidate.get("task_instruction"),
        "response": candidate.get("response"),
        "task_type": candidate.get("task_type"),
        "gold_labels": candidate.get("gold_labels") or [],
    }
    active_client = client
    if active_client is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is not configured")
        from openai import OpenAI
        active_client = OpenAI()

    cache = _load_cache(cache_path) if cache_path else _empty_cache()
    cache_lock = threading.Lock()
    force: set[str] = set()
    progress(f"Selected case {case_id}: Source and Response Graph six-agent validation in parallel")

    def validate_source():
        return _validate_graph_six_agents(
            "source", case=case, evidence_card=evidence_card, graph=source_graph,
            client=active_client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens, cache=cache, cache_lock=cache_lock,
            cache_path=cache_path, force=force,
        )

    def validate_response():
        return _validate_graph_six_agents(
            "response", case=case, evidence_card=evidence_card, graph=response_graph,
            client=active_client, model=model, reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens, cache=cache, cache_lock=cache_lock,
            cache_path=cache_path, force=force,
        )

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="selected-catch-six-agent") as executor:
        source_future = executor.submit(validate_source)
        response_future = executor.submit(validate_response)
        validated_source, source_records, source_trace, source_calls, source_hits = source_future.result()
        validated_response, response_records, response_trace, response_calls, response_hits = response_future.result()

    progress("Cross-Graph Evidence Matcher → Logic → conditional Assumption/Judge → Span Projector")
    alignment, cross_records, cross_trace, cross_calls, cross_hits = _run_cross_graph_pipeline(
        case=case, evidence_card=evidence_card,
        source_graph=validated_source, response_graph=validated_response,
        client=active_client, model=model, reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens, cache=cache, cache_lock=cache_lock,
        cache_path=cache_path, force=force,
    )
    six_predictions, six_details = _predictions_from_alignment(
        alignment,
        validated_response,
        str(candidate.get("response") or ""),
        gate_profile="v049_balanced_recall",
    )
    if cache_path:
        _save_cache(cache_path, cache)

    run_id = f"catch6_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_run_dir": str(benchmark_run_dir),
        "candidate": candidate,
        "balanced_snapshot": balanced,
        "validated_source_graph": validated_source.model_dump(),
        "validated_response_graph": validated_response.model_dump(),
        "source_six_agent": source_trace,
        "response_six_agent": response_trace,
        "cross_graph": cross_trace,
        "six_agent_alignment": alignment.model_dump(),
        "six_agent_predicted_spans": six_predictions,
        "six_agent_prediction_details": six_details,
        "api_calls_this_run": source_calls + response_calls + cross_calls,
        "cache_hits_this_run": source_hits + response_hits + cross_hits,
        "generation_records": source_records + response_records + cross_records,
    }
    result["node_comparisons"] = _node_comparison_rows(candidate, result)
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    graph_view = {
        "candidate_source_graph": balanced.get("source_graph") or {},
        "validated_source_graph": result.get("validated_source_graph") or {},
        "candidate_response_graph": balanced.get("response_graph") or {},
        "validated_response_graph": result.get("validated_response_graph") or {},
        "balanced_alignments": balanced.get("alignments") or [],
        "six_agent_verdicts": (cross_trace.get("final_verdicts") or []),
    }
    (run_dir / "graph_view.json").write_text(json.dumps(graph_view, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "report.html").write_text(_report_html(result), encoding="utf-8")
    (run_dir / "README.txt").write_text(
        "Open report.html for the interactive Source, Response, and Cross Graph views. graph_view.json contains the same graph payload.\n",
        encoding="utf-8",
    )
    return result
