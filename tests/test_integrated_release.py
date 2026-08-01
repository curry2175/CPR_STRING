from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_chrome_extension_manifest_and_assets():
    folder = ROOT / "extensions" / "chrome"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert manifest["name"] == "STRING"
    for name in ("service-worker.js", "content-script.js", "sidepanel.html", "options.html"):
        assert (folder / name).exists()


def test_impact_factor_app_uses_single_root_core():
    from applications.impact_factor import analyze_batch
    assert analyze_batch.DEFAULT_MODULE.resolve() == ROOT.resolve()
    assert not (ROOT / "applications" / "impact_factor" / "modules" / "discussion_lab").exists()


def test_aime_auditor_and_question_builder():
    from applications.aime_self_revision.auditor import audit
    from applications.aime_self_revision.questioner import build
    graph = {
        "nodes": [{"id": "n1", "role": "conclusion", "source_text": "Therefore 39",
                   "plain_meaning": "Therefore 39", "certainty": "proves",
                   "assertion_type": "universal"}],
        "edges": [], "issues": [],
    }
    defects = audit(graph)
    assert any(row.kind == "unsupported_conclusion" for row in defects)
    assert build(graph, defects)


def test_release_has_single_vrg_package():
    copies = [p for p in ROOT.rglob("vrg") if p.is_dir() and ".venv" not in p.parts]
    assert copies == [ROOT / "vrg"]
