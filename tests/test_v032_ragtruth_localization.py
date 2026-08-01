from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from vrg.ragtruth_localization import (
    ClaimNode,
    DirectSpanOutput,
    LightClaimGraphOutput,
    SpanPrediction,
    build_evidence_card,
    load_ragtruth_cases,
    locate_exact_quote,
    run_ragtruth_localization,
    score_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data" / "ragtruth_fixture"


def test_v032_fixture_loader_balances_task_and_hallucination():
    rows, info = load_ragtruth_cases(
        FIXTURE / "response.jsonl",
        FIXTURE / "source_info.jsonl",
        limit=4,
        seed=1,
    )
    assert len(rows) == 4
    assert sum(row["gold_has_hallucination"] for row in rows) == 2
    assert set(row["task_type"] for row in rows) == {"QA", "Summary"}
    assert info["selected"] == 4


def test_v032_evidence_card_uses_stable_ids():
    card = build_evidence_card(
        {"question": "What is the capital?", "passages": "passage 1: Paris is the capital.\n\npassage 2: Lyon is a city."},
        "The capital is Lyon.",
        max_context_chars=10_000,
    )
    assert card["selected_unit_count"] >= 2
    assert "[e1]" in card["text"]


def test_v032_quote_location_and_character_score():
    response = "The capital of France is Lyon."
    located = locate_exact_quote(response, "Lyon", "a1")
    assert located == (25, 29)
    predicted = [{"start": 25, "end": 29, "text": "Lyon", "label_type": "contradiction"}]
    gold = [{"start": 25, "end": 29, "text": "Lyon", "label_type": "contradiction"}]
    score = score_predictions(response, predicted, gold)
    assert score["char_f1"] == 1.0
    assert score["span_f1_iou50"] == 1.0
    assert score["matched_type_accuracy"] == 1.0


class _Responses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        user = " ".join(str(kwargs.get("input") or "").split())
        output_type = kwargs["text_format"]
        hallucinated = "capital of France is Lyon" in user or "cut mortality by 50 percent" in user
        if output_type is DirectSpanOutput:
            if "capital of France is Lyon" in user:
                parsed = DirectSpanOutput(hallucinated_spans=[SpanPrediction(
                    sentence_id="a1", text="Lyon", label_type="contradiction", evidence_ids=["e1"]
                )])
            elif "cut mortality by 50 percent" in user:
                parsed = DirectSpanOutput(hallucinated_spans=[SpanPrediction(
                    sentence_id="a1", text="cut mortality by 50 percent", label_type="unsupported"
                )])
            else:
                parsed = DirectSpanOutput(hallucinated_spans=[])
        elif output_type is LightClaimGraphOutput:
            if "capital of France is Lyon" in user:
                parsed = LightClaimGraphOutput(claims=[ClaimNode(
                    id="c1", sentence_id="a1", text="Lyon", relation="contradicted", evidence_ids=["e1"]
                )])
            elif "cut mortality by 50 percent" in user:
                parsed = LightClaimGraphOutput(claims=[ClaimNode(
                    id="c1", sentence_id="a1", text="cut mortality by 50 percent", relation="unsupported"
                )])
            else:
                text = "Paris" if "capital of France is Paris" in user else "no significant mortality difference"
                parsed = LightClaimGraphOutput(claims=[ClaimNode(
                    id="c1", sentence_id="a1", text=text, relation="supported", evidence_ids=["e1"]
                )])
        else:
            raise AssertionError(output_type)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            output=[message],
            usage=usage,
        )


class _Client:
    def __init__(self):
        self.responses = _Responses()


def test_v032_one_call_graph_and_persistent_cache(tmp_path):
    client = _Client()
    cache = tmp_path / "cache.json"
    kwargs = dict(
        response_path=FIXTURE / "response.jsonl",
        source_path=FIXTURE / "source_info.jsonl",
        output_root=tmp_path / "outputs",
        small_model="gpt-5.4-nano",
        include_reference=False,
        include_checklist=False,
        limit=4,
        seed=1,
        generation_cache_path=cache,
        client=client,
    )
    first = run_ragtruth_localization(**kwargs)
    assert len(client.responses.calls) == 8  # 4 direct + 4 one-pass graph
    assert first["summary"]["actual_api_calls_this_run"] == 8
    assert first["method_summaries"]["small_claim_graph"]["char_f1_percent"] == 100.0

    second = run_ragtruth_localization(**kwargs)
    assert len(client.responses.calls) == 8
    assert second["summary"]["actual_api_calls_this_run"] == 0
    assert second["cache_summary"]["cache_hits_by_method"] == {
        "small_direct_span": 4,
        "small_claim_graph": 4,
    }
