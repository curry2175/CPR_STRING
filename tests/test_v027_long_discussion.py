from __future__ import annotations

from types import SimpleNamespace

from vrg.discussion_graph import (
    DiscussionGraphOutput,
    DiscussionNode,
    generate_discussion_graph,
    split_discussion_text,
)


class _Responses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.outputs.pop(0)
        item = SimpleNamespace(type="output_text", parsed=parsed)
        message = SimpleNamespace(type="message", content=[item])
        usage = SimpleNamespace(model_dump=lambda: {"input_tokens": 100, "output_tokens": 40, "total_tokens": 140})
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            model=kwargs["model"],
            status="completed",
            output=[message],
            usage=usage,
        )


class _Client:
    def __init__(self, outputs):
        self.responses = _Responses(outputs)


def _fixture_for_chunk(chunk: str, index: int) -> DiscussionGraphOutput:
    first = chunk.split(".", 1)[0].strip() + "."
    return DiscussionGraphOutput(
        paragraph_summary=f"Chunk {index} summary.",
        nodes=[
            DiscussionNode(
                id="local",
                sentence_index=1,
                source_text=first,
                plain_meaning=f"Claim from chunk {index}.",
                role="observation",
                assertion_type="descriptive",
                polarity="positive",
                certainty="reported",
            )
        ],
        edges=[],
        issues=[],
        overall_assessment="internally_consistent",
    )


def test_v027_splitter_preserves_full_text_without_total_limit():
    text = "\n\n".join(
        f"Section {i}. " + ("This is a complete scientific sentence. " * 140)
        for i in range(1, 12)
    )
    assert len(text) > 30000
    chunks = split_discussion_text(text, max_chars=4000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 4000 for chunk in chunks)
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


def test_v027_long_discussion_is_auto_chunked_and_does_not_use_z3(monkeypatch):
    monkeypatch.setenv("DISCUSSION_CHUNK_CHARS", "4000")
    monkeypatch.delenv("DISCUSSION_MAX_INPUT_CHARS", raising=False)
    monkeypatch.delenv("DISCUSSION_MAX_CHUNKS", raising=False)
    text = "\n\n".join(
        f"Section {i}. " + ("Treatment was associated with an outcome. " * 130)
        for i in range(1, 12)
    )
    assert len(text) > 30000
    chunks = split_discussion_text(text, max_chars=4000)
    client = _Client([_fixture_for_chunk(chunk, i) for i, chunk in enumerate(chunks, 1)])

    result = generate_discussion_graph(
        text,
        model="gpt-5.6",
        reasoning_effort="low",
        client=client,
        architecture_mode="legacy_single_pass",
    )

    assert result["schema_version"] == "0.27.0"
    assert result["analysis_mode"] == "auto_chunked"
    assert result["chunk_count"] == len(chunks)
    assert result["api_call_count"] == len(chunks)
    assert result["input_char_count"] == len(text)
    assert result["application_input_limit"] is None
    assert result["verification_engine"] == "typed_graph_structural_rules"
    assert result["z3_used"] is False
    assert result["usage"]["total_tokens"] == 140 * len(chunks)
    assert len(client.responses.calls) == len(chunks)
    assert result["summary"]["node_count"] == len(chunks)


def test_v027_optional_admin_limit_can_be_reenabled(monkeypatch):
    monkeypatch.setenv("DISCUSSION_MAX_INPUT_CHARS", "5000")
    text = "A" * 5001
    client = _Client([])
    try:
        generate_discussion_graph(text, model="gpt-5.6", client=client)
    except ValueError as exc:
        assert "관리자가 설정한 한도" in str(exc)
    else:
        raise AssertionError("configured administrator limit should be enforced")


def test_v027_discussion_ui_explains_unbounded_chunking_and_z3_status():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    page = (root / "static" / "discussion_lab.html").read_text(encoding="utf-8")
    assert "Discussion Reasoning Lab · v028" in page
    assert "고정 입력 글자 수 제한은 없습니다" in page
    assert "자동 chunking" in page
    assert "Discussion Lab에서는 현재 Z3를 사용하지 않습니다" in page
    assert "api_call_count" in page
    assert "chunk_count" in page
