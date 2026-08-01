"""collect_discussions.py — Europe PMC에서 Discussion 섹션을 자동 수집한다.

이 스크립트는 인터넷이 되는 PC(= 동현님 로컬)에서 돌아야 한다.
클라우드 샌드박스는 NCBI/EuropePMC 접근이 차단되어 있다.

의존성: 표준 라이브러리만 사용한다. pip install 불필요.

수집 경로
    1. Europe PMC search API 로 후보 논문을 찾는다.
    2. OA full text (JATS XML) 가 있는 것만 남긴다.
    3. XML 에서 Discussion 섹션만 잘라낸다. 소제목 구조도 함께 기록한다.
    4. 저널명 → JIF 매핑을 붙인다 (journals.csv).
    5. corpus JSONL 로 저장한다. discussion_lab 이 그대로 먹을 수 있는 형태다.

중요한 한계 — 반드시 읽을 것
    PMC OA 수록 여부는 저널마다 다르고, **최상위 IF 저널일수록 OA 전문이 없을
    확률이 높다** (NEJM 등). 이건 독립변수(IF)와 상관된 선택편향이다.
    그래서 이 스크립트는 OA 로 못 가져온 논문을 조용히 버리지 않고
    `missing_oa.csv` 에 남긴다. 그 목록은 손으로 채워 넣어야 한다.
    수집 방식은 레코드마다 `text_source` 필드에 남는다: oa_jats | manual

사용법
    python collect_discussions.py --config studysets/hcq_covid.json
    python collect_discussions.py --query "hydroxychloroquine AND covid-19" --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
UA = "STRING-agent-ifxlogic/0.1 (YAI hackathon; academic use)"
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- http


def _get(url: str, *, retries: int = 3, pause: float = 0.34) -> bytes:
    """Europe PMC 는 무인증이지만 초당 요청을 제한한다. 예의를 지킨다."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            time.sleep(pause)
            return payload
        except Exception as exc:  # noqa: BLE001 - 네트워크는 무엇이든 던진다
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}\n  {last}")


def search(query: str, *, limit: int, page_size: int = 100) -> Iterator[dict[str, Any]]:
    """cursorMark 페이지네이션. resultType=core 여야 저널명·연도가 온다."""
    cursor = "*"
    fetched = 0
    while fetched < limit:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "format": "json",
                "resultType": "core",
                "pageSize": min(page_size, limit - fetched),
                "cursorMark": cursor,
            }
        )
        data = json.loads(_get(f"{EPMC}/search?{params}"))
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return
        for record in results:
            yield record
            fetched += 1
            if fetched >= limit:
                return
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            return
        cursor = next_cursor


def fetch_fulltext_xml(pmcid: str) -> bytes | None:
    try:
        return _get(f"{EPMC}/{pmcid}/fullTextXML")
    except RuntimeError:
        return None


# ------------------------------------------------------------------ JATS 파싱


_DISCUSSION_TITLE = re.compile(
    r"^\s*(discussion|comment|interpretation)\b", re.IGNORECASE
)
# Discussion 뒤에 붙는 섹션. 여기서 멈춘다.
_STOP_TITLE = re.compile(
    r"^\s*(conflict|competing|acknowledg|funding|author contribution|"
    r"supplementary|reference|abbreviation|ethic|data availability|"
    r"declaration|appendix)",
    re.IGNORECASE,
)
_DROP_TAGS = {"xref", "table-wrap", "fig", "graphic", "media", "supplementary-material",
              "table", "disp-formula", "inline-formula", "label"}


def _node_text(node: ET.Element) -> str:
    """xref(참고문헌 위첨자)·표·그림을 버리고 사람이 읽는 텍스트만 남긴다."""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        tag = child.tag.split("}")[-1]
        if tag not in _DROP_TAGS:
            parts.append(_node_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _clean(text: str) -> str:
    text = text.replace(" ", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_discussion(xml_bytes: bytes) -> dict[str, Any] | None:
    """Discussion 섹션 본문 + 소제목 목록을 돌려준다.

    소제목(예: BMJ 의 'Strengths and limitations of study')은 그 자체가
    저널 서식 정책 변수이므로 따로 보관한다. 본문에서는 제거한다.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    body = root.find(".//body")
    if body is None:
        return None

    def section_title(section: ET.Element) -> str:
        title = section.find("title")
        return _clean(_node_text(title)) if title is not None else ""

    target: ET.Element | None = None
    for section in body.findall("sec"):
        sec_type = (section.get("sec-type") or "").lower()
        title = section_title(section)
        if "discussion" in sec_type or _DISCUSSION_TITLE.match(title):
            target = section
            break
    if target is None:
        return None

    headings: list[str] = []
    paragraphs: list[str] = []

    def walk(section: ET.Element, depth: int) -> None:
        for child in section:
            tag = child.tag.split("}")[-1]
            if tag == "title":
                text = _clean(_node_text(child))
                if depth > 0 and text:
                    headings.append(text)
                continue
            if tag == "p":
                text = _clean(_node_text(child))
                if text:
                    paragraphs.append(text)
            elif tag == "sec":
                if _STOP_TITLE.match(section_title(child)):
                    continue
                walk(child, depth + 1)
            elif tag == "list":
                for item in child.iter():
                    if item.tag.split("}")[-1] == "p":
                        text = _clean(_node_text(item))
                        if text:
                            paragraphs.append(text)

    walk(target, 0)
    text = "\n\n".join(paragraphs)
    if len(text.split()) < 120:  # Discussion 이라기엔 너무 짧다. 파싱 실패로 본다.
        return None
    return {"discussion_text": text, "section_headings": headings}


# ------------------------------------------------------------------- 저널 IF


def load_journal_table(path: Path) -> dict[str, dict[str, Any]]:
    """journals.csv: journal,journal_short,jif,metric_source,tier"""
    table: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return table
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("journal") or "").strip()
            if not name:
                continue
            table[_journal_key(name)] = {
                "journal": name,
                "journal_short": (row.get("journal_short") or "").strip(),
                "jif": float(row["jif"]) if (row.get("jif") or "").strip() else None,
                "metric_source": (row.get("metric_source") or "").strip(),
                "tier": (row.get("tier") or "").strip(),
            }
    return table


def _journal_key(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    name = re.sub(r"\b(the|journal|of|and|for)\b", " ", name)
    return re.sub(r"\s+", " ", name).strip()


# ------------------------------------------------------------------ 레코드화


def to_record(meta: dict[str, Any], extracted: dict[str, Any],
              journals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    journal_name = meta.get("journalInfo", {}).get("journal", {}).get("title", "") or ""
    matched = journals.get(_journal_key(journal_name), {})
    text = extracted["discussion_text"]
    return {
        "paper_id": meta.get("pmcid") or meta.get("id"),
        "pmid": meta.get("pmid"),
        "pmcid": meta.get("pmcid"),
        "doi": meta.get("doi"),
        "title": meta.get("title"),
        "first_author": meta.get("authorString", "").split(",")[0],
        "journal": journal_name,
        "journal_short": matched.get("journal_short") or meta.get("journalInfo", {}).get("journal", {}).get("isoabbreviation", ""),
        "jif": matched.get("jif"),
        "jif_source": matched.get("metric_source"),
        "tier": matched.get("tier"),
        "year": int(meta["pubYear"]) if str(meta.get("pubYear", "")).isdigit() else None,
        "pub_types": meta.get("pubTypeList", {}).get("pubType", []),
        "is_open_access": meta.get("isOpenAccess"),
        "cited_by_count": meta.get("citedByCount"),
        "text_source": "oa_jats",
        "section_headings": extracted["section_headings"],
        "has_explicit_limitation_heading": any(
            re.search(r"limitation|strength", h, re.IGNORECASE)
            for h in extracted["section_headings"]
        ),
        "discussion_text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
    }


# ---------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="studyset JSON (query/limit/filters 를 담는다)")
    parser.add_argument("--query", help="Europe PMC query. --config 없을 때 사용")
    parser.add_argument("--limit", type=int, default=200, help="검색 상한")
    parser.add_argument("--journals", default=str(HERE / "journals.csv"))
    parser.add_argument("--out", default=str(HERE / "corpus" / "collected.jsonl"))
    parser.add_argument("--require-jif", action="store_true",
                        help="journals.csv 에 IF 가 있는 저널만 남긴다")
    parser.add_argument("--min-words", type=int, default=200)
    args = parser.parse_args()

    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        query = config["query"]
        limit = int(config.get("limit", args.limit))
        require_jif = bool(config.get("require_jif", args.require_jif))
    elif args.query:
        query, limit, require_jif = args.query, args.limit, args.require_jif
    else:
        parser.error("--config 또는 --query 중 하나는 있어야 한다")
        return 2

    # OA 전문이 있는 것만 검색 단계에서 거른다.
    if "OPEN_ACCESS" not in query.upper():
        query = f"({query}) AND (OPEN_ACCESS:Y)"

    journals = load_journal_table(Path(args.journals))
    print(f"journal table: {len(journals)} 개 저널", file=sys.stderr)
    print(f"query: {query}", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_path = out_path.with_name("missing_oa.csv")

    kept = 0
    seen_dois: set[str] = set()
    missing: list[dict[str, str]] = []

    with out_path.open("w", encoding="utf-8") as out_fh:
        for index, meta in enumerate(search(query, limit=limit), start=1):
            title = (meta.get("title") or "")[:70]
            pmcid = meta.get("pmcid")
            journal_name = meta.get("journalInfo", {}).get("journal", {}).get("title", "") or ""
            matched = journals.get(_journal_key(journal_name))

            if require_jif and not (matched and matched.get("jif") is not None):
                continue
            doi = (meta.get("doi") or "").lower()
            if doi and doi in seen_dois:
                continue

            reason = ""
            record = None
            if not pmcid:
                reason = "no_pmcid"
            else:
                xml_bytes = fetch_fulltext_xml(pmcid)
                if xml_bytes is None:
                    reason = "no_fulltext_xml"
                else:
                    extracted = extract_discussion(xml_bytes)
                    if extracted is None:
                        reason = "no_discussion_section"
                    elif len(extracted["discussion_text"].split()) < args.min_words:
                        reason = "discussion_too_short"
                    else:
                        record = to_record(meta, extracted, journals)

            if record is None:
                missing.append({
                    "reason": reason, "pmid": meta.get("pmid", ""),
                    "pmcid": pmcid or "", "doi": meta.get("doi", ""),
                    "journal": journal_name, "year": str(meta.get("pubYear", "")),
                    "title": meta.get("title", ""),
                })
                print(f"  [{index:>4}] skip({reason:<22}) {journal_name[:34]:<34} {title}", file=sys.stderr)
                continue

            if doi:
                seen_dois.add(doi)
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            jif = record.get("jif")
            jif_text = f"IF={jif:>5}" if jif is not None else "IF=  ?  "
            print(f"  [{index:>4}] KEEP {jif_text} w={record['word_count']:>5} "
                  f"{record['journal_short'][:22]:<22} {title}", file=sys.stderr)

    if missing:
        with missing_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(missing[0].keys()))
            writer.writeheader()
            writer.writerows(missing)

    print(f"\n수집 완료: {kept} 건 -> {out_path}", file=sys.stderr)
    print(f"미수집   : {len(missing)} 건 -> {missing_path}", file=sys.stderr)
    print("\n미수집 목록은 버리지 마라. 고IF 저널일수록 여기 몰린다 = 선택편향.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
