"""journal_frequency.py — 어떤 저널의 IF 를 채워야 하는지 알려준다.

절차는 이렇다. IF 표를 먼저 만들지 마라. 순서가 반대다.

    1. collect_discussions.py 를 --require-jif 없이 돌린다 (전량 수집)
    2. 이 스크립트로 저널별 논문 수를 센다
    3. 상위 N개 저널만 JCR/SJR 에서 IF 를 찾아 journals.csv 에 적는다
    4. collect 를 --require-jif 로 다시 돌리거나, 이미 받은 corpus 를 필터한다

30개 저널만 채우면 보통 논문의 70~80% 가 커버된다. 200개를 다 채울 필요는 없다.

사용법
    python journal_frequency.py --corpus corpus/collected.jsonl
    python journal_frequency.py --corpus corpus/collected.jsonl --template todo.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(HERE / "corpus" / "collected.jsonl"))
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--template", help="IF 를 채워야 할 저널 목록을 CSV 로 쓴다")
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    has_jif: dict[str, bool] = {}
    for line in Path(args.corpus).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        journal = record.get("journal") or "(unknown)"
        counts[journal] += 1
        has_jif[journal] = has_jif.get(journal, False) or record.get("jif") is not None

    total = sum(counts.values())
    covered = sum(count for journal, count in counts.items() if has_jif.get(journal))
    print(f"논문 {total}건 · 저널 {len(counts)}개 · IF 매칭됨 {covered}건 "
          f"({covered / total * 100:.1f}%)\n")

    print(f"{'n':>4}  {'누적%':>6}  IF  저널")
    print("-" * 78)
    running = 0
    todo: list[dict[str, str]] = []
    for journal, count in counts.most_common(args.top):
        running += count
        mark = "O " if has_jif.get(journal) else "  "
        print(f"{count:>4}  {running / total * 100:>5.1f}%  {mark}  {journal[:60]}")
        if not has_jif.get(journal):
            todo.append({"journal": journal, "journal_short": "", "jif": "",
                         "metric_source": "", "tier": "", "n_papers": str(count)})

    if args.template and todo:
        path = Path(args.template)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(todo[0].keys()))
            writer.writeheader()
            writer.writerows(todo)
        print(f"\nIF 를 채워야 할 저널 {len(todo)}개 -> {path}")
        print("jif 열을 채운 뒤 journals.csv 에 이어붙여라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
